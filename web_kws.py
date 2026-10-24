"""Phone microphone enrollment and rolling-window content matching on a LAN."""
import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import logging
import math
from pathlib import Path
import plistlib
import secrets
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import numpy as np
import torch
from torch.nn import functional

from audio import AudioConfig, LogMel
from content_model import ContentEncoder, encoder_fingerprint
from model_catalog import DEFAULT_MODEL, checkpoint_for, model_names

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "web_kws"


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ContentMatcher:
    def __init__(self, checkpoint, profiles, window_seconds=2., min_rms=.002, hop_seconds=.1):
        if not 1 <= window_seconds <= 4 or not 0 <= min_rms <= .1:
            raise ValueError("Window must be 1–4 seconds; RMS gate must be in [0, 0.1]")
        if not .05 <= hop_seconds <= window_seconds:
            raise ValueError("Listening hop must be between 0.05 seconds and the window length")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not {"encoder_config", "encoder_state", "audio_config"} <= payload.keys():
            raise ValueError("Use a content-classification encoder.pt, not an old CTC checkpoint")
        self.encoder = ContentEncoder(**payload["encoder_config"]).cpu().eval()
        self.encoder.load_state_dict(payload["encoder_state"], strict=True)
        self.fingerprint = encoder_fingerprint(self.encoder)
        calibration = payload.get("calibration", {})
        if (calibration.get("encoder_fingerprint") != self.fingerprint
                or calibration.get("dataset_fingerprint") != payload.get("dataset_fingerprint")
                or calibration.get("protocol", {}).get("split") != "validation"):
            raise ValueError("Checkpoint needs matching validation calibration")
        self.threshold = float(calibration["threshold"])
        if not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("Invalid validation threshold")
        self.audio_config = AudioConfig(**payload["audio_config"])
        self.frontend = LogMel(self.audio_config)
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.min_rms = min_rms
        self.model_name = Path(checkpoint).name
        self.dimensions = self.encoder.config["embedding_dim"]
        self.profile = dict(encoder_fingerprint=self.fingerprint, audio_config=self.audio_config.to_dict(),
                            window_seconds=window_seconds, pipeline="content-logmel-mean-l2-no-trim-v1")
        identity = hashlib.sha256(json.dumps(self.profile, sort_keys=True).encode()).hexdigest()
        self.profile_path = Path(profiles) / f"{identity}.json"
        self.lock = threading.RLock()
        self.computation = threading.Lock()
        self.templates = {}
        if self.profile_path.exists():
            saved = json.loads(self.profile_path.read_text())
            if saved["profile"] != self.profile:
                raise ValueError("Registration profile does not match this model/pipeline")
            for label, vectors in saved["templates"].items():
                self.validate_label(label)
                if not 1 <= len(vectors) <= 5:
                    raise ValueError("Invalid saved enrollment count")
                tensor = torch.tensor(vectors, dtype=torch.float32)
                if (tensor.shape != (len(vectors), self.dimensions) or not torch.isfinite(tensor).all()
                        or not torch.allclose(tensor.norm(dim=-1), torch.ones(len(vectors)), atol=1e-4)):
                    raise ValueError("Invalid saved enrollment embeddings")
                self.templates[label] = vectors
            if len(self.templates) > 20:
                raise ValueError("Too many saved keywords")

    @staticmethod
    def validate_label(label):
        if not isinstance(label, str) or not label.strip() or len(label) > 40 or any(ord(char) < 32 for char in label):
            raise RequestError("Keyword labels must contain 1–40 characters without control characters")

    def state(self):
        with self.lock:
            return dict(model=self.model_name, encoder_type=self.encoder.encoder_type,
                        activation=self.encoder.config.get("cnn_activation", "gelu"),
                        fingerprint=self.fingerprint, dimensions=self.dimensions, threshold=self.threshold,
                        window_seconds=self.window_seconds, hop_seconds=self.hop_seconds, min_rms=self.min_rms,
                        keywords=[dict(label=label, samples=len(vectors)) for label, vectors in self.templates.items()],
                        max_samples=5, max_keywords=20,
                        scope="Whole-window cosine matching; offline threshold is not calibrated for continuous listening")

    def save_templates(self, templates):
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.profile_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(profile=self.profile, templates=templates), ensure_ascii=False))
        temporary.chmod(0o600)
        temporary.replace(self.profile_path)
        self.templates = templates

    @torch.inference_mode()
    def embed_pcm(self, content, rate):
        if not 8000 <= rate <= 96000 or len(content) % 2:
            raise RequestError("Expected mono PCM16 audio at 8–96 kHz")
        expected = round(rate * self.window_seconds)
        if abs(len(content) // 2 - expected) > 1:
            raise RequestError(f"Audio must be {self.window_seconds:g} seconds long; please record again")
        started = time.perf_counter()
        samples = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768.
        rms = float(np.sqrt(np.mean(samples ** 2)))
        info = dict(rms=rms, peak=float(np.max(np.abs(samples))), seconds=len(samples) / rate,
                    sample_rate=rate, inference_ms=0.)
        if rms < self.min_rms:
            return None, info
        features = self.frontend(samples, rate)
        features = features - features.mean(0, keepdim=True)
        vector = self.encoder.embed(features[None], torch.tensor([len(features)]))[0]
        if not torch.isfinite(vector).all() or vector.norm() < .9:
            raise RequestError("Invalid model output; please record again")
        info["inference_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return vector, info

    def process(self, content, rate, label=None, threshold=None):
        if label is not None:
            self.validate_label(label)
            label = label.strip()
        threshold = self.threshold if threshold is None else threshold
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise RequestError("The similarity threshold must be between -1 and 1")
        if not self.computation.acquire(blocking=False):
            raise RequestError("The computer is processing the previous window; please retry", 429)
        try:
            vector, info = self.embed_pcm(content, rate)
            if label is not None:
                if vector is None:
                    raise RequestError("Enrollment audio is too quiet; move closer to the phone microphone and record again")
                with self.lock:
                    current = self.templates.get(label, [])
                    if len(current) >= 5:
                        raise RequestError("At most five recordings per keyword; delete its templates before starting over")
                    if label not in self.templates and len(self.templates) >= 20:
                        raise RequestError("At most 20 keywords can be enrolled")
                    updated = dict(self.templates)
                    updated[label] = [*current, vector.tolist()]
                    self.save_templates(updated)
                return dict(state=self.state(), audio=info)
            with self.lock:
                items = list(self.templates.items())
            if not items:
                raise RequestError("Enroll at least one keyword first", 409)
            if vector is None:
                return dict(scores=[], label=None, score=None, candidate=False, threshold=threshold,
                            low_energy=True, audio=info)
            prototypes = torch.stack([functional.normalize(torch.tensor(vectors).mean(0), dim=0)
                                      for _, vectors in items])
            values = (prototypes @ vector).tolist()
            ranked = sorted([dict(label=item[0], cosine=score) for item, score in zip(items, values)],
                            key=lambda row: row["cosine"], reverse=True)
            return dict(scores=ranked, label=ranked[0]["label"], score=ranked[0]["cosine"],
                        candidate=ranked[0]["cosine"] >= threshold, threshold=threshold,
                        low_energy=False, audio=info)
        finally:
            self.computation.release()

    def delete(self, label):
        with self.lock:
            if label is None:
                updated = {}
            else:
                self.validate_label(label)
                if label not in self.templates:
                    raise RequestError("Keyword not found", 404)
                updated = {name: vectors for name, vectors in self.templates.items() if name != label}
            self.save_templates(updated)
        return self.state()


class TLSHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler, context):
        self.context = context
        super().__init__(address, handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        try:
            return self.context.wrap_socket(connection, server_side=True, do_handshake_on_connect=False), address
        except OSError:
            connection.close()
            raise

    def process_request_thread(self, request, address):
        try:
            request.do_handshake()
        except OSError as error:
            logging.getLogger(__name__).warning("TLS connection from %s failed: %s", address[0], error)
            self.shutdown_request(request)
            return
        super().process_request_thread(request, address)


def handler_for(matcher, token=None, allowed_hosts=()):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass

        def reply(self, content, mime="application/json; charset=utf-8", status=200):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "microphone=(self), camera=()")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                             "style-src 'self' 'unsafe-inline'; media-src 'self' blob:; "
                             "connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            hosts = {"localhost", "127.0.0.1", self.server.server_address[0], *allowed_hosts}
            authorities = {f"{host}:{self.server.server_port}" for host in hosts if host != "0.0.0.0"}
            if self.headers.get("Host") not in authorities or len(self.headers.get_all("Host", [])) != 1:
                raise RequestError("Use the LAN address displayed by the computer", 403)
            if token:
                provided = self.headers.get("X-KWS-Token", "")
                if not hmac.compare_digest(provided.encode(), token.encode()):
                    raise RequestError("Invalid access token; reopen the full link printed in the terminal", 401)
            if self.command == "POST":
                scheme = "https" if isinstance(self.connection, ssl.SSLSocket) else "http"
                if self.headers.get("Origin") != f"{scheme}://{self.headers.get('Host')}":
                    raise RequestError("Requests must originate from this web page", 403)

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.mjs": ("app.mjs", "text/javascript; charset=utf-8"),
                      "/capture-worklet.js": ("capture-worklet.js", "text/javascript; charset=utf-8")}
            try:
                if path == "/api/state":
                    self.authorized()
                    self.reply(dict(matcher.state(), token_required=bool(token)))
                elif path in assets:
                    filename, mime = assets[path]
                    self.reply((ASSETS / filename).read_bytes(), mime)
                else:
                    self.reply({"error": "Not found"}, status=404)
            except RequestError as error:
                self.reply({"error": str(error)}, status=error.status)

        def do_POST(self):
            try:
                self.authorized()
                url = urlsplit(self.path)
                if url.path not in {"/api/enroll", "/api/detect", "/api/delete"}:
                    raise RequestError("Not found", 404)
                if self.headers.get("Transfer-Encoding"):
                    raise RequestError("Chunked requests are not supported")
                length = int(self.headers.get("Content-Length", "0"))
                limit = 4096 if url.path == "/api/delete" else 96000 * 4 * 2 + 2
                if not 0 < length <= limit:
                    raise RequestError("Invalid upload size", 413)
                content = self.rfile.read(length)
                if len(content) != length:
                    raise RequestError("Incomplete audio upload")
                if url.path == "/api/delete":
                    if self.headers.get_content_type() != "application/json":
                        raise RequestError("Expected JSON", 415)
                    data = json.loads(content)
                    if not isinstance(data, dict) or "label" not in data:
                        raise RequestError("Specify a keyword to delete, or set label to null to delete all")
                    self.reply(dict(state=matcher.delete(data["label"])))
                    return
                if self.headers.get_content_type() != "application/octet-stream":
                    raise RequestError("Expected PCM16 binary audio", 415)
                query = parse_qs(url.query, keep_blank_values=True)
                if any(len(values) != 1 for values in query.values()):
                    raise RequestError("Repeated query parameter")
                rate = int(query.get("rate", ["0"])[0])
                label = query.get("label", [""])[0] if url.path == "/api/enroll" else None
                threshold = float(query.get("threshold", [str(matcher.threshold)])[0])
                self.reply(matcher.process(content, rate, label, threshold))
            except RequestError as error:
                self.reply({"error": str(error)}, status=error.status)
            except (ValueError, TypeError, KeyError, UnicodeError):
                self.reply({"error": "Invalid request parameters or audio format"}, status=400)
            except OSError:
                self.reply({"error": "Could not read the request or save templates; check the computer logs and disk"}, status=500)

    return Handler


def prepare_tls(folder, address):
    address = str(ipaddress.ip_address(address))
    if ":" in address:
        raise ValueError("Use an IPv4 LAN address")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)
    ca = folder / "local-ca.crt"
    ca_key = folder / "local-ca.key"
    certificate = folder / "server.crt"
    key = folder / "server.key"
    def openssl(*arguments):
        return subprocess.run(["openssl", *map(str, arguments)], check=True, capture_output=True).stdout
    if ca.exists() != ca_key.exists():
        raise ValueError("Incomplete local CA; restore its key/certificate or use a new --tls-dir")
    if not ca.exists():
        openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365", "-sha256",
                "-keyout", ca_key, "-out", ca, "-subj", "/CN=KWS Local Test CA",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    openssl("x509", "-in", ca, "-checkend", "86400", "-noout")
    with tempfile.TemporaryDirectory(dir=folder) as temporary:
        temporary = Path(temporary)
        request = temporary / "server.csr"
        temporary_key = temporary / "server.key"
        temporary_cert = temporary / "server.crt"
        openssl("req", "-new", "-newkey", "rsa:2048", "-nodes", "-sha256", "-keyout", temporary_key,
                "-out", request, "-subj", "/CN=KWS LAN Test")
        extensions = temporary / "extensions.conf"
        extensions.write_text(f"subjectAltName=IP:{address},IP:127.0.0.1,DNS:localhost\n"
                              "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                              "extendedKeyUsage=serverAuth\n")
        openssl("x509", "-req", "-in", request, "-CA", ca, "-CAkey", ca_key, "-CAcreateserial",
                "-out", temporary_cert, "-days", "30", "-sha256", "-extfile", extensions)
        temporary_key.replace(key)
        temporary_cert.replace(certificate)
    key.chmod(0o600)
    ca_key.chmod(0o600)
    return certificate, key, ca


def certificate_profile(certificate):
    der = ssl.PEM_cert_to_DER_cert(certificate.read_text())
    return plistlib.dumps(dict(PayloadContent=[dict(PayloadType="com.apple.security.root", PayloadVersion=1,
        PayloadIdentifier="local.kws.test.root", PayloadUUID=str(uuid.uuid4()), PayloadDisplayName="KWS Local Test CA",
        PayloadContent=der)], PayloadType="Configuration", PayloadVersion=1, PayloadIdentifier="local.kws.test",
        PayloadUUID=str(uuid.uuid4()), PayloadDisplayName="KWS LAN Test Certificate",
        PayloadDescription="For local KWS testing on this computer only. Remove this profile after testing."))


def setup_handler(certificate, app_url=None):
    class SetupHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/ca.crt":
                content, mime = certificate.read_bytes(), "application/x-x509-ca-cert"
            elif path == "/ca.mobileconfig":
                content, mime = certificate_profile(certificate), "application/x-apple-aspen-config"
            elif path == "/":
                app_link = (f"<p><a href='{escape(app_url, quote=True)}'>Open the KWS demo (HTTPS)</a></p>"
                            if app_url else "")
                content = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,initial-scale=1'><title>KWS Phone Setup</title>"
                    "<body style='font:18px/1.8 system-ui;max-width:680px;margin:40px auto;padding:20px'>"
                    "<h1>Connect your phone microphone</h1><p>This page provides the local test certificate. Use the full HTTPS application link printed in the terminal for recording.</p>"
                    + app_link +
                    "<h2>Android</h2><p><a href='/ca.crt'>Download the KWS test CA certificate</a> and install it using the CA certificate option in your system settings.</p>"
                    "<h2>iPhone</h2><p>Using Safari, <a href='/ca.mobileconfig'>download the certificate profile</a> and install it under Settings > General > "
                    "VPN &amp; Device Management. Then open General > About > Certificate Trust Settings and enable full trust for KWS Local Test CA.</p>"
                    "<p><b>Install only on your trusted LAN, after verifying the certificate SHA256 fingerprint printed on the computer. Remove the certificate/profile after testing.</b></p>"
                    "<p>Finally, open the printed HTTPS link in system Chrome or Safari, not an embedded in-app browser.</p></body></html>").encode()
                mime = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
    return SetupHandler


def lan_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        try:
            connection.connect(("192.0.2.1", 9))
            return connection.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", choices=model_names(), help=f"Bundled model; default {DEFAULT_MODEL}")
    models.add_argument("--checkpoint", type=Path)
    parser.add_argument("--address", default=lan_address(), help="IPv4 LAN address advertised in the certificate")
    parser.add_argument("--host", help="Bind address; defaults to --address, not all interfaces")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--setup-port", type=int, default=8767)
    parser.add_argument("--profiles-dir", type=Path, default=ROOT / "recordings/web_kws")
    parser.add_argument("--tls-dir", type=Path, default=ROOT / ".web_kws_tls")
    parser.add_argument("--window-seconds", type=float, default=2.)
    parser.add_argument("--hop-seconds", type=float, default=.1, help="Matching interval; default 0.1 seconds")
    parser.add_argument("--min-rms", type=float, default=.002)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--require-token", action="store_true", help="Opt into a random access token; default is trusted-LAN access")
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or checkpoint_for(args.model or DEFAULT_MODEL)
    if not 1 <= args.port <= 65535 or not 1 <= args.setup_port <= 65535 or args.port == args.setup_port or args.threads < 1:
        parser.error("Invalid ports or thread count")
    torch.set_num_threads(args.threads)
    matcher = ContentMatcher(args.checkpoint, args.profiles_dir, args.window_seconds, args.min_rms, args.hop_seconds)
    certificate, key, ca = prepare_tls(args.tls_dir, args.address)
    token = secrets.token_urlsafe(24) if args.require_token else None
    host = args.host or args.address
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate, key)
    server = TLSHTTPServer((host, args.port), handler_for(matcher, token, (args.address,)), context)
    setup = ThreadingHTTPServer((host, args.setup_port),
                               setup_handler(ca, f"https://{args.address}:{args.port}/"))
    threading.Thread(target=setup.serve_forever, daemon=True).start()
    ca_fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(ca.read_text())).hexdigest()
    print(f"CPU model: {args.checkpoint} ({matcher.fingerprint[:12]})", flush=True)
    print(f"Audio window: {args.window_seconds:g} s; matching interval: {args.hop_seconds:g} s", flush=True)
    print(f"Phone certificate setup: http://{args.address}:{args.setup_port}/", flush=True)
    print(f"CA SHA256: {ca_fingerprint}", flush=True)
    suffix = f"#token={token}" if token else ""
    print(f"Phone app: https://{args.address}:{args.port}/{suffix}", flush=True)
    if not token:
        print("Trusted LAN mode: no access token. Other devices on this LAN can access and change enrollments.", flush=True)
    print("Enrollment embeddings are stored locally; raw microphone audio is not saved. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        setup.shutdown()
        setup.server_close()
        server.server_close()


if __name__ == "__main__":
    main()
