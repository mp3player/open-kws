import hashlib
import http.client
import json
from pathlib import Path
import plistlib
import socket
import ssl
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

import numpy as np
import torch

from audio import AudioConfig, LogMel
from content_model import ContentEncoder, encoder_fingerprint
from web_kws import ContentMatcher, RequestError, TLSHTTPServer, certificate_profile, handler_for, prepare_tls, setup_handler

torch.set_num_threads(2)


class WebKWSTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.manual_seed(4)
        self.encoder = ContentEncoder(encoder_type="cnn", cnn_activation="relu", cnn_channels=16,
                                      cnn_blocks=2, embedding_dim=8, dropout=0).eval()
        self.checkpoint = self.root / "encoder.pt"
        self.payload = dict(encoder_config=self.encoder.config, encoder_state=self.encoder.state_dict(),
                            audio_config=AudioConfig().to_dict(), dataset_fingerprint="fixture",
                            calibration=dict(threshold=.5, encoder_fingerprint=encoder_fingerprint(self.encoder),
                                             dataset_fingerprint="fixture", protocol=dict(split="validation")))
        torch.save(self.payload, self.checkpoint)
        self.matcher = ContentMatcher(self.checkpoint, self.root / "profiles")
        times = np.arange(32000) / 16000
        self.content = (np.sin(times * 440 * 2 * np.pi) * 5000).astype("<i2").tobytes()

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_training_frontend_and_normalized_embeddings(self):
        vector, info = self.matcher.embed_pcm(self.content, 16000)
        samples = np.frombuffer(self.content, dtype="<i2").astype(np.float32) / 32768.
        features = LogMel(AudioConfig())(samples, 16000)
        features -= features.mean(0, keepdim=True)
        with torch.inference_mode():
            expected = self.encoder.embed(features[None], torch.tensor([len(features)]))[0]
        torch.testing.assert_close(vector, expected)
        self.assertAlmostEqual(float(vector.norm()), 1, places=5)
        self.assertEqual(info["seconds"], 2)
        with self.assertRaises(RequestError):
            self.matcher.embed_pcm(self.content, 48000)
        with self.assertRaises(RequestError):
            self.matcher.embed_pcm(b"a", 16000)

    def test_enrollment_persistence_scoring_gating_limits_and_deletion(self):
        with self.assertRaises(RequestError):
            self.matcher.process(self.content, 16000)
        for _ in range(3):
            self.matcher.process(self.content, 16000, label="\u5c0f\u7231\u540c\u5b66")
        detection = self.matcher.process(self.content, 16000)
        self.assertTrue(detection["candidate"])
        self.assertGreater(detection["score"], .999)
        self.assertEqual(detection["label"], "\u5c0f\u7231\u540c\u5b66")
        restored = ContentMatcher(self.checkpoint, self.root / "profiles")
        self.assertEqual(restored.state()["keywords"], [dict(label="\u5c0f\u7231\u540c\u5b66", samples=3)])
        quiet = restored.process(bytes(len(self.content)), 16000)
        self.assertTrue(quiet["low_energy"])
        self.assertFalse(quiet["candidate"])
        with self.assertRaises(RequestError):
            restored.process(bytes(len(self.content)), 16000, label="\u80cc\u666f")
        for _ in range(2):
            restored.process(self.content, 16000, label="\u5c0f\u7231\u540c\u5b66")
        with self.assertRaises(RequestError):
            restored.process(self.content, 16000, label="\u5c0f\u7231\u540c\u5b66")
        for label in ("", "  ", "bad\nname", "x" * 41):
            with self.assertRaises(RequestError):
                restored.process(self.content, 16000, label=label)
        for threshold in (float("nan"), float("inf"), 1.1):
            with self.assertRaises(RequestError):
                restored.process(self.content, 16000, threshold=threshold)
        self.assertEqual(restored.delete("\u5c0f\u7231\u540c\u5b66")["keywords"], [])
        restored.process(self.content, 16000, label="\u53e6\u4e00\u4e2a\u8bcd")
        self.assertEqual(restored.delete(None)["keywords"], [])

    def test_profiles_are_isolated_and_bad_calibration_is_rejected(self):
        self.matcher.process(self.content, 16000, label="\u8bcd\u7ec4")
        different_window = ContentMatcher(self.checkpoint, self.root / "profiles", window_seconds=3)
        self.assertFalse(different_window.state()["keywords"])
        payload = dict(self.payload, calibration=dict(self.payload["calibration"], encoder_fingerprint="wrong"))
        torch.save(payload, self.root / "wrong.pt")
        with self.assertRaisesRegex(ValueError, "calibration"):
            ContentMatcher(self.root / "wrong.pt", self.root / "profiles")
        self.matcher.computation.acquire()
        try:
            with self.assertRaises(RequestError) as error:
                self.matcher.process(self.content, 16000)
            self.assertEqual(error.exception.status, 429)
        finally:
            self.matcher.computation.release()

    def test_listening_interval_is_configurable_without_changing_templates(self):
        self.assertEqual(self.matcher.state()["hop_seconds"], .1)
        self.matcher.process(self.content, 16000, label="\u6d4b\u8bd5")
        slower = ContentMatcher(self.checkpoint, self.root / "profiles", hop_seconds=.5)
        self.assertEqual(slower.state()["hop_seconds"], .5)
        self.assertEqual(slower.profile_path, self.matcher.profile_path)
        self.assertEqual(slower.state()["keywords"], [dict(label="\u6d4b\u8bd5", samples=1)])
        for interval in (0, .01, 2.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "hop"):
                ContentMatcher(self.checkpoint, self.root / "profiles", hop_seconds=interval)

    def test_api_auth_origin_payload_and_static_allowlist(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.matcher, "secret"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port
        origin = f"http://127.0.0.1:{port}"
        def request(method, path, body=None, token="secret", source=origin, content_type="application/octet-stream"):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request(method, path, body=body, headers={"X-KWS-Token": token, "Origin": source,
                               "Content-Type": content_type})
            response = connection.getresponse()
            value = response.read()
            status = response.status
            connection.close()
            return status, value
        try:
            self.assertEqual(request("GET", "/api/state", token="")[0], 401)
            self.assertEqual(request("POST", "/api/enroll?rate=16000&label=test", self.content,
                                     source="https://evil.example")[0], 403)
            self.assertEqual(request("GET", "/api/state")[0], 200)
            for path in ("/", "/app.mjs", "/capture-worklet.js"):
                self.assertEqual(request("GET", path, token="")[0], 200)
            for path in ("/../web_kws.py", "/.web_kws_tls/local-ca.key", "/recordings/secret.wav"):
                self.assertEqual(request("GET", path)[0], 404)
            self.assertEqual(request("POST", "/api/enroll?rate=16000&label=test", self.content)[0], 200)
            status, value = request("POST", "/api/detect?rate=16000", self.content)
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(value)["candidate"])
            self.assertEqual(request("POST", "/api/detect?rate=0", self.content)[0], 400)
            self.assertEqual(request("POST", "/api/detect?rate=16000&threshold=NaN", self.content)[0], 400)
            self.assertEqual(request("POST", "/api/delete", b'{}', content_type="application/json")[0], 400)
            self.assertEqual(request("POST", "/api/delete", b'{"label":null}', content_type="application/json")[0], 200)
            self.assertEqual(request("POST", "/api/detect?rate=16000", self.content)[0], 409)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_tokenless_lan_access_still_checks_origin_and_host(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.matcher))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port
        origin = f"http://127.0.0.1:{port}"
        def request(method, path, body=None, source=origin, host=None):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            headers = {"Origin": source, "Content-Type": "application/octet-stream"}
            if host is not None:
                headers["Host"] = host
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            value = response.read()
            status = response.status
            connection.close()
            return status, value
        try:
            status, value = request("GET", "/api/state")
            self.assertEqual(status, 200)
            self.assertFalse(json.loads(value)["token_required"])
            self.assertEqual(request("POST", "/api/enroll?rate=16000&label=test", self.content)[0], 200)
            self.assertEqual(request("POST", "/api/detect?rate=16000", self.content)[0], 200)
            self.assertEqual(request("POST", "/api/enroll?rate=16000&label=bad", self.content,
                                     source="https://evil.example")[0], 403)
            self.assertEqual(request("GET", "/api/state", host=f"evil.example:{port}")[0], 403)
            self.assertEqual(request("POST", "/api/enroll?rate=16000&label=bad", self.content,
                                     source=f"http://evil.example:{port}", host=f"evil.example:{port}")[0], 403)
            self.assertEqual(self.matcher.state()["keywords"], [dict(label="test", samples=1)])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_tls_chain_and_idle_handshakes_do_not_block_other_clients(self):
        certificate, key, ca = prepare_tls(self.root / "tls", "127.0.0.1")
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        original_ca = hashlib.sha256(ca.read_bytes()).hexdigest()
        prepare_tls(self.root / "tls", "127.0.0.1")
        self.assertEqual(hashlib.sha256(ca.read_bytes()).hexdigest(), original_ca)
        mobile = plistlib.loads(certificate_profile(ca))
        self.assertEqual(mobile["PayloadContent"][0]["PayloadContent"], ssl.PEM_cert_to_DER_cert(ca.read_text()))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
        server = TLSHTTPServer(("127.0.0.1", 0), handler_for(self.matcher, "secret"), context)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        idle = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        partial = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        partial.sendall(b"\x16\x03\x01\x00\x10")
        try:
            with self.assertLogs("web_kws", level="WARNING") as logs:
                plain = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                try:
                    plain.request("GET", "/")
                    with self.assertRaises(http.client.RemoteDisconnected):
                        plain.getresponse()
                finally:
                    plain.close()
            self.assertTrue(any("HTTP_REQUEST" in message for message in logs.output))
            client = http.client.HTTPSConnection("127.0.0.1", server.server_port,
                        context=ssl.create_default_context(cafile=str(ca)), timeout=2)
            client.request("GET", "/api/state", headers={"X-KWS-Token": "secret"})
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            client.close()
        finally:
            idle.close()
            partial.close()
            server.shutdown()
            server.server_close()
            thread.join()
        app_url = f"https://127.0.0.1:{server.server_port}/"
        setup = ThreadingHTTPServer(("127.0.0.1", 0), setup_handler(ca, app_url))
        thread = threading.Thread(target=setup.serve_forever, daemon=True)
        thread.start()
        try:
            for path, expected in (("/", 200), ("/ca.crt", 200), ("/ca.mobileconfig", 200), ("/local-ca.key", 404)):
                client = http.client.HTTPConnection("127.0.0.1", setup.server_port, timeout=5)
                client.request("GET", path)
                response = client.getresponse()
                self.assertEqual(response.status, expected)
                content = response.read()
                if path == "/":
                    self.assertIn(f"href='{app_url}'".encode(), content)
                client.close()
        finally:
            setup.shutdown()
            setup.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
