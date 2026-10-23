"""CPU file-based enrollment and cosine matching without training dependencies."""
import argparse
import json
import math
from pathlib import Path

import soundfile
import torch
from torch.nn import functional

from audio import AudioConfig, LogMel
from content_model import ContentEncoder, encoder_fingerprint
from model_catalog import DEFAULT_MODEL, checkpoint_for, model_names

DEFAULT_CHECKPOINT = checkpoint_for()
PIPELINE = "content-logmel-mean-l2-no-trim-v1"


class KeywordEncoder:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, threads=2):
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.encoder = ContentEncoder(**payload["encoder_config"]).cpu().eval()
        self.encoder.load_state_dict(payload["encoder_state"], strict=True)
        self.fingerprint = encoder_fingerprint(self.encoder)
        self.audio_config = AudioConfig(**payload["audio_config"])
        self.frontend = LogMel(self.audio_config)
        calibration = payload["calibration"]
        if (calibration["encoder_fingerprint"] != self.fingerprint
                or calibration["dataset_fingerprint"] != payload["dataset_fingerprint"]
                or calibration["protocol"]["split"] != "validation"):
            raise ValueError("Checkpoint calibration does not match its encoder")
        self.threshold = float(calibration["threshold"])
        if not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("Invalid calibration threshold")
        self.profile = dict(format_version=1, encoder_fingerprint=self.fingerprint,
                            audio_config=self.audio_config.to_dict(), pipeline=PIPELINE)

    @torch.inference_mode()
    def embed(self, samples, sample_rate, channels_first=True):
        features = self.frontend(samples, sample_rate, channels_first)
        features = features - features.mean(0, keepdim=True)
        vector = self.encoder.embed(features[None], torch.tensor([len(features)]))[0]
        if not torch.isfinite(vector).all() or vector.norm() < .9:
            raise ValueError("Invalid embedding")
        return vector

    def embed_file(self, path):
        samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
        return self.embed(samples, rate, channels_first=False)

    def load_templates(self, path):
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        if saved["profile"] != self.profile:
            raise ValueError("Templates use another model/frontend; register again")
        templates = saved["templates"]
        if not isinstance(templates, dict) or not 1 <= len(templates) <= 20:
            raise ValueError("Expected 1–20 registered keywords")
        for label, vectors in templates.items():
            validate_label(label)
            if not isinstance(vectors, list) or not 1 <= len(vectors) <= 5:
                raise ValueError("Expected 1–5 recordings per keyword")
            tensor = torch.tensor(vectors, dtype=torch.float32)
            if (tensor.shape != (len(vectors), self.encoder.config["embedding_dim"])
                    or not torch.isfinite(tensor).all()
                    or not torch.allclose(tensor.norm(dim=-1), torch.ones(len(vectors)), atol=1e-4)):
                raise ValueError("Invalid enrollment embeddings")
        return templates

    def enroll(self, path, label, recordings):
        validate_label(label)
        path = Path(path)
        templates = self.load_templates(path) if path.exists() else {}
        current = templates.get(label, [])
        if not recordings or len(current) + len(recordings) > 5 or (label not in templates and len(templates) >= 20):
            raise ValueError("Use 1–5 independent recordings per keyword, at most 20 keywords")
        vectors = [self.embed_file(recording).tolist() for recording in recordings]
        templates[label] = [*current, *vectors]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(dict(profile=self.profile, templates=templates), ensure_ascii=False), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
        return dict(keyword=label, samples=len(templates[label]), keywords=len(templates))

    def match(self, templates, recording, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Threshold must be finite and between -1 and 1")
        vector = self.embed_file(recording)
        ranked = sorted((dict(label=label, cosine=float(functional.normalize(torch.tensor(vectors).mean(0), dim=0) @ vector))
                         for label, vectors in templates.items()), key=lambda item: item["cosine"], reverse=True)
        return dict(audio=str(recording), scores=ranked, accepted=ranked[0]["cosine"] >= threshold,
                    threshold=threshold, scope="Whole-file matching; not continuous-listening false alarms/hour")


def validate_label(label):
    if not isinstance(label, str) or not label.strip() or len(label) > 40 or any(ord(char) < 32 for char in label):
        raise ValueError("Keyword label must contain 1–40 characters without control characters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", choices=model_names(), help=f"Bundled model; default {DEFAULT_MODEL}")
    models.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    compare = commands.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("query", type=Path)
    compare.add_argument("--threshold", type=float)
    enroll = commands.add_parser("enroll")
    enroll.add_argument("--keyword", required=True)
    enroll.add_argument("--audio", type=Path, nargs="+", required=True)
    enroll.add_argument("--templates", type=Path, default=Path("templates.json"))
    match = commands.add_parser("match")
    match.add_argument("--templates", type=Path, default=Path("templates.json"))
    match.add_argument("--audio", type=Path, nargs="+", required=True)
    match.add_argument("--threshold", type=float)
    args = parser.parse_args()
    try:
        encoder = KeywordEncoder(args.checkpoint or checkpoint_for(args.model or DEFAULT_MODEL), args.threads)
        if args.command == "info":
            result = dict(encoder_fingerprint=encoder.fingerprint, encoder_config=encoder.encoder.config,
                          audio_config=encoder.audio_config.to_dict(), threshold=encoder.threshold,
                          parameters=sum(parameter.numel() for parameter in encoder.encoder.parameters()), device="cpu")
        elif args.command == "compare":
            threshold = encoder.threshold if args.threshold is None else args.threshold
            if not math.isfinite(threshold) or not -1 <= threshold <= 1:
                raise ValueError("Threshold must be finite and between -1 and 1")
            score = float(encoder.embed_file(args.reference) @ encoder.embed_file(args.query))
            result = dict(cosine=score, threshold=threshold, accepted=score >= threshold,
                          scope="Whole-file pair matching; cosine is not a probability")
        elif args.command == "enroll":
            result = encoder.enroll(args.templates, args.keyword.strip(), args.audio)
        else:
            templates = encoder.load_templates(args.templates)
            result = [encoder.match(templates, recording, args.threshold) for recording in args.audio]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
