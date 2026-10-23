"""Verify the bundled model and deterministic synthetic inference reference."""
import hashlib
import argparse
import json

import numpy as np
import torch

from inference import KeywordEncoder
from model_catalog import checkpoint_for, metadata_for, model_names


def verify_model(name):
    checkpoint = checkpoint_for(name)
    metadata = json.loads(metadata_for(name).read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("Checkpoint checksum failed")
    encoder = KeywordEncoder(checkpoint)
    if encoder.fingerprint != metadata["encoder_fingerprint"]:
        raise ValueError("Encoder fingerprint failed")
    positions = np.arange(32000, dtype=np.int32)
    pcm = ((positions % 97 - 48) * 256 + (positions % 31 - 15) * 64).astype("<i2")
    vector = encoder.embed(pcm.astype(np.float32) / 32768., 16000)
    reference = torch.tensor(metadata["synthetic_reference"]["embedding"])
    torch.testing.assert_close(vector, reference, rtol=1e-4, atol=2e-5)
    if abs(float(vector.norm()) - 1) > 1e-5:
        raise ValueError("Embedding normalization failed")
    return dict(verified=True, model=name, checkpoint=checkpoint.name, sha256=digest,
                encoder_fingerprint=encoder.fingerprint, dimensions=len(vector),
                max_reference_error=float((vector - reference).abs().max()),
                scope="Synthetic wiring/regression check, not speech accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_names(), help="Verify one model; default verifies all bundled models")
    args = parser.parse_args()
    names = (args.model,) if args.model else model_names()
    print(json.dumps([verify_model(name) for name in names], indent=2))


if __name__ == "__main__":
    main()
