import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile
import torch

from inference import KeywordEncoder
from model_catalog import checkpoint_for, model_names
from verify import verify_model
from web_kws import ContentMatcher


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = KeywordEncoder()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        positions = np.arange(32000, dtype=np.int32)
        self.samples = ((positions % 97 - 48) * 256 + (positions % 31 - 15) * 64).astype(np.float32) / 32768.
        self.audio = self.root / "synthetic.wav"
        soundfile.write(self.audio, self.samples, 16000, subtype="PCM_16")

    def tearDown(self):
        self.temporary.cleanup()

    def test_file_decoding_and_unit_embeddings(self):
        expected = self.encoder.embed(self.samples, 16000)
        torch.testing.assert_close(self.encoder.embed_file(self.audio), expected)
        stereo = np.stack((self.samples, self.samples), axis=1)
        torch.testing.assert_close(self.encoder.embed(stereo, 16000, channels_first=False), expected)
        self.assertEqual(tuple(expected.shape), (128,))
        self.assertAlmostEqual(float(expected.norm()), 1, places=5)
        with self.assertRaises(ValueError):
            self.encoder.embed(np.zeros(20), 16000)

    def test_registration_matching_and_model_isolation(self):
        path = self.root / "templates.json"
        self.encoder.enroll(path, "\u6d4b\u8bd5\u8bcd", [self.audio])
        templates = self.encoder.load_templates(path)
        result = self.encoder.match(templates, self.audio)
        self.assertTrue(result["accepted"])
        self.assertGreater(result["scores"][0]["cosine"], .9999)
        for threshold in (float("nan"), float("inf"), 1.1):
            with self.assertRaises(ValueError):
                self.encoder.match(templates, self.audio, threshold)
        for _ in range(4):
            self.encoder.enroll(path, "\u6d4b\u8bd5\u8bcd", [self.audio])
        with self.assertRaises(ValueError):
            self.encoder.enroll(path, "\u6d4b\u8bd5\u8bcd", [self.audio])
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["profile"]["encoder_fingerprint"] = "another-model"
        path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "another model"):
            self.encoder.load_templates(path)

    def test_registration_does_not_save_partial_failures(self):
        path = self.root / "templates.json"
        with self.assertRaises((OSError, RuntimeError)):
            self.encoder.enroll(path, "\u6d4b\u8bd5\u8bcd", [self.audio, self.root / "missing.wav"])
        self.assertFalse(path.exists())

    def test_documented_training_dataset_inventory(self):
        root = Path(__file__).resolve().parents[1]
        summary = json.loads((root / "data_summary.json").read_text(encoding="utf-8"))
        rows = [row for row in summary["by_primary_source"] if row["split"] == 0]
        self.assertEqual({row["source"] for row in rows}, {"thchs30", "fsc", "slurp", "wenetphrase", "gigaphrase"})
        self.assertEqual(sum(row["classes"] for row in rows), 97346)
        self.assertEqual(sum(row["samples"] for row in rows), 1558850)
        for document in ("README.md", "TRAINING.md", "DATA_SOURCES.md"):
            content = (root / document).read_text(encoding="utf-8")
            for dataset in ("THCHS-30", "Fluent Speech Commands", "SLURP", "WenetPhrase", "GigaPhrase"):
                self.assertIn(dataset, content)

    def test_all_bundled_models_and_cross_model_template_rejection(self):
        path = self.root / "templates.json"
        self.encoder.enroll(path, "\u6d4b\u8bd5\u8bcd", [self.audio])
        self.assertEqual(set(model_names()), {"cnn256", "gru160", "gru96"})
        for name in model_names():
            with self.subTest(model=name):
                self.assertTrue(verify_model(name)["verified"])
                encoder = KeywordEncoder(checkpoint_for(name))
                vector = encoder.embed_file(self.audio)
                self.assertEqual(tuple(vector.shape), (128,))
                self.assertAlmostEqual(float(vector.norm()), 1, places=5)
                matcher = ContentMatcher(checkpoint_for(name), self.root / "web-templates")
                pcm = np.round(self.samples * 32768).astype("<i2").tobytes()
                web_vector, info = matcher.embed_pcm(pcm, 16000)
                torch.testing.assert_close(web_vector, vector)
                self.assertEqual(info["seconds"], 2)
                self.assertEqual(matcher.state()["hop_seconds"], .1)
                if name != "cnn256":
                    self.assertEqual(encoder.encoder.encoder_type, "gru")
                    with self.assertRaisesRegex(ValueError, "another model"):
                        encoder.load_templates(path)


if __name__ == "__main__":
    unittest.main()
