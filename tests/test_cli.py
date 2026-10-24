import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import soundfile

import inference
import kws
from model_catalog import checkpoint_for, metadata_for, model_names


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_help_and_model_listing_without_dependencies_or_repository_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            for command in (["--help"], ["models"], ["models", "--help"]):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, "-S", str(ROOT / "kws.py"), *command],
                                            cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if command == ["models"]:
                        for name in model_names():
                            self.assertIn(name, result.stdout)
                        self.assertIn("Default: cnn256", result.stdout)

    def test_default_starts_web_and_commands_forward_arguments(self):
        cases = [([], "web_kws", []),
                 (["web", "--model", "gru160"], "web_kws", ["--model", "gru160"]),
                 (["verify", "--model", "gru96"], "verify", ["--model", "gru96"]),
                 (["info", "--model", "cnn256-relu-last"], "inference",
                  ["info", "--model", "cnn256-relu-last"])]
        for arguments, module, expected in cases:
            with self.subTest(arguments=arguments):
                entrypoint = mock.Mock()
                with mock.patch("kws.importlib.import_module", return_value=SimpleNamespace(main=entrypoint)) as load:
                    kws.main(arguments)
                load.assert_called_once_with(module)
                entrypoint.assert_called_once_with(expected)

    def test_missing_dependency_has_installation_hint(self):
        with mock.patch("kws.importlib.import_module", side_effect=ModuleNotFoundError(name="torch")):
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as stopped:
                kws.main(["info"])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("README installation steps", error.getvalue())

    def test_models_rejects_unknown_options(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            kws.main(["models", "--typo"])
        self.assertEqual(stopped.exception.code, 2)

    def test_legacy_model_options_work_on_either_side_of_subcommand(self):
        fingerprints = []
        for arguments in (["--model", "gru160", "--threads", "1", "info"],
                          ["info", "--model", "gru160", "--threads", "1"]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                inference.main(arguments)
            info = json.loads(output.getvalue())
            self.assertEqual(info["parameters"], 317505)
            fingerprints.append(info["encoder_fingerprint"])
        self.assertEqual(fingerprints[0], fingerprints[1])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            inference.main(["--model", "gru160", "info", "--checkpoint", "unused.pt"])
        self.assertEqual(stopped.exception.code, 2)

    def test_models_have_colocated_metadata_and_training_config(self):
        for name in model_names():
            with self.subTest(model=name):
                folder = checkpoint_for(name).parent
                self.assertNotEqual(folder, ROOT / "weights")
                self.assertEqual(metadata_for(name).parent, folder)
                self.assertTrue((folder / "training_config.json").is_file())
        self.assertEqual(checkpoint_for("cnn256-relu-best").name, "best.pt")
        self.assertEqual(checkpoint_for("cnn256-relu-last").name, "last.pt")

    def test_unified_audio_file_workflow_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positions = np.arange(32000, dtype=np.int32)
            samples = ((positions % 97 - 48) * 256 + (positions % 31 - 15) * 64).astype(np.float32) / 32768.
            soundfile.write(root / "sample.wav", samples, 16000, subtype="PCM_16")
            commands = [
                ["compare", "sample.wav", "sample.wav", "--model", "gru160"],
                ["enroll", "--keyword", "Wake phrase", "--audio", "sample.wav", "--model", "gru160"],
                ["match", "--audio", "sample.wav", "--model", "gru160"],
            ]
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(ROOT / "kws.py"), *command],
                                            cwd=root, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    payload = json.loads(result.stdout)
                    if command[0] == "compare":
                        self.assertTrue(payload["accepted"])
                    elif command[0] == "enroll":
                        self.assertEqual(payload["samples"], 1)
                        self.assertTrue((root / "templates.json").is_file())
                    else:
                        self.assertTrue(payload[0]["accepted"])
                        self.assertEqual(payload[0]["scores"][0]["label"], "Wake phrase")


if __name__ == "__main__":
    unittest.main()
