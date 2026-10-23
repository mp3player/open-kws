"""Resolve bundled model names without accessing the network."""
import json
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent / "weights"
CATALOG = json.loads((WEIGHTS / "models.json").read_text(encoding="utf-8"))
DEFAULT_MODEL = CATALOG["default_model"]


def model_names():
    return tuple(CATALOG["models"])


def checkpoint_for(name=DEFAULT_MODEL):
    return WEIGHTS / CATALOG["models"][name]["checkpoint"]


def metadata_for(name):
    return WEIGHTS / CATALOG["models"][name]["metadata"]
