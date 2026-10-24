"""Local keyword spotting: run the phone demo or test your audio files."""
import argparse
import importlib
import json

from model_catalog import CATALOG, DEFAULT_MODEL, metadata_for, model_names


def print_models():
    print(f"{'Model':<20} {'Encoder':<9} {'Epoch':>5} {'Parameters':>12} {'Threshold':>10}")
    for name in model_names():
        model = CATALOG["models"][name]
        metadata = json.loads(metadata_for(name).read_text(encoding="utf-8"))
        print(f"{name:<20} {model['encoder_type']:<9} {model['epoch']:>5} "
              f"{metadata['parameters']:>12,} {metadata['validation']['threshold']:>10.6f}")
    print(f"Default: {DEFAULT_MODEL}. Select another model with --model NAME.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="No command starts the phone demo. Use COMMAND --help for options.")
    commands = parser.add_subparsers(dest="command")
    descriptions = {
        "web": "Start the phone-microphone web demo (default)",
        "models": "List bundled models without loading PyTorch",
        "info": "Show the selected model's configuration and threshold",
        "verify": "Verify bundled weights and synthetic reference outputs",
        "compare": "Compare two complete audio files",
        "enroll": "Register a keyword from audio files",
        "match": "Match audio files against registered keywords",
    }
    for name, description in descriptions.items():
        commands.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command is None and remaining:
        parser.error("Put options after a command, for example: web --model gru160")
    command = args.command or "web"
    if command == "models":
        if remaining == ["--help"]:
            commands.choices[command].print_help()
        elif remaining:
            parser.error(f"Unrecognized models options: {' '.join(remaining)}")
        else:
            print_models()
        return
    module = "web_kws" if command == "web" else "verify" if command == "verify" else "inference"
    try:
        entrypoint = importlib.import_module(module).main
    except ModuleNotFoundError as error:
        if error.name.split(".")[0] not in {"numpy", "torch", "torchaudio", "soundfile"}:
            raise
        parser.exit(2, f"Missing dependency: {error.name}. Follow the README installation steps "
                       "using this Python environment, then try again.\n")
    entrypoint(remaining if module != "inference" else [command, *remaining])


if __name__ == "__main__":
    main()
