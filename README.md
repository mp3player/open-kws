# Open KWS

Register your own wake phrase and test it with a phone microphone or audio files. Inference runs locally on the computer's CPU; no GPU or CUDA is needed.

## Quick start: phone demo

Run these commands from the repository root. Python 3.11+ and the `openssl` executable are required.

### 1. Install

Linux CPU example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Use compatible PyTorch and torchaudio builds. For other platforms, see the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 2. Start

```bash
python kws.py
```

The program selects the bundled `cnn256` model, detects a LAN address, and prints two links:

- **Phone certificate setup (HTTP, port 8767):** open this first.
- **Phone app (HTTPS, port 8766):** open this after certificate setup.

The phone and computer must be on the same trusted LAN. If the detected address belongs to a VPN or the wrong network adapter, specify the computer's Wi-Fi/Ethernet IPv4 address:

```bash
python kws.py web --address YOUR_COMPUTER_LAN_IP
```

To try the smaller 408k-parameter CNN96 (about 1.66 MB of FP32 inference weights):

```bash
python kws.py web --model cnn96-relu-last
```

The default remains CNN256; the smaller model trades some validation accuracy for size. Register fresh templates when changing models.

### 3. Register and listen

1. Open the printed certificate setup link in system Chrome or Safari. Compare the CA fingerprint with the terminal output before installing the test CA. On iPhone, also enable full trust under **Settings > General > About > Certificate Trust Settings**.
2. Follow the setup page's **HTTPS** app link and allow microphone access.
3. Enter a keyword label. Wait for **Start speaking**, record your phrase, and replay it to confirm your voice is present. Three independently recorded takes are recommended.
4. Select **Start listening**, keep the browser in the foreground, and keep the phone unlocked.

The label is only a name: changing its text does not change what the model listens for. Use **Ctrl+C** in the terminal to stop the server.

The default rolling window is two seconds, evaluated every **100 ms**. The first match needs a full window. Two consecutive hits trigger an event, followed by a 1.5-second cooldown and low-score rearming. Busy inference/network periods skip old windows rather than queuing them.

**Privacy and security:** the phone sends microphone audio to this computer, not a third-party cloud. The server stores enrollment vectors, not raw recordings. Tokenless access is the default: other devices on the LAN can access and modify templates. Use a trusted LAN only, optionally enable `--require-token`, and never expose this development server publicly. Certificate keys in `.web_kws_tls/` and templates in `recordings/` must stay private. Remove the test CA from the phone after testing. System trust stores and firewall rules are not changed automatically.

## Test audio files instead

No phone or microphone is required if you already have WAV/FLAC files.

Compare two recordings:

```bash
python kws.py compare enroll.wav query.wav
```

Register a keyword and test several files:

```bash
python kws.py enroll --keyword "My wake phrase" --audio enroll1.wav enroll2.wav enroll3.wav
python kws.py match --audio query.wav other_phrase.wav
```

Enrollment saves `templates.json` in the current directory, containing vectors and labels, not audio. Repeating enrollment appends takes, with at most five per keyword and 20 keywords. To keep separate registrations, pass the same `--templates PATH` to both commands:

```bash
python kws.py enroll --model gru160 --templates gru160_templates.json --keyword "My wake phrase" --audio enroll.wav
python kws.py match --model gru160 --templates gru160_templates.json --audio query.wav
```

Use the same model for enrollment and matching. Switching models requires fresh enrollment; file and web templates are separate formats and cannot be interchanged.

Audio is averaged to mono, resampled to 16 kHz, and processed as one complete spoken-content unit. No trimming, fixed-duration padding, or loudness normalization is applied to file tests. Do not compare a long ambient recording as though it were a short phrase. File tests have no web energy gate or consecutive-hit logic.

Cosine similarity is **not a probability**. Each model uses its own validation-calibrated starting threshold. Higher thresholds generally reduce accepts, including both false and correct accepts. Calibrate with independent recordings representative of your setting:

```bash
python kws.py match --audio query.wav --threshold 0.5
```

## Choose a model

```bash
python kws.py models
python kws.py web --model cnn256-relu-best
python kws.py info --model gru160
python kws.py compare enroll.wav query.wav --model cnn256-relu-last
python kws.py web --model cnn96-relu-last
python kws.py compare enroll.wav query.wav --model cnn96-relu-background-last
python kws.py verify
```

`verify` checks all bundled weights, fingerprints, and deterministic synthetic outputs. Use `python kws.py verify --model gru160` to check one model. This checks installation and inference wiring, **not speech accuracy**. Weights are loaded locally with `torch.load(weights_only=True)`; no model download is performed.

| Model name | Encoder / objective | Epoch | Encoder parameters | Validation threshold |
|---|---|---:|---:|---:|
| `cnn256` (default) | Noncausal GELU CNN + ArcFace / CB | 11 | 2,659,649 | 0.322372 |
| `cnn256-relu-best` | Noncausal ReLU CNN + ArcFace / CB | 26 | 2,659,649 | 0.284015 |
| `cnn256-relu-last` | Same ReLU CNN configuration | 30 | 2,659,649 | 0.282165 |
| `cnn96-relu-best` | Noncausal ReLU CNN96 + ArcFace / CB | 16 | 408,049 | 0.441902 |
| `cnn96-relu-last` | Same CNN96 configuration | 20 | 408,049 | 0.436371 |
| `cnn96-relu-background-best` | CNN96 background fine-tune | 3 additional | 408,049 | 0.432430 |
| `cnn96-relu-background-last` | Same fine-tune; identical encoder to best | 3 additional | 408,049 | 0.432430 |
| `gru160` | Two-layer unidirectional GRU160 + Softmax / CB | 24 | 317,505 | 0.333835 |
| `gru96` | Two-layer unidirectional GRU96 + Softmax | 16 | 136,513 | 0.553505 |

### Model folders

Each folder contains its weights, metadata, and training configuration. Select models by name; you do not need to type checkpoint paths.

```text
weights/
  models.json                  Model names and relative paths
  cnn256/
    epoch11.pt
    model_info.json
    training_config.json
  cnn256_relu/
    best.pt                    Validation-best checkpoint, epoch 26
    last.pt                    Final checkpoint, epoch 30
    best_model_info.json
    last_model_info.json
    training_config.json
  cnn96_relu/
    best.pt                    Validation-best original CNN96, epoch 16
    last.pt                    Final original CNN96, epoch 20
    best_model_info.json
    last_model_info.json
    training_config.json
  cnn96_relu_background/
    best.pt                    Background fine-tune, additional epoch 3
    last.pt                    Same encoder as best in this completed run
    best_model_info.json
    last_model_info.json
    training_config.json
  gru160/
    epoch24.pt
    model_info.json
    training_config.json
  gru96/
    epoch16.pt
    model_info.json
    training_config.json
```

CNN256 ReLU and original CNN96 best/last are distinct weights, with separate fingerprints, thresholds, and enrollment templates. The background fine-tune's best and last select the same additional epoch 3, so their encoders and thresholds are identical. Best is selected by minimum validation source-macro EER, not maximum TAR. Only encoder/frontend/calibration assets are included, not classification heads or optimizer states. The default remains the evaluated GELU `cnn256` baseline.

Updating from the earlier flat `weights/` layout? The original five model names and weight contents are unchanged; CNN96 adds four named exports. Use `--model NAME` instead of an old checkpoint path. Existing templates remain valid when you keep using the same model and audio configuration.

## Troubleshooting and options

| Problem | What to check |
|---|---|
| Phone cannot connect | Same LAN; correct computer address; firewall permits ports 8766/8767; no Wi-Fi client isolation. Try the printed **HTTP setup link** first. |
| Empty response on port 8766 | Open the full `https://...` link, not HTTP. |
| Microphone unavailable | Install and trust the test CA, use HTTPS in system Chrome/Safari, and grant microphone permission. Ordinary LAN HTTP pages cannot access live microphone capture. |
| Meter moves but voice is missing | Replay the take; check the displayed actual input and microphone selector; disconnect unwanted Bluetooth/headset devices. The meter also moves before **Start speaking**, but preparation audio is not enrolled. |
| Address already in use | Stop the old demo or choose another pair of ports. |
| Templates use another model | Register again with the selected model, or choose the matching template file. |
| Missing Python dependency | Activate the virtual environment and repeat installation with that environment's `python -m pip`. |

Examples:

```bash
python kws.py web --port 8866 --setup-port 8867
python kws.py web --model gru160 --hop-seconds 0.2
python kws.py web --require-token
python kws.py --help
python kws.py web --help
python kws.py enroll --help
```

Restart the server to change models. Web profiles are isolated automatically by model fingerprint and audio-window configuration. Shorter matching intervals increase scan frequency, not necessarily accuracy; overlapping windows are not independent confirmations.

Legacy entry points (`inference.py`, `web_kws.py`, and `verify.py`) remain available. File-inference model options work before or after the subcommand.

## How it works

```text
Enrollment audio -> log-Mel -> GRU/CNN -> 128-D unit vector
Query audio      -> same encoder      -> cosine match against templates
```

Training classifies **exact spoken content**, not intent or speaker identity. CNN uses ArcFace with class-balanced cross-entropy; GRU uses Softmax, with class balancing for GRU160. Inference needs no classification head, transcript, pinyin, or CTC. Registering a new phrase does not update model parameters. Multiple enrollment vectors are averaged and L2-normalized before matching.

FP32 checkpoint sizes are approximately 10.17 MiB for CNN256, 1.58 MiB for CNN96, 1.22 MiB for GRU160, and 0.53 MiB for GRU96. ReLU replaces GELU activations but does not reduce parameter count or convolution MACs. CNN96 reduces channel width; it still uses LayerNorm, attention pooling, Tanh, and Softmax. Weight size is not runtime RAM usage.

This is an experimental content-similarity release for PC CPU testing, **not** general ASR, speaker authentication, or a validated production always-on wake-word engine. Near-sounding phrases, unregistered speech, noise, and distribution shifts can cause false accepts or misses. Arbitrary Chinese phrases are not guaranteed to work reliably. No quantization, MCU export, or MCU hardware validation has been completed.

## Training datasets

All released configurations share the speech pool built from **five corpus families** below. CNN96 additionally uses one background training class; its background counts are separate from this speech-only table.

| Dataset | Language | Retained training classes | Training recordings / segments |
|---|---|---:|---:|
| [THCHS-30](https://www.openslr.org/18/) | Mandarin Chinese | 899 | 12,010 |
| [Fluent Speech Commands (FSC)](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) | English | 222 | 27,181 |
| [SLURP](https://github.com/pswietojanski/slurp) | English | 1,427 | 10,757 |
| [WenetPhrase](https://github.com/aizhiqi-work/MM-KWS#data-pipeline-for-wenetphrase) | Mandarin Chinese | 7,274 | 114,379 |
| [GigaPhrase-1000](https://github.com/aizhiqi-work/GigaPhrase-1000) | English | 87,524 | 1,394,523 |
| **Total** | | **97,346** | **1,558,850** |

Counts describe the filtered training index, **not original dataset sizes**. Classes are attributed to their primary source. The mixture is dominated by English GigaPhrase data and is not language-balanced. GigaPhrase includes material derived from LibriSpeech/GigaSpeech; those are upstream sources, not additional independently prepared datasets.

CNN96's original run adds 1,002 training background clips, mostly synthetic white/pink noise and silence. The two nominal ESC-50 training clips and one validation clip were later found to be entirely zero: they are not effective real-background coverage. The background fine-tune replaces that background pool with 1,043 training clips (43 real clips from 20 sources plus 1,000 synthetic clips). Real audio comes from selected [ESC-50](https://github.com/karolpiczak/ESC-50) categories via [ashraq/esc50](https://huggingface.co/datasets/ashraq/esc50); the recorded upstream license is CC BY-NC 3.0. This does not grant commercial use or establish weight-redistribution clearance. See [DATA_SOURCES.md](DATA_SOURCES.md).

Validation contains 5,772 content classes / 113,747 segments; test contains 6,021 classes / 151,051 segments. These are available pools, not evaluation trial counts.

See [TRAINING.md](TRAINING.md) for the training process, filtering, and split construction, [DATA_SOURCES.md](DATA_SOURCES.md) for provenance and restrictions, and `data_summary.json` for aggregate statistics. Training code, dataset audio, per-recording transcripts, personal recordings, and unfinished experiments are not distributed.

## Recorded offline results

Existing fixed-episode results use 100 enrolled classes with one recording each, 100 unregistered background classes, and seed 42. The test episode contains 478 positive comparisons, 97,322 negative comparisons, and 500 background queries. Source utterances are disjoint within the episode. Each model uses its own threshold calibrated on validation and frozen for testing.

| Model | Test global EER | Test source-macro EER | Test TAR | Test pair FAR | Background accept rate |
|---|---:|---:|---:|---:|---:|
| CNN256 GELU, epoch 11 | 2.929% | 3.170% | 94.561% | 0.868% | 49.400% |
| GRU160, epoch 24 | 3.766% | 2.473% | 92.050% | 0.975% | 44.400% |
| CNN256 ReLU best, epoch 26 | 3.138% | 3.367% | 96.025% | 1.005% | 50.200% |
| CNN256 ReLU last, epoch 30 | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| GRU96, epoch 16 | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| CNN96 ReLU best / last | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| CNN96 background fine-tune best / last | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated |

| Model | Validation source-macro EER | Validation TAR |
|---|---:|---:|
| CNN256 GELU, epoch 11 | 0.741% | 98.109% |
| GRU160, epoch 24 | 1.198% | 96.008% |
| GRU96, epoch 16 | 4.621% | 69.748% |
| CNN256 ReLU best, epoch 26 | 0.652453% | 98.319% |
| CNN256 ReLU last, epoch 30 | 0.654215% | 98.950% |
| CNN96 ReLU best, epoch 16 | 2.256065% | 93.067% |
| CNN96 ReLU last, epoch 20 | 2.257540% | 93.277% |
| CNN96 background fine-tune best / last, additional epoch 3 | 2.244099% | 93.067% |

The CNN96 background fine-tune did **not** demonstrate improved background rejection. On the same v3 validation pool, the original epoch-20 model accepted 4/11 real background clips and the fine-tuned model also accepted 4/11. Including synthetic noise/silence, acceptance changed from 77.48% to 87.39%; thresholds were independently calibrated on speech validation (0.436371 vs 0.432430). These are comparisons against 100 speech templates without an energy gate, not always-on false alarms/hour. The real-background validation set has only two independent source recordings. Original v2 and new v3 background pools must not be compared as identical episodes. No independent fixed-test result is claimed for the CNN96 exports.

EER scans thresholds; TAR uses the fixed validation threshold. Macro EER averages per-source EERs, not per-class EERs. **Pair FAR is not false alarms per hour.** Background acceptance means an unregistered-content query is accepted by any of 100 templates, not ambient-silence false alarms. These high background acceptance rates show that open-set rejection remains insufficient for production claims.

CNN is not better than GRU on every metric. Architecture, objective, and augmentation changed together, so this is not an isolated architecture ablation. ReLU epoch 26 improves test TAR over the earlier epoch-10 snapshot (91.213% to 96.025%), but background acceptance increases (39.800% to 50.200%); this is not an unconditional improvement.

Both ReLU validation EERs round to 0.0065 in training logs. Epoch 26 has slightly lower EER; epoch 30 has higher TAR. Their tiny EER difference does not establish a meaningful real-world advantage. Epoch-26 test metrics must not be attributed to epoch 30. Detailed protocols, calibration, checksums, and synthetic references are stored beside each model.

## Licensing and development

**Release draft:** code and weight licenses have not been selected, and upstream data restrictions still require review. Do not assume commercial use or redistribution is authorized. In particular, some training corpora have non-commercial or academic-only restrictions. See [DATA_SOURCES.md](DATA_SOURCES.md).

This inference directory runs independently of the private training project. File checksums are recorded in `SHA256SUMS`. To run regression tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
node --test tests/test_web_kws.mjs
```

Python web tests require `openssl`. Node tests are optional; running the demo does not require Node. `content_model.py` contains inference encoders, `audio.py` the frontend, `model_catalog.py` local model selection, and `web_kws/` browser assets. This is not a fully reproducible training release.
