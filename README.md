# User-defined keyword spotting with GRU/CNN content embeddings

A standalone inference release containing **five fixed checkpoints**: GRU96, GRU160, GELU CNN256, and both the validation-best and final checkpoints of the completed ReLU CNN256 run. Audio-file tests and a phone-microphone web demo are included.

Training code, training audio, personal recordings, enrollment templates, server addresses, private certificate keys, and models that are still being trained are **not included**. This directory runs independently of the parent project.

**Release draft:** code and weight licenses have not been selected, and upstream data restrictions still require review. Do not assume commercial use or redistribution is authorized. See [Data sources and licensing](DATA_SOURCES.md).

## Available models

| `--model` | Encoder / training objective | Epoch | Encoder parameters | Validation threshold | Role |
|---|---|---:|---:|---:|---|
| `cnn256` (default) | Noncausal GELU CNN + ArcFace / CB | 11 | 2,659,649 | 0.322372 | Evaluated CNN baseline |
| `gru160` | Two-layer unidirectional GRU160 + Softmax / CB | 24 | 317,505 | 0.333835 | Evaluated GRU baseline |
| `gru96` | Two-layer unidirectional GRU96 + Softmax | 16 | 136,513 | 0.553505 | Early lightweight baseline; weaker validation results |
| `cnn256-relu-best` | Noncausal ReLU CNN + ArcFace / CB | 26 | 2,659,649 | 0.284015 | Best validation source-macro EER of the completed run |
| `cnn256-relu-last` | Same ReLU CNN configuration | 30 | 2,659,649 | 0.282165 | Final completed epoch; retained separately |

The three older baselines remain unchanged. The completed ReLU run includes both best and last, not every intermediate epoch. Ongoing training snapshots and unsuccessful CTC, reconstruction, Transformer, and DS-CNN experiments are excluded. The default remains `cnn256`; publishing new weights does not silently change the default model.

`cnn_relu_best_epoch26.pt` was selected by minimum validation macro EER, while `cnn_relu_last_epoch30.pt` is the final epoch, with a slightly better validation TAR. They are distinct weights, with separate fingerprints and enrollment templates. Each contains only the encoder, frontend configuration, and calibration metadata, not the training head or optimizer state.

The model catalog is `weights/models.json`. Each model has its own checksum, fingerprint, calibration threshold, and metadata.

## Training datasets

All released configurations used the same filtered content-class index built from these **five corpus families**:

| Dataset | Language | Retained training classes | Training recordings / segments |
|---|---|---:|---:|
| [THCHS-30](https://www.openslr.org/18/) | Mandarin Chinese | 899 | 12,010 |
| [Fluent Speech Commands (FSC)](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) | English | 222 | 27,181 |
| [SLURP](https://github.com/pswietojanski/slurp) | English | 1,427 | 10,757 |
| [WenetPhrase](https://github.com/aizhiqi-work/MM-KWS#data-pipeline-for-wenetphrase) | Mandarin Chinese | 7,274 | 114,379 |
| [GigaPhrase-1000](https://github.com/aizhiqi-work/GigaPhrase-1000) | English | 87,524 | 1,394,523 |
| **Total** | | **97,346** | **1,558,850** |

Counts describe our filtered training index, **not the original dataset sizes**. Classes are attributed to their primary source. The mixture is dominated by English GigaPhrase data and is not language-balanced. GigaPhrase includes material derived from LibriSpeech/GigaSpeech; those are upstream sources, not additional independently prepared datasets in this release.

Validation contains 5,772 content classes / 113,747 segments; test contains 6,021 classes / 151,051 segments. These are available pools, not the number of trials used in the reported evaluation episodes.

See [DATA_SOURCES.md](DATA_SOURCES.md) for upstream links, the Hugging Face mirrors actually used, and license restrictions; [TRAINING.md](TRAINING.md) for filtering and split construction; and `data_summary.json` for machine-readable aggregate statistics. No dataset audio or per-recording transcripts are distributed.

## Method and limitations

```text
Enrollment audio -> log-Mel -> selected GRU/CNN -> 128-D unit vector
Query audio      -> same encoder             -> cosine match against templates
```

Training classifies exact spoken content. CNN uses ArcFace with class-balanced cross-entropy; GRU uses Softmax, with class balancing for GRU160. Inference requires no classification head, text, pinyin, or CTC.

A keyword name is a display label, not a speech-recognition target. New keywords do not update model parameters. Multiple enrollment unit vectors are averaged and L2-normalized. Cosine similarity is **not a probability**.

- FP32 checkpoint sizes: approximately 10.17 MiB for CNN256, 1.22 MiB for GRU160, and 0.53 MiB for GRU96; all produce 128-D embeddings.
- Intended for content-similarity experiments and PC CPU demos, not general ASR or speaker authentication.
- No quantization, MCU export, or MCU hardware validation has been completed.
- Near-sounding phrases, unregistered speech, and distribution shifts can cause false accepts or misses. Arbitrary Chinese phrases are not guaranteed to work reliably.
- This is not a validated always-on production wake-word engine or a security-sensitive authentication mechanism.

## Installation and verification

Python 3.11+ is required. No GPU or CUDA is needed. Example for Linux CPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify.py
python inference.py info
```

Use mutually compatible PyTorch and torchaudio builds. For other platforms, consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

`verify.py` checks every bundled model's SHA256, fingerprint, and deterministic synthetic output. It is a regression check, **not a speech-accuracy benchmark**. Use `python verify.py --model gru160` to verify one model. Weights are loaded locally with `torch.load(weights_only=True)`; no model download is performed.

## Audio-file tests

Provide your own WAV/FLAC files. Multichannel audio is averaged to mono and resampled to 16 kHz. Files are not trimmed, padded to a fixed duration, or loudness-normalized. Each complete file is treated as one spoken-content unit; do not compare a long ambient recording as though it were a short phrase.

```bash
python inference.py compare enroll1.wav query.wav
python inference.py --model gru160 compare enroll1.wav query.wav
python inference.py --model gru96 info
python inference.py --model cnn256-relu-best compare enroll1.wav query.wav
python inference.py --model cnn256-relu-last info
```

Enroll three independently recorded examples of one keyword:

```bash
python inference.py enroll --keyword "My wake phrase" --audio enroll1.wav enroll2.wav enroll3.wav
python inference.py match --audio query.wav other_phrase.wav
python inference.py match --audio query.wav --threshold 0.5
```

The default output is `templates.json` in the current directory, storing embeddings and labels, not audio. Repeating enrollment for the same label appends samples: at most five per keyword and 20 keywords. Use `--templates another.json` for a separate template file.

Put `--model` before the subcommand, and use the same model for enrollment and matching. Each model uses its own validation threshold; do not copy CNN thresholds or templates to GRU. File templates and fixed-window web templates are separate formats and cannot be interchanged.

Thresholds are only starting points. Calibrate on independent recordings representative of your deployment. File tests do not apply the web demo's energy gate or consecutive-hit logic; silence and noise still produce embeddings.

## Phone enrollment and continuous matching

The phone records audio and the computer runs inference. Both must share a trusted LAN. Use system Chrome/Safari and make sure the computer has the `openssl` executable.

Replace `YOUR_COMPUTER_LAN_IP` with the computer's IPv4 LAN address:

```bash
python web_kws.py --address YOUR_COMPUTER_LAN_IP
python web_kws.py --model gru160 --address YOUR_COMPUTER_LAN_IP
python web_kws.py --model cnn256-relu-best --address YOUR_COMPUTER_LAN_IP
python web_kws.py --model cnn256-relu-last --address YOUR_COMPUTER_LAN_IP
```

1. Open the printed HTTP certificate setup URL on the phone (default port 8767).
2. Compare the CA fingerprint with the terminal output and manually install the test CA. iPhone also requires explicitly enabling full certificate trust.
3. Follow the setup page's HTTPS link (default port 8766). Ordinary LAN HTTP pages cannot access the browser's live microphone API.
4. Disconnect unwanted Bluetooth/headset devices and check the displayed actual input. After permission is granted, you can select another microphone.
5. Enter a label and wait for **Start speaking**. Record once, replay to confirm your voice is present, and enroll three independent takes if possible.
6. Select **Start listening**, keep the page in the foreground, and keep the phone unlocked.

Defaults: a two-second rolling window and a **100 ms matching interval**. The first match still requires a full two-second buffer. Busy network/inference periods skip old windows rather than queuing them. Two consecutive hits trigger a wake event, followed by a 1.5-second cooldown and low-score rearming.

The level meter also moves during preparation, but countdown audio is not enrolled. Raw audio is held in memory; the computer stores enrollment vectors only. The replay WAV is saved only if the user explicitly downloads it.

```bash
python web_kws.py --model gru160 --address YOUR_COMPUTER_LAN_IP --hop-seconds 0.2
python web_kws.py --address YOUR_COMPUTER_LAN_IP --port 8866 --setup-port 8867
```

Choose unused ports if another demo is running. Restart to change models. Model fingerprints isolate web templates automatically; a new model or window length requires fresh enrollment. Shorter intervals increase scan frequency, not necessarily accuracy; highly overlapping windows are not independent confirmations.

**Security and privacy:** tokenless access is the default, so other devices on the LAN can access and modify templates. Use only on a trusted LAN and do not expose this development server publicly. Enable `--require-token` if needed. No firewall or system trust-store changes are made automatically.

Certificates/private keys are generated locally in `.web_kws_tls/`; never commit or share them. Remove the test CA from the phone after testing. Templates in `recordings/web_kws/` should also remain private. Microphone audio goes to the computer running this program, not to a third-party cloud service.

## Recorded offline results

The following are existing **GELU CNN epoch-11 fixed-episode** results, not real-time web measurements or exhaustive evaluations over all test data. The threshold was calibrated on validation data and frozen for testing.

| Metric | Validation episode | Test episode |
|---|---:|---:|
| Global EER | 1.261% | 2.929% |
| Source-macro EER | 0.741% | 3.170% |
| TAR | 98.109% | 94.561% |
| Test pair FAR | — | 0.868% |
| Background accept rate against 100 templates | 54.509% | 49.400% |

Each episode enrolls 100 classes with one recording per class and includes 100 unregistered content classes as background, seed=42. The test episode contains 478 positive comparisons, 97,322 negative comparisons, and 500 background queries. Source utterances are disjoint within the episode.

EER scans score thresholds; TAR uses the fixed validation threshold. Macro EER averages per-source EERs, not per-class EERs. **Pair FAR is not false alarms per hour.** Background accept rate means an unregistered-content query is accepted by any of 100 templates; it is not an ambient-silence false-alarm rate. The high background acceptance shows that open-set rejection remains insufficient for production claims.

GRU160 epoch 24 has an existing report on the same fixed test episode:

| Model | Global test EER | Test macro EER | TAR | Pair FAR | Background accept rate |
|---|---:|---:|---:|---:|---:|
| GRU160 epoch 24 | 3.766% | 2.473% | 92.050% | 0.975% | 44.400% |
| CNN256 epoch 11 | 2.929% | 3.170% | 94.561% | 0.868% | 49.400% |

Each model uses its own validation threshold. CNN is not better on every metric, and architecture, objective, and augmentation changed together; this is not an isolated CNN-versus-GRU ablation.

GRU96 currently includes a verified validation report only: macro EER 4.621%, TAR 69.748%. Missing standardized test results are not invented. Detailed protocols and numbers are in `weights/model_info.json`, `weights/gru160_model_info.json`, and `weights/gru96_model_info.json`. Evaluation audio and per-recording episode files are not included.

### Completed ReLU CNN run: best and last

| Checkpoint | Validation macro EER | Validation TAR | Fixed-test global EER | Fixed-test macro EER | Fixed-test TAR | Test background accept rate |
|---|---:|---:|---:|---:|---:|---:|
| Best, epoch 26 | 0.652453% | 98.319% | 3.138% | 3.367% | 96.025% | 50.200% |
| Last, epoch 30 | 0.654215% | 98.950% | Not evaluated | Not evaluated | Not evaluated | Not evaluated |

Epoch 26 uses the same frozen test episode described above, with its own validation-only threshold; its test pair FAR is 1.005%. Compared with the deployed epoch-10 snapshot, test TAR increased from 91.213% to 96.025%, but background acceptance against 100 templates increased from 39.800% to 50.200%. This is not an unconditional improvement in rejection quality.

Epoch 30 has only validation metrics and synthetic inference checks in this release. Do not treat epoch-26 test metrics as epoch-30 results. The tiny validation EER difference does not establish a meaningful real-world advantage for epoch 26.

Metadata is recorded separately in `weights/cnn_relu_best_model_info.json` and `weights/cnn_relu_last_model_info.json`; the shared training recipe is in `training_config_cnn_relu.json`. ReLU does not reduce the 2,659,649 encoder parameters or the convolution MAC count relative to the GELU CNN.

## Files and regression tests

```text
weights/               Five fixed checkpoints, model catalog, and metadata
content_model.py       GRU/CNN encoders only; no training classification heads
model_catalog.py       Local model selection
audio.py               Training-compatible log-Mel frontend
inference.py           CPU file enrollment, comparison, and matching
web_kws.py, web_kws/    Phone-microphone web demo and local HTTPS service
verify.py, tests/      Synthetic verification and regression tests
TRAINING.md            Training procedure and differences between model recipes
training_config*.json  Sanitized CNN/GRU training configurations
data_summary.json      Aggregate class/sample statistics
DATA_SOURCES.md        Dataset provenance, restrictions, and release checklist
```

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
node --test tests/test_web_kws.mjs
```

Python web tests require `openssl`. Node tests are optional; the web service itself does not require Node. This package contains inference assets and training documentation, not training entry points or a fully reproducible training release.
