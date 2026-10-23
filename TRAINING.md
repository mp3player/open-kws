# Training procedure for the released checkpoints

The main sections describe the completed GELU CNN experiment released as `weights/cnn_gelu_epoch11.pt`. Section 8 documents the two GRU baselines separately.

Training code, classification heads, optimizer states, data-download scripts, training/evaluation audio, and ongoing experiments are not included. Configurations are recorded in `training_config.json`, `training_config_gru160.json`, and `training_config_gru96.json`; these are records, not executable training scripts.

## 1. Task

Treat the **exact spoken content** of an utterance or phrase as its class identity, not its speaker, semantic intent, or frame-level phoneme labels. Different wordings are different classes even when their meanings match; different speakers reading the same wording belong to one content class.

No CTC, reconstruction objective, frame alignment, or text input is used. Content classification is intended to learn useful embeddings for unseen phrases, but open-set generalization is an empirical objective, not a guarantee.

## 2. Dataset construction

The mixture contains **THCHS-30, Fluent Speech Commands, SLURP, WenetPhrase, and GigaPhrase-1000**. Upstream and Hugging Face sources are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

Audio records are grouped by exact normalized content rather than intent. Some records are phrase segments derived from longer source utterances.

- Remove whitespace from Chinese text; casefold English and collapse whitespace. Hash the language and normalized text to form a content ID.
- Require at least four Chinese characters or two whitespace-separated English words.
- Require at least five samples and three distinct source-utterance groups per class.
- Where speaker IDs exist, require at least two known speakers. Classes with no known speaker IDs may remain; they do not have a guaranteed cross-speaker identity check.
- Deduplicate identical audio references. Exclude ambiguous records when one audio reference has conflicting content or known speaker labels.
- Assign approximately 90%/5%/5% of content IDs to train/validation/test using a deterministic hash, seed=42.
- If one source group crosses splits, retain only its highest-priority split: test > validation > train. This prevents same-source segments from crossing those splits.
- Global speaker-disjointness across train, validation, and test is not additionally guaranteed.

Retained pools:

| Split | Content classes | Recordings / segments |
|---|---:|---:|
| Train | 97,346 | 1,558,850 |
| Validation | 5,772 | 113,747 |
| Test | 6,021 | 151,051 |

Training counts by primary source:

| Source | Content classes | Samples |
|---|---:|---:|
| FSC | 222 | 27,181 |
| GigaPhrase | 87,524 | 1,394,523 |
| SLURP | 1,427 | 10,757 |
| THCHS-30 | 899 | 12,010 |
| WenetPhrase | 7,274 | 114,379 |

English GigaPhrase dominates this mixture; the data are not language-balanced. These are retained index counts, not original corpus sizes or exclusively independent full-utterance recordings. Aggregate statistics, filtering settings, and the index fingerprint are in `data_summary.json`; audio and class/transcript inventories are not distributed.

## 3. Audio frontend

```text
Channel mean -> resample to 16 kHz -> MelSpectrogram
-> clamp_min(1e-10) -> natural logarithm
-> subtract each frequency band's mean over the utterance
```

FFT size 512; Hann window length 400 samples (25 ms); hop 160 samples (10 ms); 40 Mel bands; power=2; center=False. Other Mel settings use the corresponding torchaudio defaults, with the same implementation retained in this release.

No energy-based trimming, uniform two-second training crop, or per-recording loudness normalization is applied. Variable-length examples are padded within a batch and masked using valid lengths throughout the encoder and pooling.

The web demo's fixed two-second window is an **inference protocol**, not a universal training-recording duration.

## 4. CNN architecture

```text
40-D log-Mel
-> Linear(40 -> 64) + LayerNorm + GELU
-> Conv1d(64 -> 128, kernel=5, stride=2) + LayerNorm + GELU + Dropout(0.1)
-> Conv1d(128 -> 256, kernel=5, stride=2) + LayerNorm + GELU + Dropout(0.1)
-> 6 residual blocks, dilation=1,2,4,8,16,32
   Two Conv1d(256 -> 256, kernel=3) layers per block,
   each followed by LayerNorm + GELU + Dropout(0.1)
-> Masked attention pooling concatenated with the last valid state (512-D)
-> Linear(512 -> 128) + LayerNorm
-> Training: ArcFace head; inference: L2 normalization
```

Convolutions are noncausal with two-sided padding; total temporal downsampling is 4x. Attention scoring is Linear(256 -> 64) + Tanh + Linear(64 -> 1), followed by temporal Softmax.

The encoder has 2,659,649 parameters, occupying 10,638,596 bytes as raw FP32 parameters. The checkpoint also contains configuration and calibration metadata. The roughly 12.46-million-parameter training head is excluded.

GELU, LayerNorm, Tanh, and Softmax remain at inference; this is not an MCU-optimized architecture.

## 5. Objective, sampling, and optimization

Encoder outputs and classifier weights are L2-normalized to obtain cosine logits. ArcFace applies an angular margin to the target class, with a monotonicity-preserving piecewise correction; other classes retain their original cosine:

```text
Target logit:     s * cos(theta_y + m), m = 0.2 radians, s = 30
Non-target:      s * cos(theta_j)
Raw CB weight:   w_c = (1 - beta) / (1 - beta**n_c), beta = 0.9999
Normalization:   mean weight across training samples equals 1
Loss:            mean(w_label * cross_entropy_per_sample)
```

The loss is class-balanced CE on ArcFace logits, not a combination with CTC, reconstruction, or a separate contrastive loss. Logged training accuracy includes the target margin and is not ordinary margin-free classification accuracy.

- Shuffle training recordings without replacement each epoch; no class/source-balanced resampling in this CNN run.
- Three DDP processes, batch size 32 per GPU, global batch size 96.
- 16,238 steps per complete epoch. To avoid duplicated DDP padding, at most world_size-1 tail samples are omitted; two for this index.
- AdamW: learning rate 0.0003, weight decay 0.01, other optimizer parameters at their defaults.
- Per-step CosineAnnealingLR over the originally planned 30 full epochs, eta_min=0. Epoch 11 was not retrained with an 11-epoch scheduler.
- Gradient-norm clipping at 5; seed=42; AMP disabled.
- Evaluate each epoch and select the encoder using validation source-macro EER. This release pins epoch 11 and does not track later snapshots automatically.

## 6. Augmentation

Augmentations are independently sampled during training and disabled for validation/inference.

1. With probability 0.5, apply a uniformly sampled gain between -6 and +6 dB.
2. With probability 0.5, add independently sampled 0-0.3 seconds of silence at the left/right boundaries, not inside the utterance.
3. With probability 0.5, mix synthetic white or pink noise, chosen randomly, with SNR sampled uniformly from 10-30 dB.
4. After amplitude-changing operations, scale down only if the peak exceeds 1 to avoid overflow; this is not fixed loudness normalization.
5. Apply SpecAugment after per-band temporal mean subtraction: two time masks, each up to min(40 frames, 10% of valid frames), and two frequency masks, up to six bands. Fill with the feature mean.

The implementation supported optional recorded background noise, but **this run used noise_dir=None and background_files=0**. No real environmental background recordings were mixed in this run. An available option must not be reported as an augmentation actually used.

## 7. Validation, testing, and limits

Each fixed-seed validation episode enrolls 100 classes, one recording each, with up to five queries per class, plus 100 unregistered content classes as background.

Source-utterance groups are disjoint within the episode. Cross-speaker positive queries are preferred when speaker IDs are known; queries with unknown speaker IDs are counted separately.

Calibrate a threshold targeting 1% pair FAR on validation negative comparisons and freeze it for testing. Macro EER is the arithmetic mean of per-source EERs, not a per-content-class average.

The CNN threshold is 0.3223716021. Recorded validation/test results are in README and `weights/model_info.json`.

This is not a fully reproducible training release and does not include fixed-episode audio. Offline pair FAR is not false alarms per hour; informal phone tests are not sufficient evidence of production reliability.

Remaining work includes deployment-specific calibration, near-sounding negatives, continuous-background false-alarm evaluation, quantization/compression, and target-hardware measurements.

## 8. Additional GRU baselines

Both GRU checkpoints classify **whole-utterance content**; they are not the earlier unsuccessful GRU+CTC experiments. They share the CNN's 40-band frontend, temporal mean subtraction, 128-D embedding, and inference L2/cosine pipeline.

All three saved dataset fingerprints match, but sampling, objectives, and augmentation differ. Do not describe the GRUs using the CNN recipe above.

```text
40-D log-Mel -> Linear(40 -> 64) + LayerNorm + GELU
-> Two-layer unidirectional GRU, hidden=96 or 160, inter-layer dropout=0.1
-> Masked attention pooling concatenated with the last valid state
-> Linear(2 * hidden -> 128) + LayerNorm
-> Training: ordinary Linear/Softmax head; inference: remove head and L2-normalize
```

GRU96 has 136,513 encoder parameters; GRU160 has 317,505. Legacy configs without an `encoder_type` field are explicitly interpreted as GRU, not CNN.

### GRU160 + Softmax/CB, epoch 24

- `gru160_softmax_epoch24.pt` is the best validation source-macro-EER snapshot of this run, not the later epoch-28 snapshot.
- 97,346 content classes; shuffled training recordings; three GPUs; batch 32 per GPU; 16,238 steps per full epoch.
- Class-balanced ordinary Softmax CE, beta=0.9999; **no ArcFace margin**.
- AdamW, lr=0.0003, weight_decay=0.01; originally 30 epochs; AMP off; seed=42.
- Saved configuration enables feature augmentation but does not record the CNN run's waveform-noise settings. Do not claim the same white/pink-noise recipe.
- Validation macro EER 1.198%, TAR 96.008%, threshold 0.3338347077. Existing fixed-test results are listed in README.

### GRU96 + Softmax, epoch 16

- `gru96_softmax_epoch16.pt` is the early small-encoder run's best validation snapshot; epoch 17 was not its best.
- Original batching config: classes_per_batch=16 and samples_per_class=2, with 2,000 fixed steps per epoch. This is not a full no-replacement traversal.
- Ordinary Softmax classification. The legacy config has no CB weighting field; do not apply the later GRU160 recipe retroactively.
- AdamW, lr=0.0003, weight_decay=0.01; originally 30 epochs; AMP off; seed=42.
- Feature augmentation is enabled. The old config does not recover every sampling implementation detail; this release neither includes the old training code nor invents missing settings.
- Validation macro EER 4.621%, TAR 69.748%, threshold 0.5535052419. Retained as a lightweight baseline, not as a model of equal quality to CNN.

Use separate enrollment templates and each model's own threshold.
