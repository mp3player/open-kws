# Training procedure for the released checkpoints

The main sections describe the completed GELU CNN experiment released as `weights/cnn256/epoch11.pt`. Section 8 documents the two GRU baselines; section 9 documents the completed ReLU CNN256 run; section 10 documents CNN96 and its background fine-tune; section 11 documents the smaller CNN48 run. Completed ReLU CNN runs each retain best/last exports.

Training code, classification heads, optimizer states, data-download scripts, training/evaluation audio, and ongoing experiments are not included. Each model folder under `weights/` contains its own `training_config.json`; these are records, not executable training scripts.

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

The CNN threshold is 0.3223716021. Recorded validation/test results are in README and `weights/cnn256/model_info.json`.

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

- `weights/gru160/epoch24.pt` is the best validation source-macro-EER snapshot of this run, not the later epoch-28 snapshot.
- 97,346 content classes; shuffled training recordings; three GPUs; batch 32 per GPU; 16,238 steps per full epoch.
- Class-balanced ordinary Softmax CE, beta=0.9999; **no ArcFace margin**.
- AdamW, lr=0.0003, weight_decay=0.01; originally 30 epochs; AMP off; seed=42.
- Saved configuration enables feature augmentation but does not record the CNN run's waveform-noise settings. Do not claim the same white/pink-noise recipe.
- Validation macro EER 1.198%, TAR 96.008%, threshold 0.3338347077. Existing fixed-test results are listed in README.

### GRU96 + Softmax, epoch 16

- `weights/gru96/epoch16.pt` is the early small-encoder run's best validation snapshot; epoch 17 was not its best.
- Original batching config: classes_per_batch=16 and samples_per_class=2, with 2,000 fixed steps per epoch. This is not a full no-replacement traversal.
- Ordinary Softmax classification. The legacy config has no CB weighting field; do not apply the later GRU160 recipe retroactively.
- AdamW, lr=0.0003, weight_decay=0.01; originally 30 epochs; AMP off; seed=42.
- Feature augmentation is enabled. The old config does not recover every sampling implementation detail; this release neither includes the old training code nor invents missing settings.
- Validation macro EER 4.621%, TAR 69.748%, threshold 0.5535052419. Retained as a lightweight baseline, not as a model of equal quality to CNN.

Use separate enrollment templates and each model's own threshold.

## 9. Completed ReLU CNN256: best epoch 26 and last epoch 30

The ReLU run keeps the CNN's 256 channels, six residual blocks, 128-D output, attention pooling, normalization, ArcFace objective, CB weighting, full-data sampling, and augmentation recipe. Projection and temporal-convolution activations are ReLU instead of GELU. The encoder still has 2,659,649 parameters and 10,638,596 raw FP32 parameter bytes; this is an activation variant, not a smaller-parameter model.

The completed run has 30 full epochs, 16,238 steps per epoch, 1,558,850 training utterances, global batch 96, ArcFace margin 0.2, scale 30, and CB beta 0.9999. Synthetic-noise augmentation is enabled; recorded background files remain zero. The sanitized saved configuration is `weights/cnn256_relu/training_config.json`.

The training save rule is strict improvement in validation source-macro EER. This selects **epoch 26** (0.006524534300330631), whereas **epoch 30** has 0.006542148778072026. Both round to 0.0065 in the training log. Epoch-30 validation TAR is higher, so the two selection criteria must not be conflated.

- `weights/cnn256_relu/best.pt`: encoder export selected by validation macro EER, epoch 26; its frozen test episode has been evaluated.
- `weights/cnn256_relu/last.pt`: encoder export from the final training checkpoint, epoch 30; no standardized test result is claimed for it yet.

Both exports retain their own validation calibration and fingerprint and exclude classification heads and optimizer states. Complete training-resume checkpoints are not distributed through this inference repository. Older GRU and GELU exports remain unchanged.

## 10. Completed CNN96 and background fine-tune

CNN96 uses the same noncausal residual CNN structure, ReLU activations, six blocks, 64-D input projection, 128-D embedding, and attention/last-state pooling. Temporal stages use 48 and 96 channels instead of 128 and 256. The encoder has **408,049 parameters**, 1,632,196 raw FP32 parameter bytes, and approximately 20.1 million Conv/Linear MACs for a two-second, 197-frame input. MACs exclude the frontend, pooling, normalization, and runtime overhead; no MCU benchmark or quantization is provided.

### Original CNN96: best epoch 16, last epoch 20

- 20 full epochs, 8,125 steps per epoch, three GPUs, batch 64 per GPU / global batch 192; AMP disabled.
- ArcFace margin 0.1, scale 30; CB beta 0.9999; AdamW lr=0.0003, weight decay 0.01; cosine schedule across 20 epochs.
- Same 1,558,850 speech training examples, plus one background class with 1,002 clips: 1,000 synthetic clips and two nominal ESC-50 clips later audited as all-zero. Total training classes: 97,347; total recordings: 1,559,852. There was no effective real-background coverage in that preparation.
- Voice waveform/feature augmentation is retained. Background-class examples bypass both waveform augmentation and SpecAugment. Frontend processing is unchanged; per-band temporal mean subtraction means simple volume reduction is not a reliable rejection signal.
- Best is epoch 16 by minimum validation source-macro EER (2.256065%), not the final epoch 20 (2.257540%). Their validation TARs are 93.067% and 93.277%, respectively.
- Exports: `weights/cnn96_relu/best.pt` and `last.pt`. Original best epoch 16 has now been evaluated on the frozen test episode: source-macro EER 5.383986%, TAR 87.029%, pair FAR 0.9854%. Its unchanged speech test records were verified against the corrected v3 index; metadata distinguishes the evaluation-index fingerprint from the original training-index fingerprint. Last remains untested on that episode. See colocated metadata and sanitized training configuration. The original v2 background metrics are historical logs, not evidence of real-background rejection; the entry labeled `engine` was actually zero audio.

This run changes width, margin, batch size, training duration, and the background class relative to CNN256; it is not an isolated width ablation.

### Background fine-tune: best and last at additional epoch 3

- Initialize encoder **and classification head** from original CNN96 epoch 20. Preserve class-label order, reset AdamW/scheduler/calibration, and run three additional full epochs at lr=0.00003 with full speech replay. Batch, architecture, margin, and scale stay unchanged.
- Replace the background pool with 1,043 training clips: 43 real clips from 20 source recordings and 1,000 synthetic clips. Total training recordings: 1,559,893; classes remain 97,347.
- Real audio is selected from ESC-50 rain, sea waves, wind, washing machine, vacuum cleaner, and engine categories. Keep original levels, resample to mono 16 kHz, and slice two-second clips at 1.5-second hops. Folds 1-3/4/5 define train/validation/test; independent source recordings do not cross splits.
- Screen real clips before and after padding: RMS between -80 and -28 dBFS, peak at most -12 dBFS, 20 ms frame RMS at most -20 dBFS, frame rise at most 18 dB, strongest-three-adjacent-bin spectral energy fraction at most 0.65, and at least 20% active frames. These are heuristics, not VAD or calibrated sound pressure thresholds. Real all-zero audio is rejected; explicitly synthetic silence is separately marked.
- Retain speech relative CB weights, but set the background class to 5% of nominal weighted sample mass, then renormalize. This does not guarantee 5% of loss or gradient contribution. No extra head, metric loss, or VAD is added.
- Hold out 11 real validation clips / two sources and 14 real test clips / seven sources, plus 100 synthetic clips in each split. Held-out background is never enrolled or used to calibrate the speech threshold.
- Best and last both select additional epoch 3 and contain identical encoder weights and calibration. They are saved separately as `weights/cnn96_relu_background/best.pt` and `last.pt`; epoch 3 is relative to initialization, not a three-epoch training-from-scratch claim.

On the same new validation pool, original epoch 20 vs fine-tuned epoch 3 has speech macro EER 2.257540% vs 2.244099%, TAR 93.277% vs 93.067%, and real-background acceptance 4/11 vs 4/11. Aggregate non-speech acceptance including synthetic clips is 77.48% vs 87.39%, using separately speech-calibrated thresholds. These results do not establish improved background rejection; the background training-class accuracy of about 99.81% is not a cosine-rejection metric. No independent fixed-test evaluation is claimed for the fine-tuned exports.

ESC-50 provenance, its recorded CC BY-NC 3.0 terms, and upstream review requirements are documented in [DATA_SOURCES.md](DATA_SOURCES.md). The classification head, optimizer state, audio, manifests, source paths, and personal recordings are excluded from all four inference exports.

## 11. Completed CNN48: best epoch 19 and last epoch 20

This is a trained noncausal ReLU CNN with **48 channels and four residual blocks**, not a random-weight hardware timing fixture. It retains the 40-band frontend, 64-D projection, LayerNorm, attention/last-state pooling, 128-D embedding, and final L2 normalization. Temporal stages use 24 and 48 channels; the four residual dilations are 1, 2, 4, and 8.

- 88,729 encoder parameters / 354,916 raw FP32 parameter bytes, approximately 0.37 MB per checkpoint. Conv/Linear MACs are 4,486,528 for a two-second, 197-frame input, excluding frontend, normalization, pooling, and runtime overhead.
- Train from scratch for 20 complete epochs; no CNN96 weight initialization, distillation, pruning, or quantization-aware training. Three DDP processes, batch 64 per GPU / global batch 192; 8,125 steps per epoch; AMP disabled.
- ArcFace margin 0.1, scale 30; normal class-balanced CE for all 97,347 classes, beta 0.9999. The background class has no special 5% weighted-mass override.
- AdamW lr=0.0003, weight decay 0.01, per-step cosine schedule over 20 epochs, gradient clip 5, seed 42. Voice waveform augmentation and SpecAugment are retained; selected background-class clips bypass those augmentations.
- Same 1,558,850 examples from THCHS-30, Fluent Speech Commands, SLURP, WenetPhrase, and GigaPhrase-1000, plus the corrected 1,043-clip background pool described above. Total training recordings: 1,559,893. The real-background subset contains 43 clips from 20 source recordings, separate from 1,000 synthetic clips. The inference/frontend policy is unchanged.
- Best is **epoch 19**, selected by minimum validation source-macro EER 5.218730%, TAR 73.5294%, threshold 0.5620555282. Final **epoch 20** has validation macro EER 5.329304%, TAR 73.3193%, threshold 0.5628631115.
- `weights/cnn48_relu/best.pt` and `last.pt` have distinct fingerprints and calibrations. Only epoch 19 has a fixed-test result; do not attribute that result to epoch 20. Classification heads, optimizer states, private paths, personal recordings, and random timing models are not distributed.

The best checkpoint's frozen test episode has 478 positive comparisons and 97,322 negative comparisons: global EER 7.693170%, source-macro EER 7.647995%, TAR 71.9665%, pair FAR 0.9361%, and unregistered-speech acceptance 35.8% against 100 templates. The original CNN96 best uses exactly the same verified speech records and in-memory features, with its own original validation threshold: test TAR 87.0293% and source-macro EER 5.383986%. Width, depth, and training background pool all changed; this is not a single-factor architecture ablation.

On the same v3 held-out background pool, each best model accepts 6/14 real clips from seven source recordings, or 67/114 including synthetic clips. No energy gate or noise enrollment is used. This is not a continuous false-alarms/hour evaluation, and the small real-background pool does not establish broad robustness.

These are FP32 PyTorch inference checkpoints, not INT8, ONNX, or MCU-ready binaries. Parameter/MAC reductions do not guarantee MCU real-time operation or memory fit; no target-hardware result is claimed. Dataset restrictions, including the recorded ESC-50 CC BY-NC 3.0 terms and unresolved speech-corpus permissions, still apply.
