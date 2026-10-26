# Dataset provenance, licensing, and release checklist

As of 2026-10-26, this directory is a **release draft** for maintainer review. No code or weight license has been selected; no MIT, Apache-2.0, commercial-use, or redistribution permission is granted here.

Publishing only weights does not remove the need to review upstream dataset terms. The conditions for distributing and using trained weights still require maintainer verification and, where necessary, confirmation from the rights holders.

This is a provenance record and risk notice, not legal advice or a completed legal clearance.

## The five training corpus families

| Dataset | Language / content | Upstream or official source | Hugging Face mirror used during preparation |
|---|---|---|---|
| THCHS-30 | Mandarin read sentences | [OpenSLR SLR18](https://www.openslr.org/18/) | [changelinglab/thchs30-segment](https://huggingface.co/datasets/changelinglab/thchs30-segment) |
| Fluent Speech Commands (FSC) | English spoken commands, grouped by exact wording rather than intent | [Fluent.ai dataset description and terms](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) | [leo19941227/fluent_speech_commands](https://huggingface.co/datasets/leo19941227/fluent_speech_commands) |
| SLURP | English spoken commands / assistant requests | [Official repository](https://github.com/pswietojanski/slurp) | [marcel-gohsen/slurp](https://huggingface.co/datasets/marcel-gohsen/slurp) |
| WenetPhrase | Mandarin phrase segments derived from WenetSpeech | [MM-KWS data pipeline](https://github.com/aizhiqi-work/MM-KWS#data-pipeline-for-wenetphrase) | [ZhiqiAi/Wenetphrase](https://huggingface.co/datasets/ZhiqiAi/Wenetphrase) |
| GigaPhrase-1000 | English phrase segments; includes upstream LibriSpeech/GigaSpeech material | [Dataset authors' repository](https://github.com/aizhiqi-work/GigaPhrase-1000) | [ZhiqiAi/GigaPhrase-1000](https://huggingface.co/datasets/ZhiqiAi/GigaPhrase-1000) |

These links document provenance; they do not imply mirror uploaders can expand upstream permissions. No dataset downloader or dataset redistribution is included.

LibriSpeech, GigaSpeech, and WenetSpeech are upstream sources for the phrase collections, not additional independent prepared datasets counted in our five-way mixture. Exact source revisions are not recovered here; this document does not claim bit-for-bit dataset reproducibility.

## Actual retained training subset

| Dataset | Content classes | Recordings / segments |
|---|---:|---:|
| THCHS-30 | 899 | 12,010 |
| FSC | 222 | 27,181 |
| SLURP | 1,427 | 10,757 |
| WenetPhrase | 7,274 | 114,379 |
| GigaPhrase | 87,524 | 1,394,523 |
| **Total** | **97,346** | **1,558,850** |

Counts are after content-length, sample-count, source-group, and known-speaker filtering and split isolation. They are attributed to each class's primary source, not presented as original corpus sizes. See `data_summary.json` and [TRAINING.md](TRAINING.md) for aggregate split counts and preparation details.

## Identified restrictions

### Additional CNN96 / CNN48 background data: ESC-50 and synthetic noise

The CNN96 and CNN48 variants use the same five-corpus speech pool plus a single background class. The original CNN96 20-epoch run had 1,000 synthetic training clips (white noise, pink noise, and silence) and two nominal ESC-50 clips. Both nominal training clips and one nominal validation clip were audited as entirely zero; they must not be counted as effective real-background coverage. The original category-ID mapping was also incorrect. Historical v2 metadata is retained with an explicit audit warning, not endorsed as correct category selection.

The completed three-epoch CNN96 background fine-tune and the CNN48 trained-from-scratch run used the same corrected pool of selected nonzero clips from [ESC-50 upstream](https://github.com/karolpiczak/ESC-50), obtained through [ashraq/esc50](https://huggingface.co/datasets/ashraq/esc50) at revision `e3e2a63ffff66b9a9735524551e3818e96af03ee`. Candidate category names are rain, sea waves, wind, washing machine, vacuum cleaner, and engine. Selection keeps original levels and applies energy/activity/transient/tonality checks; it does not attenuate loud effects into background examples. These checks do not guarantee the absence of incidental speech. CNN48 uses ordinary CB weighting, not the CNN96 fine-tune's special 5% nominal background weighted-mass setting.

| Split | Real clips / independent source recordings | Synthetic clips |
|---|---:|---:|
| Train | 43 / 20 | 1,000 |
| Validation | 11 / 2 | 100 |
| Test | 14 / 7 | 100 |

Source-disjoint folds 1-3/4/5 define the background splits. Multiple windows from one source are not independent recordings. Only training clips enter the additional class; held-out background is neither enrolled as a keyword nor used for speech-threshold calibration. Original CNN96 best and CNN48 best have now been evaluated on the same frozen speech episode and corrected held-out background pool; their last checkpoints and the fine-tuned CNN96 exports do not have such a test claim. Aggregate results and limitations are recorded beside the corresponding best weights, without audio or per-recording transcripts.

The recorded ESC-50 terms are **CC BY-NC 3.0**; preserve upstream clip attribution and review the upstream license and derived-weight distribution conditions. No audio is distributed here. This additional provenance does not resolve the existing speech-corpus restrictions or grant commercial rights. The sanitized `background_data` in each CNN96 / CNN48 training configuration records the dataset revision, manifest hash, policy, counts, and limitations without private paths.

### Speech-corpus restrictions

- **THCHS-30:** the OpenSLR page lists Apache-2.0. Preserve the notices and citations required by the actual data version used.
- **FSC:** the official page specifies academic research only and prohibits commercial training, testing, benchmarking, and product development. It links the complete license. Do not attach an unconditional commercial-use claim to these weights.
- **SLURP:** the official license separates text under CC BY 4.0 from audio under **CC BY-NC 4.0**. Audio was used for training; citing only the text license is insufficient. See the [official LICENSE](https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt).
- **WenetPhrase:** the authors state that the audio is sliced from WenetSpeech and copyrighted by the original data authors. MM-KWS's code license is not an audio license. Separately review the actual [WenetSpeech](https://github.com/wenet-e2e/WenetSpeech) data terms and derived-data requirements.
- **GigaPhrase:** the corresponding derived-data versions, upstream audio terms, and conditions for trained-weight distribution have not been fully audited. Availability on Hugging Face is not an authorization determination.

These observations do not establish a final license for the trained weights. They must be reviewed before public distribution or commercial deployment.

## Before public release

1. Confirm the inference code's authorship, ownership, notices, and chosen license. Training code is not included.
2. Separately confirm weight redistribution, modification, and commercial-use conditions, obtaining upstream authorization where necessary. A new custom LICENSE cannot override upstream restrictions.
3. For a version without these restrictions, consider obtaining permission or retraining on data licensed for the intended purpose, then evaluating the replacement model.
4. Keep ongoing models and unnecessary intermediate snapshots excluded. This directory pins GRU96 epoch 16, GRU160 epoch 24, GELU CNN epoch 11, ReLU CNN256 best epoch 26 / last epoch 30, CNN96 best epoch 16 / last epoch 20, the completed background fine-tune's best / last at additional epoch 3, and trained CNN48 best epoch 19 / last epoch 20. All are encoder-only inference exports without training heads or optimizer states. Random-weight hardware timing fixtures are not included.
5. Do not add personal recordings, per-recording transcripts, enrollment vectors, server paths, access tokens, training heads, or optimizer states.
6. Exclude `.web_kws_tls/`, `recordings/`, `templates.json`, `.venv/`, and `__pycache__/`. The clean release contains no such runtime artifacts.
7. Keep metrics and checksums accurate. Never relabel offline pair FAR as always-on false alarms per hour or claim MCU deployment has been validated.

Until these checks are complete, treat the directory as a locally reviewable draft, not a formally cleared open-source release.
