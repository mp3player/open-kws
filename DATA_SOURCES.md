# Dataset provenance, licensing, and release checklist

As of 2026-10-23, this directory is a **release draft** for maintainer review. No code or weight license has been selected; no MIT, Apache-2.0, commercial-use, or redistribution permission is granted here.

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
4. Keep ongoing models and unnecessary intermediate snapshots excluded. This directory pins GRU96 epoch 16, GRU160 epoch 24, GELU CNN epoch 11, and the now-completed ReLU CNN run's best epoch 26 and last epoch 30. The two ReLU exports contain no training head or optimizer state.
5. Do not add personal recordings, per-recording transcripts, enrollment vectors, server paths, access tokens, training heads, or optimizer states.
6. Exclude `.web_kws_tls/`, `recordings/`, `templates.json`, `.venv/`, and `__pycache__/`. The clean release contains no such runtime artifacts.
7. Keep metrics and checksums accurate. Never relabel offline pair FAR as always-on false alarms per hour or claim MCU deployment has been validated.

Until these checks are complete, treat the directory as a locally reviewable draft, not a formally cleared open-source release.
