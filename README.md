# Neuro-Symbolic Fake News Detection with ProbLog

A teaching demo for a Neuro-Symbolic AI lab: composing multiple noisy, independently-trained
perception models into a single auditable verdict (`Support` / `Refute` / `Unverified`) using
probabilistic logic, instead of a single end-to-end black-box classifier.

## Idea

Given a **claim** (image + text) and a candidate **document** (image + text) from a real news
source, decide whether the document supports, refutes, or fails to address the claim.

The pipeline splits the work exactly along the neuro-symbolic line:

- **Neural (perception only)** — a handful of small, frozen models, each producing one fuzzy
  probability:
  - an NLI model → `text_entails`, `text_contradicts`
  - CLIP → `image_matches`
  - *(experimental)* a small generative LM used as an explicit negation detector →
    `claim_negated`
- **Symbolic (fixed, auditable rules)** — a [ProbLog](https://dtai.cs.kuleuven.be/problog/)
  program combines those probabilities into a verdict:

  ```prolog
  0.9::supported :- text_entails, \+ claim_negated.
  refuted        :- text_contradicts.
  unverified     :- \+ supported, \+ refuted.
  ```

  No neural network ever sees a rule; no rule ever sees raw text or pixels.

> **Note on framework scope.** This demo uses `problog` directly for inference over
> externally-computed probabilistic facts — it does **not** yet use DeepProbLog's integrated
> neural-predicate / end-to-end backprop training loop. Calibration is currently done with a
> lightweight external step (Platt scaling on each predicate), not gradient descent through the
> logic program. Wiring up true end-to-end DeepProbLog training is the natural next lab exercise
> (see `src/nesy/` for where the neural predicates would plug in).

## Dataset: Factify

We use [Factify](https://github.com/Shreyashm16/Factify) / [Factify 2](https://github.com/surya1701/Factify-2.0):
real claim/document pairs (image + text) scraped from verified news sources, labeled
`Support_Text`, `Support_Multimodal`, `Insufficient_Text`, `Insufficient_Multimodal`, `Refute`
(mapped down to `Support` / `Unverified` / `Refute` for this demo).

**Access is gated** — the dataset is not on Hugging Face or a public `wget` link. You must
register via the DE-FACTIFY shared-task organizers' form to get a download link, then place
`factify_train.csv` in `data/`. Because the images are Twitter-sourced from 2021–2022, expect
real link rot — the download script accounts for this (see below).

## Repository layout

```
.
├── Dockerfile
├── requirements.txt
├── scripts/
│   └── start_bash.sh          # launches the "nesy" container
├── src/
│   ├── data_download/
│   │   └── download_factify_images.py   # link-rot-aware image downloader
│   └── nesy/
│       ├── zero_shot_classify.py        # frozen-model baseline, no calibration
│       ├── tuning_classify.py           # single tune/test split + Platt scaling
│       ├── calibration_curve.py         # mean+/-std sweep over tuning-set size,
│       │                                #   majority baseline, final full-pool calibration
│       └── calibration_curve_negation.py # experimental 3rd predicate (negation LM)
└── data/                        # gitignored — CSV, images, and generated outputs live here
```

## Setup

1. Build and launch the container:
```bash
   docker build -t nesy:latest .
   ./scripts/start_bash.sh
```
   CPU-only by default; picks up GPU passthrough automatically if `DEVICE` is set in `.env`
   and `nvidia-smi` is available on the host. Requires driver ≥570.x on the host for
   Blackwell-class GPUs (RTX 50-series).

2. Register for Factify access, drop `factify_train_500.csv` into `/workdir/data/`.

3. Once inside the interactive container (`start_bash.sh` drops you into a shell), download a
   working sample of images by running the link-rot-aware downloader:
```bash
   ./src/data_download/download_factify_images.py
```
   The script defaults to reading `factify_train_500.csv` and keeps scanning past dead links
   until it hits the target count of *successful* pairs.

## Usage

Run in this order — each script builds on outputs from the previous one:

```bash
# 1. Zero-shot baseline, no calibration
python3 src/nesy/zero_shot_classify.py --save-predictions data/zero_shot_preds.csv

# 2. Single tune/test split with Platt-scaling calibration
python3 src/nesy/tuning_classify.py

# 3. The real experiment: majority baseline, repeated stratified draws (mean+/-std, never a
#    cherry-picked best run), per-class probability trends, one principled full-pool calibration
python3 src/nesy/calibration_curve.py

# 4. Experimental: adds a negation-detection predicate via a small generative LM
python3 src/nesy/calibration_curve_negation.py --preview-n 8   # sanity-check the LM's answers first
python3 src/nesy/calibration_curve_negation.py                 # full run
```

Key flags across the `nesy` scripts: `--device` (`cuda`/`cpu`), `--nli-model`, `--top-k-sentences`
(sentence-retrieval window fed to the NLI model instead of a blind 512-token document
truncation), `--test-size`, `--tune-sizes`, `--n-seeds`.

## Results (500 downloaded pairs, 150-row fixed held-out test set)

| Configuration | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.44 | 0.20 |
| Raw (uncalibrated) | ~0.52 | ~0.47 |
| **Calibrated (full tuning pool)** | **0.55** | **0.52** |

Macro F1 is the number to report, not accuracy — it's the one a naive majority-class predictor
can't cheat by riding the largest class. The calibrated model beats the majority floor by **~2.5x
on macro F1**.

## Known limitations (documented, not hidden)

- **CLIP image-image similarity carries almost no signal** for this task — claim and document
  photos are frequently different images of the same event, not near-duplicates.
- **NLI entailment can be fooled by lexical overlap** on some `Refute` rows — a document can
  share vocabulary with a claim while actually negating it. This motivated the negation-predicate
  experiment.
- **The negation predicate (`calibration_curve_negation.py`) is a documented negative result.**
  Three distinct implementation bugs were found and fixed along the way (wrong verbalizer token
  ID, an uncorrected model default-answer bias, and — for Qwen3 — the model's default
  chain-of-thought mode swallowing the answer token) before getting a trustworthy measurement.
  Even after all three fixes, it did not beat the two-predicate baseline. Kept in the repo as a
  worked example of how much can go wrong between "call an LLM" and "trust its output as a
  calibrated probability."

## References

- Factify: [Mishra et al., DE-FACTIFY @ AAAI](https://github.com/Shreyashm16/Factify)
- ProbLog: [De Raedt, Kimmig & Toivonen, IJCAI 2007](https://dtai.cs.kuleuven.be/problog/)
- DeepProbLog: [Manhaeve et al., NeurIPS 2018](https://github.com/ML-KULeuven/deepproblog)
- NLI model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- CLIP: `open_clip`, ViT-L/14, `laion2b_s32b_b82k`