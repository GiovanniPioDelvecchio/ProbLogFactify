#!/usr/bin/env python3
r"""
/workdir/src/nesy/calibration_curve_negation.py

v3: adds prior-bias debiasing for the negation LM (Zhao et al. 2021,
"Calibrate Before Use"). Preview logs showed the model answering 'No'
on almost every row regardless of content -- including rows where the
debunking language ("morphed image") was literally present in the
premise -- which is the signature of a default-answer bias baked into
the model's instruction tuning, not a content-reading failure.

Fix: query the model once with a CONTENT-FREE placeholder premise/claim
to measure its default P(Yes), then debias every real prediction by
dividing out that prior:

    debiased = (raw/prior) / (raw/prior + (1-raw)/(1-prior))

Both --preview-n mode and the full pipeline now report the measured
prior, and the full pipeline calibrates on the DEBIASED negation score.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from problog import get_evaluatable
from problog.program import PrologString
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from zero_shot_classify import (
    load_clip, load_nli, load_retriever, load_filtered_csv,
    raise_csv_field_limit, retrieve_relevant_sentences, image_match_prob,
    nli_probs, log, LABEL_MAP,
)

NEGATION_MODEL_NAME = "Qwen/Qwen3-8B"

PROBLOG_TEMPLATE = """
{entail_p}::text_entails.
{contra_p}::text_contradicts.
{image_p}::image_matches.
{negate_p}::claim_negated.

0.9::supported :- text_entails, \\+ claim_negated.
0.5::supported :- text_entails, image_matches, \\+ claim_negated.

refuted :- text_contradicts.
refuted :- claim_negated.
0.4::refuted :- \\+ image_matches, \\+ text_entails, \\+ text_contradicts, \\+ claim_negated.

unverified :- \\+ supported, \\+ refuted.

query(supported).
query(refuted).
query(unverified).
"""


def load_negation_lm(device: str):
    tokenizer = AutoTokenizer.from_pretrained(NEGATION_MODEL_NAME)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(NEGATION_MODEL_NAME, torch_dtype=dtype)
    model.eval().to(device)
    return model, tokenizer


def _single_token_id(tokenizer, word: str) -> int:
    for candidate in (f" {word}", word):   # space-prefixed first -- matches what the model
                                             # actually continues with after "Answer:"
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return tokenizer.encode(word, add_special_tokens=False)[0]


def _build_negation_prompt(lm_tokenizer, premise: str, claim: str) -> str:
    messages = [
        {"role": "system",
         "content": "You are a careful fact-checking assistant. Answer with exactly one word: Yes or No."},
        {"role": "user", "content": (
            f"Passage: {premise}\n\nClaim: {claim}\n\n"
            f"Does the passage indicate that the claim is false, fake, staged, morphed, "
            f"doctored, out of context, miscaptioned, or misattributed -- in other words, "
            f"does it debunk the claim in any way? Answer Yes or No."
        )},
    ]
    return lm_tokenizer.apply_chat_template(messages, tokenize=False, 
    add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def preview_generation(lm_model, lm_tokenizer, premise: str, claim: str, device: str) -> str:
    prompt = _build_negation_prompt(lm_tokenizer, premise, claim) + "Answer:"
    inputs = lm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = lm_model.generate(**inputs, max_new_tokens=8, do_sample=False)
    return lm_tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def raw_negation_prob(lm_model, lm_tokenizer, premise: str, claim: str, device: str,
                       yes_id: int, no_id: int) -> float:
    prompt = _build_negation_prompt(lm_tokenizer, premise, claim) + "Answer:"
    inputs = lm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    logits = lm_model(**inputs).logits[0, -1]
    pair_logits = torch.stack([logits[yes_id], logits[no_id]])
    probs = F.softmax(pair_logits, dim=-1)
    return probs[0].item()


def measure_negation_prior(lm_model, lm_tokenizer, device: str, yes_id: int, no_id: int) -> float:
    """Content-free placeholder query -- measures the model's default
    P(Yes) with nothing real to reason about. Far from 0.5 means a real
    default-answer bias baked into this model/prompt combination."""
    return raw_negation_prob(lm_model, lm_tokenizer, "N/A", "N/A", device, yes_id, no_id)


def debias_prob(raw_p: float, prior_p: float, eps: float = 1e-6) -> float:
    """Zhao et al. 2021 'Calibrate Before Use': divide out the model's
    measured default-answer bias, then renormalize back to a valid
    probability."""
    prior_p = min(max(prior_p, eps), 1 - eps)
    raw_p = min(max(raw_p, eps), 1 - eps)
    yes_odds = raw_p / prior_p
    no_odds = (1 - raw_p) / (1 - prior_p)
    return yes_odds / (yes_odds + no_odds)


def compute_raw_signals(df, mapping, image_dir, clip_model, clip_preprocess,
                         nli_model, nli_tokenizer, retriever,
                         negation_lm, negation_tokenizer, yes_id, no_id, negation_prior,
                         top_k_sentences, device):
    records = []
    for entry_id, files in mapping.items():
        entry_id_int = int(entry_id)
        if entry_id_int not in df.index:
            log.warning(f"Row {entry_id}: not found in parsed CSV, skipping.")
            continue
        row = df.loc[entry_id_int]

        raw_label = row["Category"]
        gold_label = LABEL_MAP.get(raw_label)
        if gold_label is None:
            log.warning(f"Row {entry_id}: unmapped Category '{raw_label}', skipping.")
            continue

        claim_img_path = image_dir / files["claim_image"]
        doc_img_path = image_dir / files["document_image"]
        image_p = image_match_prob(clip_model, clip_preprocess, claim_img_path, doc_img_path, device)

        claim_text = str(row["claim"])
        premise = retrieve_relevant_sentences(retriever, claim_text, str(row["document"]), top_k_sentences)

        nli = nli_probs(nli_model, nli_tokenizer, premise=premise, hypothesis=claim_text, device=device)
        entail_p = nli.get("entailment", 0.0)
        contra_p = nli.get("contradiction", 0.0)

        negate_p_raw = raw_negation_prob(negation_lm, negation_tokenizer, premise, claim_text, device, yes_id, no_id)
        negate_p_debiased = debias_prob(negate_p_raw, negation_prior)

        records.append({
            "entry_id": entry_id, "gold": gold_label,
            "entail_p": entail_p, "contra_p": contra_p,
            "image_p": image_p,
            "negate_p_raw": negate_p_raw, "negate_p": negate_p_debiased,
        })
        log.info(f"Row {entry_id}: gold={gold_label:<10} "
                 f"(entail={entail_p:.2f} contra={contra_p:.2f} img={image_p:.2f} "
                 f"negate_raw={negate_p_raw:.2f} negate_debiased={negate_p_debiased:.2f})")
    return pd.DataFrame(records)


SIGNAL_TARGETS = {
    "entail_p": lambda gold: (gold == "Support").astype(int),
    "contra_p": lambda gold: (gold == "Refute").astype(int),
    "image_p": lambda gold: (gold == "Support").astype(int),
    "negate_p": lambda gold: (gold == "Refute").astype(int),
}


def fit_platt_scalers(tune_df: pd.DataFrame) -> dict:
    scalers = {}
    for signal, target_fn in SIGNAL_TARGETS.items():
        X = tune_df[[signal]].to_numpy()
        y = target_fn(tune_df["gold"]).to_numpy()
        if len(set(y)) < 2:
            log.warning(f"Tuning set has only one class for '{signal}' target -- using raw probability.")
            scalers[signal] = None
            continue
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(X, y)
        scalers[signal] = clf
    return scalers


def apply_calibration(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    df = df.copy()
    for signal, clf in scalers.items():
        df[f"{signal}_cal"] = df[signal] if clf is None else clf.predict_proba(df[[signal]].to_numpy())[:, 1]
    return df


def run_problog_v2(entail_p, contra_p, image_p, negate_p) -> dict:
    eps = 1e-4
    entail_p, contra_p, image_p, negate_p = (
        min(max(v, eps), 1 - eps) for v in (entail_p, contra_p, image_p, negate_p)
    )
    program = PROBLOG_TEMPLATE_V2.format(
        entail_p=entail_p, contra_p=contra_p, image_p=image_p, negate_p=negate_p,
    )
    result = get_evaluatable().create_from(PrologString(program)).evaluate()
    return {str(k): v for k, v in result.items()}


def predict_label(query_result: dict) -> str:
    scores = {
        "Support": query_result.get("supported", 0.0),
        "Refute": query_result.get("refuted", 0.0),
        "Unverified": query_result.get("unverified", 0.0),
    }
    return max(scores, key=scores.get)


def score(df: pd.DataFrame, entail_col, contra_col, image_col, negate_col) -> dict:
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        query_result = run_problog_v2(row[entail_col], row[contra_col], row[image_col], row[negate_col])
        y_pred.append(predict_label(query_result))
        y_true.append(row["gold"])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def majority_baseline(train_labels, test_labels) -> dict:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(pd.DataFrame({"x": range(len(train_labels))}), train_labels)
    y_pred = dummy.predict(pd.DataFrame({"x": range(len(test_labels))}))
    return {
        "majority_class": dummy.classes_[dummy.class_prior_.argmax()],
        "accuracy": accuracy_score(test_labels, y_pred),
        "macro_f1": f1_score(test_labels, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(test_labels, y_pred, average="weighted", zero_division=0),
    }


def draw_stratified_subset(pool_df, size, seed):
    if size >= len(pool_df):
        return pool_df
    try:
        subset, _ = train_test_split(pool_df, train_size=size, stratify=pool_df["gold"], random_state=seed)
        return subset
    except ValueError as e:
        log.warning(f"Stratified sample failed at size={size}, seed={seed} ({e}); using plain random sample.")
        return pool_df.sample(n=size, random_state=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/workdir/data/factify_train.csv"))
    parser.add_argument("--mapping-file", type=Path, default=Path("/workdir/data/image_mapping.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("/workdir/data/images"))
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nli-model", type=str,
                         default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--top-k-sentences", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--tune-sizes", type=str, default="10,20,50,100,200,350")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path("/workdir/data/calibration_curve_negation.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("/workdir/data/calibration_curve_negation.png"))
    parser.add_argument("--prob-summary-csv", type=Path,
                         default=Path("/workdir/data/calibrated_probability_summary_negation.csv"))
    parser.add_argument("--final-calibrated-csv", type=Path,
                         default=Path("/workdir/data/final_calibrated_probs_negation.csv"))
    parser.add_argument("--preview-n", type=int, default=0,
                         help="If >0, print the negation LM's real Yes/No answer plus the measured "
                              "prior bias on this many rows, then exit -- skips CLIP/NLI/full pipeline.")
    args = parser.parse_args()

    raise_csv_field_limit()

    with open(args.mapping_file) as f:
        mapping = json.load(f)
    wanted_row_indices = {int(k) for k in mapping.keys()}
    df = load_filtered_csv(args.csv, args.sep, wanted_row_indices)
    log.info(f"Parsed {len(df)} rows (expected {len(wanted_row_indices)})")

    # -------------------- Preview mode: cheap sanity check, then exit --------------------
    if args.preview_n > 0:
        log.info("Preview mode: loading retriever + negation LM only (skipping CLIP/NLI)...")
        retriever = load_retriever(args.device)
        negation_lm, negation_tokenizer = load_negation_lm(args.device)
        yes_id = _single_token_id(negation_tokenizer, "Yes")
        no_id = _single_token_id(negation_tokenizer, "No")

        prior = measure_negation_prior(negation_lm, negation_tokenizer, args.device, yes_id, no_id)
        print(f"\nMeasured default-answer prior P(Yes | content-free placeholder) = {prior:.4f}")
        print("(0.5 = no default lean. Far from 0.5 means the model has a baked-in "
              "answer bias independent of content -- this is what debiasing corrects for.)")

        shown = 0
        for entry_id, files in mapping.items():
            if shown >= args.preview_n:
                break
            entry_id_int = int(entry_id)
            if entry_id_int not in df.index:
                continue
            row = df.loc[entry_id_int]
            gold_label = LABEL_MAP.get(row["Category"])
            if gold_label is None:
                continue

            claim_text = str(row["claim"])
            premise = retrieve_relevant_sentences(retriever, claim_text, str(row["document"]), args.top_k_sentences)
            generation = preview_generation(negation_lm, negation_tokenizer, premise, claim_text, args.device)
            raw_p = raw_negation_prob(negation_lm, negation_tokenizer, premise, claim_text,
                                       args.device, yes_id, no_id)
            debiased_p = debias_prob(raw_p, prior)

            print(f"\n[{entry_id}] gold={gold_label}")
            print(f"  claim:      {claim_text[:150]!r}")
            print(f"  premise:    {premise[:250]!r}")
            print(f"  model says: {generation!r}  (raw P(Yes)={raw_p:.3f}, debiased={debiased_p:.3f})")
            shown += 1

        print(f"\nShowed {shown} rows.")
        return

    # -------------------- Full pipeline --------------------
    log.info("Loading CLIP...")
    clip_model, clip_preprocess = load_clip(args.device)
    log.info(f"Loading NLI model ({args.nli_model})...")
    nli_model, nli_tokenizer = load_nli(args.device, args.nli_model)
    log.info("Loading sentence retriever...")
    retriever = load_retriever(args.device)
    log.info(f"Loading negation LM ({NEGATION_MODEL_NAME})...")
    negation_lm, negation_tokenizer = load_negation_lm(args.device)
    yes_id = _single_token_id(negation_tokenizer, "Yes")
    no_id = _single_token_id(negation_tokenizer, "No")
    print(repr(negation_tokenizer.decode([yes_id])), repr(negation_tokenizer.decode([no_id])))

    negation_prior = measure_negation_prior(negation_lm, negation_tokenizer, args.device, yes_id, no_id)
    log.info(f"Measured negation-LM default-answer prior: P(Yes)={negation_prior:.4f} "
             f"(0.5 = unbiased; all downstream negate_p values are debiased against this)")

    signals_df = compute_raw_signals(
        df, mapping, args.image_dir, clip_model, clip_preprocess,
        nli_model, nli_tokenizer, retriever, negation_lm, negation_tokenizer,
        yes_id, no_id, negation_prior, args.top_k_sentences, args.device,
    )
    log.info(f"Computed raw signals for {len(signals_df)} rows total.")

    pool_df, test_df = train_test_split(
        signals_df, test_size=args.test_size, stratify=signals_df["gold"], random_state=args.seed,
    )
    log.info(f"Fixed test set: {len(test_df)} rows. Pool for tuning draws: {len(pool_df)} rows.")

    maj = majority_baseline(pool_df["gold"], test_df["gold"])
    print(f"\n--- Majority-class baseline (always predict '{maj['majority_class']}') ---")
    print(f"  accuracy={maj['accuracy']:.4f}  macro_f1={maj['macro_f1']:.4f}  "
          f"weighted_f1={maj['weighted_f1']:.4f}")

    raw_scores = score(test_df, "entail_p", "contra_p", "image_p", "negate_p")
    print(f"\n--- Raw (uncalibrated, but negation already debiased) baseline, fixed test set (n={len(test_df)}) ---")
    print(f"  accuracy={raw_scores['accuracy']:.4f}  macro_f1={raw_scores['macro_f1']:.4f}  "
          f"weighted_f1={raw_scores['weighted_f1']:.4f}")
    print("  per-class mean signal (negate_p_raw = pre-debias, negate_p = post-debias):")
    print(test_df.groupby("gold")[["entail_p", "contra_p", "image_p", "negate_p_raw", "negate_p"]].mean())

    tune_sizes = [int(x) for x in args.tune_sizes.split(",")]
    per_run_scores = []
    all_calibrated_rows = []

    for size in tune_sizes:
        if size > len(pool_df):
            log.warning(f"Requested tune_size={size} exceeds pool size {len(pool_df)}, skipping.")
            continue
        for i in range(args.n_seeds):
            draw_seed = args.seed + 1000 + size * 100 + i
            tune_subset = draw_stratified_subset(pool_df, size, draw_seed)
            scalers = fit_platt_scalers(tune_subset)
            test_df_cal = apply_calibration(test_df, scalers)
            cal_scores = score(test_df_cal, "entail_p_cal", "contra_p_cal", "image_p_cal", "negate_p_cal")
            per_run_scores.append({"tune_size": size, "seed": draw_seed, **cal_scores})

            tagged = test_df_cal[["gold", "entail_p_cal", "contra_p_cal", "image_p_cal", "negate_p_cal"]].copy()
            tagged["tune_size"] = size
            all_calibrated_rows.append(tagged)
        log.info(f"tune_size={size}: completed {args.n_seeds} draws")

    per_run_df = pd.DataFrame(per_run_scores)
    agg = per_run_df.groupby("tune_size")[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    print(f"\n--- Calibration learning curve (debiased negation predicate), mean +/- std over "
          f"{args.n_seeds} draws per size (fixed test set, n={len(test_df)}) ---")
    print(agg.to_string(index=False))
    per_run_df.to_csv(args.output_csv, index=False)

    all_calibrated_df = pd.concat(all_calibrated_rows, ignore_index=True)
    prob_summary = all_calibrated_df.groupby(["tune_size", "gold"])[
        ["entail_p_cal", "contra_p_cal", "image_p_cal", "negate_p_cal"]
    ].mean()
    print(f"\n--- Average calibrated probability per class, per tuning size ---")
    print(prob_summary.to_string())
    prob_summary.to_csv(args.prob_summary_csv)

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, marker in [("accuracy", "o"), ("macro_f1", "s"), ("weighted_f1", "^")]:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        ax.plot(agg["tune_size"], agg[mean_col], marker=marker, label=metric)
        ax.fill_between(agg["tune_size"], agg[mean_col] - agg[std_col].fillna(0),
                         agg[mean_col] + agg[std_col].fillna(0), alpha=0.15)
    ax.axhline(y=raw_scores["macro_f1"], color="gray", linestyle="--", alpha=0.6, label="Raw macro F1")
    ax.axhline(y=maj["macro_f1"], color="firebrick", linestyle=":", alpha=0.6, label="Majority-class floor")
    ax.set_xlabel("Tuning set size")
    ax.set_ylabel(f"Score on fixed test set (mean +/- std, {args.n_seeds} draws)")
    ax.set_title(f"Debiased negation predicate: calibration vs. tuning size (test n={len(test_df)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)

    print(f"\n=== Final calibration: fit on the full tuning pool (n={len(pool_df)}) ===")
    final_scalers = fit_platt_scalers(pool_df)
    for signal, clf in final_scalers.items():
        if clf is not None:
            print(f"  {signal}: coef={clf.coef_[0][0]:+.3f}  intercept={clf.intercept_[0]:+.3f}")

    final_test_cal = apply_calibration(test_df, final_scalers)
    final_scores = score(final_test_cal, "entail_p_cal", "contra_p_cal", "image_p_cal", "negate_p_cal")
    print(f"\nFinal calibrated performance (n={len(test_df)}):")
    print(f"  accuracy={final_scores['accuracy']:.4f}  macro_f1={final_scores['macro_f1']:.4f}  "
          f"weighted_f1={final_scores['weighted_f1']:.4f}")
    print(f"  (majority floor: accuracy={maj['accuracy']:.4f}, macro_f1={maj['macro_f1']:.4f})")

    print("\nFinal per-class mean calibrated probability:")
    print(final_test_cal.groupby("gold")[["entail_p_cal", "contra_p_cal", "image_p_cal", "negate_p_cal"]].mean())

    final_test_cal.to_csv(args.final_calibrated_csv, index=False)


if __name__ == "__main__":
    main()