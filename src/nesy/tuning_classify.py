#!/usr/bin/env python3
"""
/workdir/src/nesy/tuning_classify.py

Splits the 100 Factify rows into a 20-row TUNING set and an 80-row TEST
set. Fits per-signal Platt scaling (1D logistic regression) on the
tuning set to recalibrate the raw entail_p / contra_p / image_p scores,
then evaluates BOTH raw and calibrated probabilities on the held-out
test set only, so the reported gain (if any) is honest -- the test
rows never participate in fitting anything.

Reuses the model-loading and per-row signal computation from
zero_shot_classify.py so the underlying CLIP/NLI/retrieval pipeline
stays identical; only the calibration step is new.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from zero_shot_classify import (
    LABEL_MAP,
    load_clip,
    load_nli,
    load_retriever,
    retrieve_relevant_sentences,
    image_match_prob,
    nli_probs,
    run_problog,
    predict_label,
    load_filtered_csv,
    raise_csv_field_limit,
    log,
)


def compute_raw_signals(df, mapping, image_dir, clip_model, clip_preprocess,
                         nli_model, nli_tokenizer, retriever, top_k_sentences, device):
    """One pass over all mapped rows, computing raw entail_p/contra_p/image_p
    for each. Kept separate from calibration/splitting so the expensive
    model forward passes only ever happen once."""
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

        records.append({
            "entry_id": entry_id, "gold": gold_label,
            "entail_p": entail_p, "contra_p": contra_p, "image_p": image_p,
        })
        log.info(f"Row {entry_id}: gold={gold_label:<10} "
                 f"(entail={entail_p:.2f} contra={contra_p:.2f} img={image_p:.2f})")
    return pd.DataFrame(records)


def fit_platt_scalers(tune_df: pd.DataFrame) -> dict:
    """Fit one 1D logistic regression per signal, mapping raw probability
    -> calibrated probability, using pseudo-targets derived from which
    class each signal is meant to indicate."""
    scalers = {}
    targets = {
        "entail_p": (tune_df["gold"] == "Support").astype(int),
        "contra_p": (tune_df["gold"] == "Refute").astype(int),
        "image_p": (tune_df["gold"] == "Support").astype(int),
    }
    for signal, target in targets.items():
        X = tune_df[[signal]].to_numpy()
        y = target.to_numpy()
        if len(np.unique(y)) < 2:
            log.warning(f"Tuning set has only one class for '{signal}' target -- "
                        f"skipping calibration for this signal, using raw probability.")
            scalers[signal] = None
            continue
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(X, y)
        scalers[signal] = clf
        log.info(f"Calibrated '{signal}': coef={clf.coef_[0][0]:.3f} intercept={clf.intercept_[0]:.3f}")
    return scalers


def apply_calibration(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    df = df.copy()
    for signal, clf in scalers.items():
        if clf is None:
            df[f"{signal}_cal"] = df[signal]
        else:
            df[f"{signal}_cal"] = clf.predict_proba(df[[signal]].to_numpy())[:, 1]
    return df


def evaluate(df: pd.DataFrame, entail_col: str, contra_col: str, image_col: str, label: str):
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        query_result = run_problog(row[entail_col], row[contra_col], row[image_col])
        pred = predict_label(query_result)
        y_true.append(row["gold"])
        y_pred.append(pred)

    print(f"\n--- {label} (n={len(df)}) ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    labels = sorted(set(y_true) | set(y_pred))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels))
    return y_pred


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
    parser.add_argument("--tune-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-predictions", type=Path, default=None,
                         help="Optional path to save TEST-set predictions (calibrated).")
    args = parser.parse_args()

    raise_csv_field_limit()

    with open(args.mapping_file) as f:
        mapping = json.load(f)
    wanted_row_indices = {int(k) for k in mapping.keys()}
    df = load_filtered_csv(args.csv, args.sep, wanted_row_indices)
    log.info(f"Parsed {len(df)} rows (expected {len(wanted_row_indices)})")

    log.info("Loading CLIP...")
    clip_model, clip_preprocess = load_clip(args.device)
    log.info(f"Loading NLI model ({args.nli_model})...")
    nli_model, nli_tokenizer = load_nli(args.device, args.nli_model)
    log.info("Loading sentence retriever...")
    retriever = load_retriever(args.device)

    signals_df = compute_raw_signals(
        df, mapping, args.image_dir, clip_model, clip_preprocess,
        nli_model, nli_tokenizer, retriever, args.top_k_sentences, args.device,
    )

    tune_df, test_df = train_test_split(
        signals_df, train_size=args.tune_size, stratify=signals_df["gold"],
        random_state=args.seed,
    )
    log.info(f"Split: {len(tune_df)} tuning rows, {len(test_df)} test rows "
             f"(stratified by gold label, seed={args.seed})")

    # Baseline: raw probabilities, evaluated ONLY on the 80 test rows --
    # this is the honest zero-shot number for comparison, on the exact
    # same held-out rows the calibrated version will be scored on.
    evaluate(test_df, "entail_p", "contra_p", "image_p", "RAW (uncalibrated), test set only")

    scalers = fit_platt_scalers(tune_df)
    test_df_cal = apply_calibration(test_df, scalers)

    evaluate(test_df_cal, "entail_p_cal", "contra_p_cal", "image_p_cal", "CALIBRATED, test set only")

    print("\n--- Per-class mean signal strength (test set, raw vs calibrated) ---")
    print(test_df_cal.groupby("gold")[
        ["entail_p", "entail_p_cal", "contra_p", "contra_p_cal", "image_p", "image_p_cal"]
    ].mean())

    if args.save_predictions:
        test_df_cal.to_csv(args.save_predictions, index=False)
        log.info(f"Test-set predictions/signals written to {args.save_predictions}")


if __name__ == "__main__":
    main()