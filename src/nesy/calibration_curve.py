#!/usr/bin/env python3
"""
/workdir/src/nesy/calibration_curve.py

Reports AVERAGES across repeated tuning draws -- never a cherry-picked
best run, which is a biased estimate (the max of N noisy draws beats
the true mean of the method by construction, even if the method does
nothing). Structure:

  1. Majority-class baseline on the fixed test set (the floor any
     result needs to clear to mean anything).
  2. Raw/uncalibrated ProbLog baseline (single deterministic value).
  3. Learning curve: mean +/- std across --n-seeds draws per tuning
     size, on the same fixed test set throughout.
  4. Per-(tune_size, class) average calibrated probabilities, averaged
     across seeds AND across rows -- shows whether class separation
     actually improves with more tuning data, not just the final score.
  5. A single "final" calibration fit on the ENTIRE available tuning
     pool (chosen because it uses all the data, not because it scored
     best) -- this is the one to actually report/ship.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from zero_shot_classify import (
    load_clip, load_nli, load_retriever, load_filtered_csv,
    raise_csv_field_limit, run_problog, predict_label, log,
)
from tuning_classify import compute_raw_signals, fit_platt_scalers, apply_calibration


def score(df: pd.DataFrame, entail_col: str, contra_col: str, image_col: str) -> dict:
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        query_result = run_problog(row[entail_col], row[contra_col], row[image_col])
        y_pred.append(predict_label(query_result))
        y_true.append(row["gold"])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def majority_baseline(train_labels: pd.Series, test_labels: pd.Series) -> dict:
    """The floor: always predict the most common class. Any calibrated
    result needs to clear this by a meaningful margin to mean anything."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(pd.DataFrame({"x": range(len(train_labels))}), train_labels)
    y_pred = dummy.predict(pd.DataFrame({"x": range(len(test_labels))}))
    return {
        "majority_class": dummy.classes_[dummy.class_prior_.argmax()],
        "accuracy": accuracy_score(test_labels, y_pred),
        "macro_f1": f1_score(test_labels, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(test_labels, y_pred, average="weighted", zero_division=0),
    }


def draw_stratified_subset(pool_df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size >= len(pool_df):
        return pool_df
    try:
        subset, _ = train_test_split(
            pool_df, train_size=size, stratify=pool_df["gold"], random_state=seed,
        )
        return subset
    except ValueError as e:
        log.warning(f"Stratified sample failed at size={size}, seed={seed} ({e}); using plain random sample.")
        return pool_df.sample(n=size, random_state=seed)


def print_calibration_summary(scalers: dict, label: str):
    print(f"\nFitted Platt-scaling coefficients [{label}] "
          f"(calibrated = sigmoid(coef * raw + intercept)):")
    for signal, clf in scalers.items():
        if clf is None:
            print(f"  {signal}: not calibrated (tuning set had only one class for this target)")
        else:
            print(f"  {signal}: coef={clf.coef_[0][0]:+.3f}  intercept={clf.intercept_[0]:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/workdir/data/factify_train.csv"))
    parser.add_argument("--mapping-file", type=Path, default=Path("/workdir/data/image_mapping.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("/workdir/data/images"))
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nli-model", type=str,
                         default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--top-k-sentences", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--tune-sizes", type=str, default="10,20,50,100,200,350")
    parser.add_argument("--n-seeds", type=int, default=5,
                         help="Number of independent tuning draws averaged per tune_size.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the fixed test/pool split.")
    parser.add_argument("--output-csv", type=Path, default=Path("/workdir/data/calibration_curve.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("/workdir/data/calibration_curve.png"))
    parser.add_argument("--prob-summary-csv", type=Path,
                         default=Path("/workdir/data/calibrated_probability_summary.csv"))
    parser.add_argument("--final-calibrated-csv", type=Path,
                         default=Path("/workdir/data/final_calibrated_probs.csv"),
                         help="Calibrated test-set probabilities from the full-pool final calibration.")
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
    log.info(f"Computed raw signals for {len(signals_df)} rows total.")

    pool_df, test_df = train_test_split(
        signals_df, test_size=args.test_size, stratify=signals_df["gold"], random_state=args.seed,
    )
    log.info(f"Fixed test set: {len(test_df)} rows (held constant for the whole sweep). "
             f"Pool for tuning draws: {len(pool_df)} rows.")

    # -------------------- 1. Majority-class baseline --------------------
    maj = majority_baseline(pool_df["gold"], test_df["gold"])
    print(f"\n--- Majority-class baseline (always predict '{maj['majority_class']}') ---")
    print(f"  test set class distribution:\n{test_df['gold'].value_counts(normalize=True).to_string()}")
    print(f"  accuracy={maj['accuracy']:.4f}  macro_f1={maj['macro_f1']:.4f}  "
          f"weighted_f1={maj['weighted_f1']:.4f}")

    # -------------------- 2. Raw / uncalibrated baseline --------------------
    raw_scores = score(test_df, "entail_p", "contra_p", "image_p")
    print(f"\n--- Raw (uncalibrated) baseline, fixed test set (n={len(test_df)}) ---")
    print(f"  accuracy={raw_scores['accuracy']:.4f}  macro_f1={raw_scores['macro_f1']:.4f}  "
          f"weighted_f1={raw_scores['weighted_f1']:.4f}")
    print("  per-class mean raw signal:")
    print(test_df.groupby("gold")[["entail_p", "contra_p", "image_p"]].mean())

    # -------------------- 3 & 4. Sweep: repeated draws, averaged --------------------
    tune_sizes = [int(x) for x in args.tune_sizes.split(",")]
    per_run_scores = []
    all_calibrated_rows = []  # for the per-(tune_size, class) probability summary

    for size in tune_sizes:
        if size > len(pool_df):
            log.warning(f"Requested tune_size={size} exceeds pool size {len(pool_df)}, skipping.")
            continue
        for i in range(args.n_seeds):
            draw_seed = args.seed + 1000 + size * 100 + i
            tune_subset = draw_stratified_subset(pool_df, size, draw_seed)
            scalers = fit_platt_scalers(tune_subset)
            test_df_cal = apply_calibration(test_df, scalers)
            cal_scores = score(test_df_cal, "entail_p_cal", "contra_p_cal", "image_p_cal")

            per_run_scores.append({"tune_size": size, "seed": draw_seed, **cal_scores})

            tagged = test_df_cal[["gold", "entail_p_cal", "contra_p_cal", "image_p_cal"]].copy()
            tagged["tune_size"] = size
            tagged["seed"] = draw_seed
            all_calibrated_rows.append(tagged)

        log.info(f"tune_size={size}: completed {args.n_seeds} draws")

    per_run_df = pd.DataFrame(per_run_scores)
    agg = per_run_df.groupby("tune_size")[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    print(f"\n--- Calibration learning curve, mean +/- std over {args.n_seeds} draws per size "
          f"(fixed test set, n={len(test_df)}) ---")
    print(agg.to_string(index=False))
    per_run_df.to_csv(args.output_csv, index=False)
    log.info(f"Per-run scores written to {args.output_csv}")

    all_calibrated_df = pd.concat(all_calibrated_rows, ignore_index=True)
    prob_summary = all_calibrated_df.groupby(["tune_size", "gold"])[
        ["entail_p_cal", "contra_p_cal", "image_p_cal"]
    ].mean()
    print(f"\n--- Average calibrated probability per class, per tuning size "
          f"(averaged over {args.n_seeds} draws and all test rows) ---")
    print(prob_summary.to_string())
    prob_summary.to_csv(args.prob_summary_csv)
    log.info(f"Probability summary written to {args.prob_summary_csv}")

    # -------------------- Plot --------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, marker in [("accuracy", "o"), ("macro_f1", "s"), ("weighted_f1", "^")]:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        ax.plot(agg["tune_size"], agg[mean_col], marker=marker, label=metric)
        ax.fill_between(agg["tune_size"], agg[mean_col] - agg[std_col].fillna(0),
                         agg[mean_col] + agg[std_col].fillna(0), alpha=0.15)
    ax.axhline(y=raw_scores["macro_f1"], color="gray", linestyle="--", alpha=0.6,
               label="Raw macro F1 (no calibration)")
    ax.axhline(y=maj["macro_f1"], color="firebrick", linestyle=":", alpha=0.6,
               label="Majority-class macro F1 (floor)")
    ax.set_xlabel("Tuning set size (rows used to fit calibration)")
    ax.set_ylabel(f"Score on fixed test set (mean +/- std, {args.n_seeds} draws)")
    ax.set_title(f"Calibration benefit vs. tuning set size (test n={len(test_df)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    log.info(f"Plot written to {args.output_plot}")

    # -------------------- 5. Final calibration: fit on the FULL pool --------------------
    # Chosen because it uses all available tuning data, not because it scored
    # best -- this is the one to actually report as "the model."
    print(f"\n=== Final calibration: fit on the full tuning pool (n={len(pool_df)}), "
          f"no selection by score ===")
    final_scalers = fit_platt_scalers(pool_df)
    print_calibration_summary(final_scalers, label=f"full pool, n={len(pool_df)}")

    final_test_cal = apply_calibration(test_df, final_scalers)
    final_scores = score(final_test_cal, "entail_p_cal", "contra_p_cal", "image_p_cal")
    print(f"\nFinal calibrated performance on fixed test set (n={len(test_df)}):")
    print(f"  accuracy={final_scores['accuracy']:.4f}  macro_f1={final_scores['macro_f1']:.4f}  "
          f"weighted_f1={final_scores['weighted_f1']:.4f}")
    print(f"  (majority-class floor: accuracy={maj['accuracy']:.4f}, macro_f1={maj['macro_f1']:.4f})")

    print("\nFinal per-class mean calibrated probability:")
    print(final_test_cal.groupby("gold")[["entail_p_cal", "contra_p_cal", "image_p_cal"]].mean())

    final_test_cal.to_csv(args.final_calibrated_csv, index=False)
    log.info(f"Final calibrated test-set probabilities written to {args.final_calibrated_csv}")


if __name__ == "__main__":
    main()