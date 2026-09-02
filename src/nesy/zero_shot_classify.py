#!/usr/bin/env python3
"""
/workdir/src/nesy/zero_shot_classify.py

Zero-shot neuro-symbolic baseline for Factify claim/document pairs.
No training: frozen CLIP (image similarity) + frozen NLI model (text
entailment/contradiction) feed fuzzy probabilistic facts into a fixed
ProbLog program that derives Support / Refute / Unverified.

v2: instead of truncating the full document to 512 tokens as the NLI
premise (which buries the one supporting/contradicting sentence in a
long article), retrieve the top-k most relevant sentences via a
lightweight sentence embedder and use only those as the premise.
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from problog.program import PrologString
from problog import get_evaluatable
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("zero_shot_classify")

LABEL_MAP = {
    "Support_Multimodal": "Support",
    "Support_Text": "Support",
    "Insufficient_Multimodal": "Unverified",
    "Insufficient_Text": "Unverified",
    "Refute": "Refute",
}

PROBLOG_TEMPLATE = """
{entail_p}::text_entails.
{contra_p}::text_contradicts.
{image_p}::image_matches.

0.9::supported :- text_entails.
0.5::supported :- text_entails, image_matches.

refuted :- text_contradicts.
0.4::refuted :- \\+ image_matches, \\+ text_entails, \\+ text_contradicts.

unverified :- \\+ supported, \\+ refuted.

query(supported).
query(refuted).
query(unverified).
"""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def raise_csv_field_limit():
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)


def split_sentences(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    # drop very short fragments (headers, bylines, stray punctuation)
    return [s for s in sentences if len(s.split()) >= 4]


def load_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    model.eval().to(device)
    return model, preprocess


def load_nli(device: str, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(device)
    return model, tokenizer


def load_retriever(device: str):
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def retrieve_relevant_sentences(retriever: SentenceTransformer, claim: str, document: str, top_k: int) -> str:
    """Return the top_k document sentences most similar to the claim,
    joined into a short premise, in their original document order."""
    sentences = split_sentences(document)
    if not sentences:
        return str(document)[:1000]  # degrade gracefully rather than crash
    if len(sentences) <= top_k:
        return " ".join(sentences)

    claim_emb = retriever.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    sent_embs = retriever.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    sims = (sent_embs @ claim_emb).cpu()
    top_indices = sorted(sims.topk(top_k).indices.tolist())  # keep original order
    return " ".join(sentences[i] for i in top_indices)


@torch.no_grad()
def image_match_prob(clip_model, preprocess, path_a: Path, path_b: Path, device: str) -> float:
    img_a = preprocess(Image.open(path_a).convert("RGB")).unsqueeze(0).to(device)
    img_b = preprocess(Image.open(path_b).convert("RGB")).unsqueeze(0).to(device)
    feat_a = F.normalize(clip_model.encode_image(img_a), dim=-1)
    feat_b = F.normalize(clip_model.encode_image(img_b), dim=-1)
    cosine_sim = (feat_a @ feat_b.T).item()
    return max(0.0, min(1.0, (cosine_sim + 1) / 2))


@torch.no_grad()
def nli_probs(nli_model, tokenizer, premise: str, hypothesis: str, device: str) -> dict:
    inputs = tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = nli_model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1)
    id2label = nli_model.config.id2label
    return {id2label[i].lower(): probs[i].item() for i in range(len(probs))}


def run_problog(entail_p: float, contra_p: float, image_p: float) -> dict:
    eps = 1e-4
    entail_p = min(max(entail_p, eps), 1 - eps)
    contra_p = min(max(contra_p, eps), 1 - eps)
    image_p = min(max(image_p, eps), 1 - eps)
    program = PROBLOG_TEMPLATE.format(entail_p=entail_p, contra_p=contra_p, image_p=image_p)
    result = get_evaluatable().create_from(PrologString(program)).evaluate()
    return {str(k): v for k, v in result.items()}


def predict_label(query_result: dict) -> str:
    scores = {
        "Support": query_result.get("supported", 0.0),
        "Refute": query_result.get("refuted", 0.0),
        "Unverified": query_result.get("unverified", 0.0),
    }
    return max(scores, key=scores.get)


def load_filtered_csv(csv_path: Path, sep: str, wanted_row_indices: set[int]) -> pd.DataFrame:
    def _skip(line_no: int) -> bool:
        if line_no == 0:
            return False
        return (line_no - 1) not in wanted_row_indices

    df = pd.read_csv(csv_path, sep=sep, quotechar='"', engine="python", on_bad_lines="warn", skiprows=_skip)
    df.index = sorted(wanted_row_indices)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/workdir/data/factify_train.csv"))
    parser.add_argument("--mapping-file", type=Path, default=Path("/workdir/data/image_mapping.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("/workdir/data/images"))
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nli-model", type=str,
                         default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--top-k-sentences", type=int, default=3,
                         help="Number of document sentences to retrieve as the NLI premise.")
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    raise_csv_field_limit()

    if args.inspect_only:
        df = pd.read_csv(args.csv, sep=args.sep, quotechar='"', engine="python", on_bad_lines="warn")
        print("Unique Category values found in CSV:")
        print(df["Category"].value_counts())
        return

    with open(args.mapping_file) as f:
        mapping = json.load(f)
    wanted_row_indices = {int(k) for k in mapping.keys()}
    log.info(f"Restricting CSV parse to {len(wanted_row_indices)} rows from {args.mapping_file}")

    df = load_filtered_csv(args.csv, args.sep, wanted_row_indices)
    log.info(f"Parsed {len(df)} rows (expected {len(wanted_row_indices)})")

    log.info("Loading CLIP...")
    clip_model, clip_preprocess = load_clip(args.device)
    log.info(f"Loading NLI model ({args.nli_model})...")
    nli_model, nli_tokenizer = load_nli(args.device, args.nli_model)
    log.info("Loading sentence retriever...")
    retriever = load_retriever(args.device)

    y_true, y_pred, records = [], [], []

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

        claim_img_path = args.image_dir / files["claim_image"]
        doc_img_path = args.image_dir / files["document_image"]

        image_p = image_match_prob(clip_model, clip_preprocess, claim_img_path, doc_img_path, args.device)

        claim_text = str(row["claim"])
        premise = retrieve_relevant_sentences(retriever, claim_text, str(row["document"]), args.top_k_sentences)

        nli = nli_probs(nli_model, nli_tokenizer, premise=premise, hypothesis=claim_text, device=args.device)
        entail_p = nli.get("entailment", 0.0)
        contra_p = nli.get("contradiction", 0.0)

        query_result = run_problog(entail_p, contra_p, image_p)
        pred_label = predict_label(query_result)

        y_true.append(gold_label)
        y_pred.append(pred_label)
        records.append({
            "entry_id": entry_id, "gold": gold_label, "pred": pred_label,
            "entail_p": entail_p, "contra_p": contra_p, "image_p": image_p,
            "premise_used": premise,
        })

        log.info(f"Row {entry_id}: gold={gold_label:<10} pred={pred_label:<10} "
                 f"(entail={entail_p:.2f} contra={contra_p:.2f} img={image_p:.2f})")

    print("\n--- Per-class mean signal strength ---")
    print(pd.DataFrame(records).groupby("gold")[["entail_p", "contra_p", "image_p"]].mean())

    print("\n--- Zero-shot classification report ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    labels = sorted(set(y_true) | set(y_pred))
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels))

    if args.save_predictions:
        pd.DataFrame(records).to_csv(args.save_predictions, index=False)
        log.info(f"Predictions written to {args.save_predictions}")


if __name__ == "__main__":
    main()