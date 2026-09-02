#!/usr/bin/env python3
"""
/workdir/src/data_download/download_factify_images.py

Downloads claim_image and document_image pairs referenced in the Factify
CSV, writes them to an output directory, and records a JSON mapping of
dataset_entry_id -> {"claim_image": filename, "document_image": filename}.

Twitter-sourced image URLs from 2021-2022 have real link rot, so this
script scans forward through the CSV until it accumulates --limit
successfully-downloaded ROW PAIRS (both images present), rather than
just grabbing the first N rows.
"""

import argparse
import json
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry


import csv
import sys

# Some Factify OCR fields (full article text) exceed the csv module's
# default 128KB field-size cap. Raise it, backing off if the platform's
# C long can't hold sys.maxsize (harmless on Linux, but defensive).
_max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int / 10)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("factify_download")

DEFAULT_HEADERS = {
    # some CDNs reject the default python-requests UA
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def build_session(timeout: float) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(DEFAULT_HEADERS)
    session.request_timeout = timeout
    return session


def guess_extension(url: str, content_type: str | None) -> str:
    # Try the URL path first (works for most twimg.com links)
    path_ext = Path(urlparse(url).path).suffix.lower()
    if path_ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return path_ext
    # Fall back to the response's Content-Type header
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ".jpg"


def download_one(session: requests.Session, url: str, dest_stub: Path, timeout: float) -> Path | None:
    """Download a single image. Returns the final file path, or None on failure."""
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        resp = session.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        ext = guess_extension(url, resp.headers.get("Content-Type"))
        dest_path = dest_stub.with_suffix(ext)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest_path
    except Exception as e:
        log.debug(f"Failed to download {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download Factify claim/document image pairs.")
    parser.add_argument("--csv", type=Path, default=Path("/workdir/data/factify_train_500.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workdir/data/images"))
    parser.add_argument("--mapping-file", type=Path, default=Path("/workdir/data/image_mapping.json"))
    parser.add_argument("--limit", type=int, default=500,
                         help="Number of successfully-downloaded row pairs to collect (default: 100).")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                         help="Skip re-downloading if both files already exist for a row (default: on).")
    parser.add_argument("--sep", type=str, default="\t",
                     help="Field separator for the CSV (default: tab, since Factify's export is TSV-like).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(
            args.csv,
            sep=args.sep,
            quotechar='"',
            engine="python",      # more forgiving than the C engine for messy quoting
            on_bad_lines="warn",  # log and skip unparseable rows instead of crashing
        )
    except Exception as e:
        raise SystemExit(
            f"Failed to parse {args.csv} with sep={args.sep!r}. "
            f"Run `head -1 {args.csv} | cat -A | head -c 300` to check the real delimiter, "
            f"then pass --sep explicitly. Original error: {e}"
        )
    log.info(f"Loaded {len(df)} rows from {args.csv}")
    

    required_cols = {"claim_image", "document_image"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing expected columns: {missing}")

    session = build_session(args.timeout)
    mapping: dict[str, dict[str, str]] = {}
    n_success = 0
    n_attempted = 0
    n_failed_claim = 0
    n_failed_document = 0

    for idx, row in df.iterrows():
        if n_success >= args.limit:
            break
        n_attempted += 1
        entry_id = str(idx)

        claim_stub = args.output_dir / f"{entry_id}_claim"
        doc_stub = args.output_dir / f"{entry_id}_document"

        existing_claim = next(args.output_dir.glob(f"{entry_id}_claim.*"), None)
        existing_doc = next(args.output_dir.glob(f"{entry_id}_document.*"), None)

        if args.skip_existing and existing_claim and existing_doc:
            claim_path, doc_path = existing_claim, existing_doc
        else:
            claim_path = download_one(session, row["claim_image"], claim_stub, args.timeout)
            doc_path = download_one(session, row["document_image"], doc_stub, args.timeout)

        if claim_path is None:
            n_failed_claim += 1
        if doc_path is None:
            n_failed_document += 1

        if claim_path is not None and doc_path is not None:
            mapping[entry_id] = {
                "claim_image": claim_path.name,
                "document_image": doc_path.name,
            }
            n_success += 1
            if n_success % 10 == 0:
                log.info(f"Progress: {n_success}/{args.limit} successful row pairs "
                         f"({n_attempted} rows scanned so far)")

    with open(args.mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)

    log.info("Done.")
    log.info(f"Rows scanned:              {n_attempted} / {len(df)}")
    log.info(f"Successful pairs:          {n_success}")
    log.info(f"Failed claim_image only:   {n_failed_claim}")
    log.info(f"Failed document_image only:{n_failed_document}")
    log.info(f"Mapping written to:        {args.mapping_file}")
    if n_success < args.limit:
        log.warning(
            f"Only found {n_success}/{args.limit} successful pairs after scanning the "
            f"entire CSV — link rot exhausted the available rows."
        )


if __name__ == "__main__":
    main()