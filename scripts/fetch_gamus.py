"""Fetch GAMUS triplets referenced by a split manifest from Hugging Face.

Uses hf_hub_download in a thread pool, NOT snapshot_download(allow_patterns=...):
with ~3,600 unique relpaths, snapshot_download would list all 26,172
repo files and run every file through fnmatch against every pattern
(~94M fnmatch calls, 60-180s of pure CPU before a single byte downloads),
repeating that cost on every resume. hf_hub_download short-circuits on a
local metadata hit instead, which is the resume mechanism.

repo_type="dataset" is mandatory (its omission 404s against the model
namespace). local_dir=, not cache_dir=: local_dir writes real files at
ROOT/images/train/... matching split-file relpaths directly; cache_dir's
blob+symlink layout needs Windows Developer Mode for symlinks and
silently falls back to full copies (doubling disk usage) otherwise.

Usage:
    python scripts/fetch_gamus.py --split data/splits/stage_a1_val.txt --limit 20
    python scripts/fetch_gamus.py --split data/splits/stage_a1_train.txt --split data/splits/stage_a1_val.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running as `python scripts/fetch_gamus.py` from the repo root
# without installing the package -- inserts the repo root (parent of
# this script's directory) onto sys.path so `from src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from src.dataset import parse_split_file

REPO_ID = "earthflow/GAMUS"
REPO_TYPE = "dataset"
MAX_RETRIES = 5
MAX_WORKERS = 8


def fetch_one(relpath: str, root: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return hf_hub_download(
                REPO_ID, relpath, repo_type=REPO_TYPE, local_dir=root
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, retried
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # 1, 2, 4, 8 s backoff, absorbs 429s
    raise RuntimeError(f"Failed to fetch {relpath} after {MAX_RETRIES} attempts") from last_exc


def fetch_gamus(split_paths: list[str], data_root: str, limit: int | None = None, max_workers: int = MAX_WORKERS) -> list[str]:
    os.makedirs(data_root, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    relpaths: set[str] = set()
    for split_path in split_paths:
        rows = parse_split_file(split_path)
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            relpaths.add(row.rgb_relpath)
            relpaths.add(row.height_relpath)
            relpaths.add(row.cls_relpath)

    relpaths_sorted = sorted(relpaths)
    print(f"[fetch_gamus] {len(relpaths_sorted)} unique files to fetch into {data_root}")

    downloaded: list[str] = []
    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, rp, data_root): rp for rp in relpaths_sorted}
        for i, future in enumerate(as_completed(futures), start=1):
            relpath = futures[future]
            try:
                path = future.result()
                downloaded.append(path)
            except Exception as exc:  # noqa: BLE001
                failures.append((relpath, exc))
            if i % 50 == 0 or i == len(relpaths_sorted):
                print(f"[fetch_gamus] {i}/{len(relpaths_sorted)} done ({len(failures)} failed)")

    if failures:
        print(f"[fetch_gamus] {len(failures)} files FAILED:")
        for relpath, exc in failures[:20]:
            print(f"  {relpath}: {exc}")
        raise RuntimeError(f"{len(failures)} of {len(relpaths_sorted)} downloads failed")

    print(f"[fetch_gamus] done: {len(downloaded)} files in {data_root}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Fetch GAMUS triplets from Hugging Face")
    parser.add_argument(
        "--split", action="append", required=True,
        help="Split manifest path (repeatable, e.g. --split a.txt --split b.txt)",
    )
    parser.add_argument("--data-root", default="data/gamus")
    parser.add_argument("--limit", type=int, default=None, help="Truncate each split to first N rows")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    fetch_gamus(args.split, args.data_root, limit=args.limit, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
