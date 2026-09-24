#!/usr/bin/env python3
"""
Push the local Chapter DatasetDict and dataset card to the Hugging Face Hub.

Usage:
    python src/push_chapter_dataset.py --private
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
LABEL2ID = {"O": 0, "B-CHAPTER": 1, "I-CHAPTER": 2}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--local-dir", type=Path, default=ROOT / "chapter_dataset")
    p.add_argument("--card", type=Path, default=ROOT / "data/dataset_card.md")
    p.add_argument("--repo-id", default="Yontenn/formatting-chapter-v1")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)
    if not (args.local_dir / "dataset_dict.json").is_file():
        raise SystemExit(f"No DatasetDict at {args.local_dir}")
    dset = load_from_disk(str(args.local_dir))
    print(f"Pushing {args.repo_id} private={args.private}: "
          + ", ".join(f"{k}={len(v)}" for k, v in dset.items()))
    dset.push_to_hub(args.repo_id, private=args.private)
    api = HfApi()
    api.upload_file(path_or_fileobj=str(args.card), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=json.dumps(LABEL2ID, indent=2).encode(),
                    path_in_repo="label_map.json", repo_id=args.repo_id, repo_type="dataset")
    print(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
