"""One-command publisher for the Hugging Face triple (dataset + model card).

Removes the friction from the roadmap's "publish the dataset / model" items:
run this once with an HF token and the corpus, splits, and cards are uploaded
to the right repos.

Requires: `pip install huggingface_hub` and either `huggingface-cli login` or
the HF_TOKEN environment variable.

Usage:
    # Publish the dataset (corpus + splits + card):
    python scripts/upload_to_hub.py dataset \
        --repo systemslibrarian/classical-cipher-corpus

    # Publish a trained model folder (weights + tokenizer + card):
    python scripts/upload_to_hub.py model \
        --repo systemslibrarian/cipher-detective-classifier \
        --model-dir cipher_model
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")
    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token)


def upload_dataset(repo_id: str) -> None:
    api = _api()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    files = {
        "data/cipher_examples.jsonl": "cipher_examples.jsonl",
        "data/splits/train.jsonl": "splits/train.jsonl",
        "data/splits/val.jsonl": "splits/val.jsonl",
        "data/splits/test.jsonl": "splits/test.jsonl",
        "hf_cards/dataset_README.md": "README.md",
    }
    for local, remote in files.items():
        path = REPO / local
        if not path.exists():
            print(f"  skip (missing): {local}")
            continue
        print(f"  upload {local} -> {remote}")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=repo_id, repo_type="dataset")
    print(f"Dataset published: https://huggingface.co/datasets/{repo_id}")


def upload_model(repo_id: str, model_dir: str) -> None:
    api = _api()
    mdir = REPO / model_dir
    if not mdir.is_dir():
        sys.exit(f"Model dir not found: {mdir}")
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"  upload folder {mdir} -> {repo_id}")
    api.upload_folder(folder_path=str(mdir), repo_id=repo_id, repo_type="model")
    card = REPO / "hf_cards" / "model_README.md"
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model")
    print(f"Model published: https://huggingface.co/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="what", required=True)

    d = sub.add_parser("dataset", help="Publish the corpus + splits + card.")
    d.add_argument("--repo", required=True)

    m = sub.add_parser("model", help="Publish a trained model folder + card.")
    m.add_argument("--repo", required=True)
    m.add_argument("--model-dir", default="cipher_model")

    args = ap.parse_args()
    if args.what == "dataset":
        upload_dataset(args.repo)
    else:
        upload_model(args.repo, args.model_dir)


if __name__ == "__main__":
    main()
