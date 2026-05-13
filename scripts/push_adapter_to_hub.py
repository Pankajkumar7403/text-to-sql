"""
Pushes the LoRA adapter to HuggingFace Hub.

Usage:
  python scripts/push_adapter_to_hub.py
  python scripts/push_adapter_to_hub.py --adapter data/models/qwen25_sql_v2_adapter --repo pankajkumar7403/qwen25-sql-v2
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

ROOT = Path(__file__).parent.parent

DEFAULT_ADAPTER = ROOT / "data" / "models" / "qwen25_sql_v2_adapter"
DEFAULT_REPO    = "pankaj74/qwen25-sql-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER,
                        help="Local adapter directory (default: data/models/qwen25_sql_v2_adapter)")
    parser.add_argument("--repo",    type=str,  default=DEFAULT_REPO,
                        help="HF Hub repo id (default: pankajkumar7403/qwen25-sql-v2)")
    parser.add_argument("--token",   type=str,  default=None,
                        help="HF token (or set HF_TOKEN env var / run huggingface-cli login)")
    args = parser.parse_args()

    if not args.adapter.exists():
        raise SystemExit(
            f"Adapter not found at {args.adapter}\n"
            "Download it from Kaggle first:\n"
            "  import shutil; shutil.make_archive('/kaggle/working/adapter_export', 'zip', '/kaggle/working/qwen25_sql_v2_adapter')\n"
            "Then unzip into data/models/qwen25_sql_v2_adapter/"
        )

    if args.token:
        login(token=args.token)

    api = HfApi()

    print(f"Creating repo: {args.repo}")
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=False)

    print(f"Uploading {args.adapter} → {args.repo}")
    api.upload_folder(
        folder_path=str(args.adapter),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add QLoRA adapter (Qwen2.5-7B-Instruct, r=16)",
    )
    print(f"Done → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
