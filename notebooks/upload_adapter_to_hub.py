# Upload LoRA adapter to HuggingFace Hub
# Copy-paste this entire file into a Kaggle notebook cell and run it.
#
# BEFORE RUNNING:
#   1. Go to huggingface.co -> Settings -> Access Tokens
#   2. Create a token with WRITE permission
#   3. Paste the token below

HF_TOKEN = "hf_PASTE_YOUR_WRITE_TOKEN_HERE"   # <-- replace this
ADAPTER_PATH = "/kaggle/working/qwen25_sql_v2_adapter"

from huggingface_hub import HfApi, login
from pathlib import Path
import os

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()
username = api.whoami()["name"]
REPO_ID = f"{username}/qwen25-sql-v2"
print(f"Logged in as : {username}")
print(f"Target repo  : {REPO_ID}")

adapter_dir = Path(ADAPTER_PATH)
if not adapter_dir.exists():
    raise SystemExit(f"Adapter not found at {ADAPTER_PATH}\nCheck: !ls /kaggle/working/")

files = list(adapter_dir.iterdir())
print(f"Adapter files: {[f.name for f in files]}")

print(f"\nCreating repo {REPO_ID} ...")
url = api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,
    private=False,
)
print(f"Repo ready   : {url}")

print(f"\nUploading adapter files...")
api.upload_folder(
    folder_path=ADAPTER_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add QLoRA adapter — Qwen2.5-7B-Instruct r=16",
)

print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")
