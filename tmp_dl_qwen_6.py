import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

print("Downloading Qwen3-ASR-0.6B...")
path = snapshot_download(repo_id="Qwen/Qwen3-ASR-0.6B")
print(f"Downloaded to {path}")
