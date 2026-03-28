"""Download a small GGUF model for CPU inference."""
import os
from huggingface_hub import hf_hub_download
from config import MODEL_DIR, LLM_MODEL_PATH

# Qwen2.5-3B - best quality-to-size ratio for CPU RAG Q&A
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"


def download():
    """Download the GGUF model from HuggingFace."""
    if os.path.exists(LLM_MODEL_PATH):
        print(f"Model already exists at {LLM_MODEL_PATH}")
        return

    print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
    print("This may take a few minutes depending on your internet speed.")
    print(f"Model size: ~2.0 GB")

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    # Rename to standard name
    target = LLM_MODEL_PATH
    if downloaded_path != target:
        os.rename(downloaded_path, target)

    print(f"Model saved to {target}")
    print("Done!")


if __name__ == "__main__":
    download()
