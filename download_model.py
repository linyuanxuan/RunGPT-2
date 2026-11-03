"""下载 GPT-2 权重到本地 models/gpt2 （一次性，之后可完全离线运行）。

用法:
  python download_model.py            # 默认 gpt2 (124M)
  python download_model.py gpt2-medium
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

FILES = [
    "config.json", "vocab.json", "merges.txt", "tokenizer.json",
    "tokenizer_config.json", "generation_config.json", "model.safetensors",
]


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    target = Path(__file__).resolve().parent.parent / "models" / repo.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)
    print(f"下载 {repo} -> {target}")
    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        allow_patterns=FILES,
        max_workers=8,
    )
    print("完成。", target)


if __name__ == "__main__":
    main()
