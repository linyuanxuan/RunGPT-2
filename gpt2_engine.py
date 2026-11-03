"""GPT-2 (124M, 2019 原版) 本地推理封装。

模型文件放在本地目录，不联网即可运行。
"""
import os
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def _default_model_dir() -> str:
    r"""优先级: 环境变量 > D:\GPT2Local\models\gpt2 > 代码同级 models/gpt2"""
    env = os.environ.get("GPT2_MODEL_DIR")
    if env:
        return env
    for cand in (
        Path("D:/GPT2Local/models/gpt2"),
        Path(__file__).resolve().parent / "models" / "gpt2",
    ):
        if (cand / "config.json").exists():
            return str(cand)
    return "D:/GPT2Local/models/gpt2"


DEFAULT_MODEL_DIR = _default_model_dir()

_cache = {}


def load(model_dir: str | None = None):
    """加载 tokenizer + model，进程内只加载一次。"""
    model_dir = model_dir or DEFAULT_MODEL_DIR
    if model_dir in _cache:
        return _cache[model_dir]

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"本地模型目录不存在: {model_dir}\n"
            f"请先运行 download_model.py 下载权重。"
        )

    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()

    _cache[model_dir] = (tokenizer, model)
    return tokenizer, model


@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    seed: int | None = None,
    num_return_sequences: int = 1,
    model_dir: str | None = None,
) -> dict:
    tokenizer, model = load(model_dir)

    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_k=top_k if top_k and top_k > 0 else 0,
        top_p=top_p if 0 < top_p < 1 else 1.0,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - t0

    prompt_len = inputs["input_ids"].shape[1]
    texts = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
    n_tokens = int(output.shape[1] - prompt_len)

    return {
        "texts": texts,
        "tokens": n_tokens,
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(n_tokens / elapsed, 1) if elapsed > 0 else 0.0,
        "params": sum(p.numel() for p in model.parameters()),
        "device": "cpu",
    }
