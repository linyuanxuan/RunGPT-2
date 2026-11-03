"""命令行生成：一次性补全文本。

用法:
  python cli.py "The meaning of life is"
  python cli.py -n 120 --temperature 0.8 -k 3 "Once upon a time"
"""
import argparse

from gpt2_engine import generate, load


def main():
    p = argparse.ArgumentParser(description="本地 GPT-2 文本生成")
    p.add_argument("prompt", help="提示词 / 开头文本")
    p.add_argument("-n", "--max-new-tokens", type=int, default=80)
    p.add_argument("-t", "--temperature", type=float, default=0.9, help="0 = 贪心解码")
    p.add_argument("-k", "--top-k", type=int, default=40)
    p.add_argument("-p", "--top-p", type=float, default=0.95)
    p.add_argument("-r", "--repetition-penalty", type=float, default=1.1)
    p.add_argument("-s", "--seed", type=int, default=None)
    p.add_argument("--num-return", type=int, default=1, help="一次生成几个候选")
    p.add_argument("--model-dir", default=None)
    a = p.parse_args()

    load(a.model_dir)  # 预热 + 校验模型
    res = generate(
        a.prompt,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_k=a.top_k,
        top_p=a.top_p,
        repetition_penalty=a.repetition_penalty,
        seed=a.seed,
        num_return_sequences=a.num_return,
        model_dir=a.model_dir,
    )

    multi = a.num_return > 1
    for i, t in enumerate(res["texts"], 1):
        if multi:
            print(f"\n===== 候选 {i} =====")
            print(t)
        else:
            print(a.prompt + t)

    print(
        f"\n[ {res['tokens']} tokens / {res['seconds']}s "
        f"= {res['tokens_per_second']} tok/s | 参数量 {res['params']/1e6:.0f}M | CPU ]"
    )


if __name__ == "__main__":
    main()
