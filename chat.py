"""交互式对话式续写（终端 REPL）。

用法: python chat.py
命令: /temp 0.8  /len 120  /seed 42  /reset  /quit
"""
import sys

from gpt2_engine import generate, load, DEFAULT_MODEL_DIR


def main():
    print("[gpt2] 加载模型 …", flush=True)
    load()
    print(f"[gpt2] 就绪，模型目录: {DEFAULT_MODEL_DIR}")
    print("输入文本回车即续写；/len N 改长度，/temp X 改温度，/seed N 固定随机，/quit 退出。\n")

    cfg = {"max_new_tokens": 80, "temperature": 0.9, "seed": None}
    context = ""

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/len"):
            cfg["max_new_tokens"] = int(line.split()[1]); print(f"  长度 = {cfg['max_new_tokens']}"); continue
        if line.startswith("/temp"):
            cfg["temperature"] = float(line.split()[1]); print(f"  温度 = {cfg['temperature']}"); continue
        if line.startswith("/seed"):
            v = line.split()[1]
            cfg["seed"] = None if v.lower() in ("none", "null") else int(v)
            print(f"  seed = {cfg['seed']}"); continue
        if line == "/reset":
            context = ""; print("  上下文已清空"); continue

        # 把前一轮输出接上，形成连续写作；超出 800 字符就截断防爆上下文
        prompt = (context + "\n" + line) if context else line
        prompt = prompt[-800:]

        res = generate(prompt, **cfg)
        text = res["texts"][0]
        print(f"gpt2> {text}")
        print(f"      [ {res['tokens']} tok · {res['seconds']}s · {res['tokens_per_second']} tok/s ]\n", flush=True)
        context = prompt + text


if __name__ == "__main__":
    sys.exit(main())
