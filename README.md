# GPT-2 Local · 本地跑通 2019 年的 GPT-2

> 在自己的电脑上离线运行 OpenAI GPT-2（124M）——纯 CPU、无需显卡、不联网、不调 API。

[![GitHub](https://img.shields.io/badge/GitHub-linyuanxuan%2FRunGPT-2-181717?logo=github)](https://github.com/linyuanxuan/RunGPT-2)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-GPT--2%20124M-4fd1c5)
![License](https://img.shields.io/badge/license-MIT-blue)

**介绍页（GitHub Pages）**：https://linyuanxuan.github.io/RunGPT-2/

---

## 这是什么

2019 年 2 月 OpenAI 发布的 GPT-2 small，权重完整落在本地磁盘，跑在一个 500 行不到的小工程里：

- **纯 CPU 推理**，不需要 CUDA、不需要显卡
- **完全离线**，断网可用，权重下载一次即可
- **零框架**，Web 界面只用 Python 标准库 `http.server`，没有 Flask / FastAPI / Gradio
- **三种入口**：网页版、命令行、交互式续写

它不是 ChatGPT。它只会「接着你的话往下写」，没有指令跟随、没有对齐训练——这恰好是它作为时代标本的价值。

## 特性

| | |
|---|---|
| 一键网页 UI | 滑块调 temperature / top-k / top-p / seed，实时显示 tok/s |
| 命令行单次生成 | `cli.py "prompt" -n 60 -t 0.9 -s 42`，可一次出多个候选 |
| 交互续写 | `chat.py` 维护上下文，支持 `/len` `/temp` `/seed` `/reset` |
| 可复现输出 | 固定 seed 即固定结果，适合做实验对比 |
| 本地权重脚本 | `download_model.py`，支持换 `gpt2-medium` 等更大规格 |
| 极简依赖 | 只有 `torch` + `transformers`，界面无第三方库 |

## 快速开始

### 1. 准备环境

```bash
git clone https://github.com/linyuanxuan/RunGPT-2.git
cd RunGPT-2

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install torch transformers
```

> 国内装 torch 建议用镜像：`pip install torch transformers -i https://mirrors.aliyun.com/pypi/simple`。
> 注意清华源目前缺 cp313 的 torch wheel，会报 `No matching distribution found`。

### 2. 下载权重（约 523 MB，只需一次）

```bash
python download_model.py
```

默认下载 `gpt2`（124M）到 `../models/gpt2`。想用 355M 版本：

```bash
python download_model.py gpt2-medium
```

国内网络可先设镜像：`set HF_ENDPOINT=https://hf-mirror.com`（Windows）/ `export HF_ENDPOINT=https://hf-mirror.com`（*nix）。

### 3. 开跑

```bash
python app.py          # 网页版 → http://127.0.0.1:7860
python cli.py "The future of AI is" -n 60
python chat.py         # 交互续写
```

Windows 用户直接双击 `启动网页版.bat` 亦可（首次启动要等 40–60 秒加载权重，窗口没反应是正常的）。

## 用法详解

### 网页版

`python app.py` 后打开 http://127.0.0.1:7860 。可在界面上调：

- **生成长度** 5–300 tokens
- **Temperature** 0（贪心，确定性）～1.6（发散）
- **Top-k / Top-p** 采样截断
- **Seed** 留空为随机，填数字可复现

端口占用时用环境变量改：`GPT2_PORT=8000 python app.py`。

### 命令行

```bash
python cli.py "Once upon a time" -n 100 -t 0.85 -k 40 -p 0.95 -s 7
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `-n, --max-new-tokens` | 生成 token 数 | 80 |
| `-t, --temperature` | 温度，0 = 贪心解码 | 0.9 |
| `-k, --top-k` | 每步候选数，0 = 关闭 | 40 |
| `-p, --top-p` | 核采样阈值 | 0.95 |
| `-r, --repetition-penalty` | 重复惩罚 | 1.1 |
| `-s, --seed` | 随机种子 | 随机 |
| `--num-return` | 一次生成几个候选 | 1 |
| `--model-dir` | 权重目录 | 自动探测 |

### 交互续写

```bash
python chat.py
```

```
you> The old lighthouse keeper
gpt2> had been there for over forty years, and he had never once seen...
      [ 80 tok · 38.2s · 2.1 tok/s ]
```

内置命令：`/len 120` `/temp 0.8` `/seed 42` `/reset` `/quit`。

## 模型规格

| 项 | 值 |
|---|---|
| 发布 | 2019-02，OpenAI |
| 架构 | GPT-2 small，decoder-only Transformer |
| 参数量 | 124M（12 层 · 12 头 · 768 维） |
| 上下文长度 | 1024 tokens |
| 词表 | 50257（BPE） |
| 权重体积 | 523 MB（safetensors，fp32） |
| 推理设备 | CPU |
| 精度 | fp32，未量化 |

### 实测性能

2 核 CPU、约 790 MB 内存占用：

| 精度 | 速度 |
|---|---|
| fp32（默认） | **约 2 tok/s** |
| int8 动态量化（实验） | 约 2.1 tok/s（首次测试 4 倍提速，稳定后优势收窄），质量略降 |

生成一段 40 tokens 的文本约 20 秒。速度由核数决定，4 核以上会明显更快。

## 项目结构

```
RunGPT-2/
├── gpt2_engine.py      # 模型加载 + 生成封装（唯一核心）
├── app.py              # 网页服务，标准库 http.server + 内嵌 HTML
├── cli.py              # 命令行单次生成
├── chat.py             # 终端交互续写
├── download_model.py   # 权重下载（huggingface_hub snapshot）
├── index.html          # 项目介绍页（GitHub Pages / Cloudflare Pages）
├── _headers            # Cloudflare Pages 响应头配置
├── 启动网页版.bat       # Windows 双击启动
└── README.md
```

代码与权重分离：权重默认落在 `../models/gpt2`，通过 `GPT2_MODEL_DIR` 环境变量可指向任何位置。

## 部署介绍页

`index.html` 是纯静态单文件，无构建步骤，任何静态托管都能直接放。

### Cloudflare Pages（推荐，支持自定义域名）

1. Cloudflare Dashboard → **Workers & Pages** → **Create** → **Pages** → **Connect to Git**
2. 选择本仓库，分支 `main`
3. 构建设置：

   | 项 | 值 |
   |---|---|
   | Framework preset | **None** |
   | Build command | 留空 |
   | Build output directory | `/` |

4. Save and Deploy。之后每次 push 到 `main` 自动重新部署
5. 绑定域名：项目 → **Custom domains** → 输入子域（如 `gpt2.example.com`）→ 域名也在 Cloudflare 时 DNS 记录自动创建，几分钟内签发证书

仓库根目录的 `_headers` 会被 CF Pages 识别，用于设置 `index.html` 的短缓存与安全响应头。

### GitHub Pages

Settings → Pages → Source 选 `main` + `/ (root)`，站点即 `https://linyuanxuan.github.io/RunGPT-2/`。

> 注意：Pages 只负责托管这份介绍页，不会运行 `app.py`。真正的 GPT-2 推理始终在你本机，这里没有后端。

## 已知限制

- **中文基本不可用**。GPT-2 是纯英文模型，中文输入输出乱码。要中文请换中文预训练模型。
- **不会回答问题**。它是续写模型，没有指令微调。问它「法国首都是哪」它只会续写出看起来像文本的字符串。
- **上下文只有 1024 tokens**。长对话会截断。
- **CPU 上很慢**。2 tok/s 意味着长文本要等几分钟。
- 知识截止到 2019 年之前的网页文本。

这些不是 bug，是 2019 年的真实水位。

## 常见问题

<details>
<summary>启动后浏览器打不开 / 端口被占用</summary>

换端口：`GPT2_PORT=8000 python app.py`（Windows PowerShell：`$env:GPT2_PORT=8000; python app.py`）。
</details>

<details>
<summary>报错「本地模型目录不存在」</summary>

先跑 `python download_model.py`，或把 `GPT2_MODEL_DIR` 指向已有权重目录。
</details>

<details>
<summary>Windows 双击 .bat 出现一堆乱码</summary>

旧版 bat 用 UTF-8 保存，cmd 按 GBK 解析就会炸。仓库内的 `启动网页版.bat` 已改为纯 ASCII + CRLF，重新拉取即可。
</details>

<details>
<summary>能换成更大的 GPT-2 吗</summary>

可以。`python download_model.py gpt2-medium`（355M，约 1.4 GB），或 `gpt2-large`（774M）、`gpt2-xl`（1.5B），然后把 `GPT2_MODEL_DIR` 指到新目录。xl 在 CPU 上基本不可用。
</details>

<details>
<summary>能不能不装 torch</summary>

不能，推理后端就是 PyTorch。想更轻量可考虑 ONNX Runtime 或 llama.cpp 的 GPT-2 转换版，但要额外转换步骤。
</details>

## License

MIT。模型权重版权归 OpenAI，遵循其原许可；本仓库仅含加载与推理代码。

---

<sub>Built for people who want to see how far 2019 got.</sub>
