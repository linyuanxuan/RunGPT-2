"""本地 GPT-2 网页界面（纯标准库，无需额外依赖）。

运行: python app.py  然后浏览器打开 http://127.0.0.1:7860
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gpt2_engine import generate, load

HOST = os.environ.get("GPT2_HOST", "127.0.0.1")
PORT = int(os.environ.get("GPT2_PORT", "7860"))

_lock = threading.Lock()  # CPU 推理串行，避免多请求抢核

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-2 · 本地推理</title>
<style>
  :root{
    --bg:#f5f6f8; --panel:#ffffff; --ink:#1a1d21; --sub:#6b7280;
    --line:#e3e6ea; --accent:#2f6df6; --accent-soft:#eaf0ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.65 "Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;}
  .wrap{max-width:920px;margin:0 auto;padding:28px 20px 60px}
  header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:4px}
  h1{font-size:21px;margin:0;letter-spacing:-.2px}
  .badge{font-size:12px;color:var(--accent);background:var(--accent-soft);
    border:1px solid #d3e0ff;border-radius:999px;padding:2px 10px}
  .desc{color:var(--sub);font-size:13px;margin:6px 0 22px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:16px;margin-bottom:14px}
  label{display:block;font-size:12px;color:var(--sub);margin-bottom:6px;
    text-transform:uppercase;letter-spacing:.06em}
  textarea{width:100%;min-height:92px;resize:vertical;border:1px solid var(--line);
    border-radius:9px;padding:11px 12px;font:inherit;color:var(--ink);background:#fcfcfd;outline:none}
  textarea:focus{border-color:var(--accent);background:#fff}
  .row{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px}
  .field{flex:1 1 130px;min-width:120px}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .val{float:right;color:var(--ink);font-variant-numeric:tabular-nums}
  input[type=number]{width:100%;border:1px solid var(--line);border-radius:8px;padding:6px 9px;
    font:inherit;background:#fcfcfd;color:var(--ink)}
  .actions{display:flex;align-items:center;gap:12px;margin-top:16px}
  button{border:0;border-radius:9px;padding:10px 22px;font:inherit;font-weight:600;
    background:var(--accent);color:#fff;cursor:pointer;transition:.15s}
  button:hover{filter:brightness(1.06)}
  button:disabled{opacity:.5;cursor:progress}
  .meta{color:var(--sub);font-size:12.5px;font-variant-numeric:tabular-nums}
  .out{white-space:pre-wrap;word-break:break-word;font-size:15px;min-height:28px}
  .out .p{color:var(--sub)}
  .cursor{display:inline-block;width:7px;height:15px;background:var(--accent);
    vertical-align:-2px;animation:blink 1s steps(1) infinite;margin-left:1px}
  @keyframes blink{50%{opacity:0}}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:640px){.grid{grid-template-columns:1fr}}
  footer{color:var(--sub);font-size:12px;text-align:center;margin-top:26px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>GPT-2</h1>
    <span class="badge">124M · 本地 CPU 推理</span>
  </header>
  <p class="desc">2019 年原版 GPT-2 small，权重在本机，不联网、不上传。输入开头文本，模型接着往下写。</p>

  <div class="panel">
    <label for="prompt">Prompt</label>
    <textarea id="prompt">The best thing about being a language model in 2019 was</textarea>
    <div class="row">
      <div class="field">
        <label>生成长度 <span class="val" id="nV">80</span></label>
        <input type="range" id="n" min="5" max="300" step="5" value="80">
      </div>
      <div class="field">
        <label>Temperature <span class="val" id="tV">0.90</span></label>
        <input type="range" id="t" min="0" max="1.6" step="0.05" value="0.9">
      </div>
      <div class="field">
        <label>Top-k <span class="val" id="kV">40</span></label>
        <input type="range" id="k" min="0" max="100" step="5" value="40">
      </div>
      <div class="field">
        <label>Top-p <span class="val" id="pV">0.95</span></label>
        <input type="range" id="p" min="0.1" max="1" step="0.05" value="0.95">
      </div>
      <div class="field">
        <label>Seed（空=随机）</label>
        <input type="number" id="s" placeholder="random">
      </div>
    </div>
    <div class="actions">
      <button id="go">生成</button>
      <span class="meta" id="speed"></span>
    </div>
  </div>

  <div class="panel">
    <label>输出</label>
    <div class="out" id="out"><span class="p">结果会显示在这里。</span></div>
  </div>

  <footer>模型目录：D:\GPT2Local\models\gpt2 · 参数量 124M · 上下文 1024 tokens</footer>
</div>
<script>
const $ = id => document.getElementById(id);
const pairs = [["n","nV"],["t","tV"],["k","kV"],["p","pV"]];
pairs.forEach(([a,b])=>{
  const el=$(a), out=$(b);
  const sync=()=>{ out.textContent = a==="t"||a==="p" ? Number(el.value).toFixed(2) : el.value; };
  el.addEventListener("input",sync); sync();
});

async function run(){
  const btn=$("go"), out=$("out"), speed=$("speed");
  const prompt=$("prompt").value;
  if(!prompt.trim()){ out.textContent="请输入 prompt"; return; }
  btn.disabled=true; btn.textContent="生成中…";
  out.innerHTML='<span class="cursor"></span>';
  speed.textContent="";
  const seed=$("s").value.trim();
  const body={
    prompt, max_new_tokens:+$("n").value, temperature:+$("t").value,
    top_k:+$("k").value, top_p:+$("p").value,
    seed: seed==="" ? null : Number(seed)
  };
  try{
    const r = await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const d = await r.json();
    if(d.error){ out.textContent="错误: "+d.error; }
    else{
      out.innerHTML="";
      const p=document.createElement("span"); p.className="p"; p.textContent=prompt; out.appendChild(p);
      const txt=document.createElement("span"); txt.textContent=d.texts[0]; out.appendChild(txt);
      speed.textContent = `${d.tokens} tokens · ${d.seconds}s · ${d.tokens_per_second} tok/s`;
    }
  }catch(e){ out.textContent="请求失败: "+e.message; }
  btn.disabled=false; btn.textContent="生成";
}
$("go").addEventListener("click",run);
$("prompt").addEventListener("keydown",e=>{ if((e.ctrlKey||e.metaKey)&&e.key==="Enter") run(); });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "GPT2Local/1.0"

    def log_message(self, fmt, *args):
        pass  # 静音访问日志

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            with _lock:
                res = generate(
                    req.get("prompt", ""),
                    max_new_tokens=int(req.get("max_new_tokens", 80)),
                    temperature=float(req.get("temperature", 0.9)),
                    top_k=int(req.get("top_k", 40)),
                    top_p=float(req.get("top_p", 0.95)),
                    repetition_penalty=float(req.get("repetition_penalty", 1.1)),
                    seed=req.get("seed"),
                )
            self._send(200, json.dumps(res, ensure_ascii=False))
        except Exception as e:  # 界面要能看见错误
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))


def main():
    print("[gpt2] 加载模型…")
    load()
    print(f"[gpt2] 就绪 -> http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
