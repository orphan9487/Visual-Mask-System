# -*- coding: utf-8 -*-
"""
一鍵啟動：Cloudflare tunnel + 伺服器 + 自動同步 LINE webhook。

解決免費 tunnel「每次重啟換網址」的痛點——本腳本會自動抓新網址、更新 .env、
並用 LINE API 自動設定 webhook endpoint，全程無需手動同步。

用法：
    .venv\\Scripts\\python scripts\\launch.py
按 Ctrl+C 結束（會一併關閉伺服器與 tunnel）。

前置：.env 已填好 LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN；
cloudflared 已安裝（winget install Cloudflare.cloudflared）。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PORT = int(os.getenv("PORT", "8000"))
CF_FALLBACK = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_cloudflared() -> str:
    return shutil.which("cloudflared") or CF_FALLBACK


def read_env() -> dict:
    d = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d


def update_env_url(url: str):
    p = ROOT / ".env"
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    out, done = [], False
    for line in lines:
        if line.startswith("PUBLIC_BASE_URL="):
            out.append(f"PUBLIC_BASE_URL={url}"); done = True
        else:
            out.append(line)
    if not done:
        out.append(f"PUBLIC_BASE_URL={url}")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def set_line_webhook(url: str, token: str):
    """用 LINE API 設定 webhook endpoint；失敗不致命（改提示手動）。"""
    endpoint = f"{url}/webhook"
    req = urllib.request.Request(
        "https://api.line.me/v2/bot/channel/webhook/endpoint",
        data=json.dumps({"endpoint": endpoint}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="PUT")
    try:
        urllib.request.urlopen(req, timeout=15)
        return True, f"已自動設定 LINE webhook → {endpoint}"
    except Exception as e:  # noqa: BLE001
        return False, f"自動設定失敗（請手動把 {endpoint} 貼進 LINE Console）：{e}"


def wait_health(timeout=90) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    cf = find_cloudflared()
    if not Path(cf).exists() and shutil.which("cloudflared") is None:
        print(f"[error] 找不到 cloudflared，請先安裝：winget install Cloudflare.cloudflared")
        return

    print("[1/5] 啟動 Cloudflare tunnel …")
    tunnel = subprocess.Popen([cf, "tunnel", "--url", f"http://localhost:{PORT}"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, encoding="utf-8", errors="replace")
    url = None
    t0 = time.time()
    for line in tunnel.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0); break
        if time.time() - t0 > 60:
            break
    if not url:
        print("[error] 60 秒內取不到 tunnel 網址，請重試"); tunnel.terminate(); return
    print(f"       網址：{url}")

    update_env_url(url)
    os.environ["PUBLIC_BASE_URL"] = url
    print("[2/5] 已更新 .env 的 PUBLIC_BASE_URL")

    print("[3/5] 啟動伺服器 …")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server = subprocess.Popen(
        [str(PY), "-m", "uvicorn", "src.line_bot.app:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT), env=env)

    print("[4/5] 等待伺服器就緒（首次載入模型可能 1–2 分鐘）…")
    if not wait_health():
        print("[warn] 伺服器逾時未就緒，仍繼續（稍後可自行確認 /health）")

    token = read_env().get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if token:
        ok, msg = set_line_webhook(url, token)
        print(f"[5/5] {msg}")
    else:
        print(f"[5/5] 未找到 access token，請手動把 {url}/webhook 貼進 LINE Console")

    print("\n" + "=" * 56)
    print(f"✅ 全部就緒　Webhook：{url}/webhook")
    print("   健康檢查： /health　除錯： /debug/instruction")
    print("   按 Ctrl+C 結束（會一併關閉伺服器與 tunnel）")
    print("=" * 56)

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n收到中斷，關閉中 …")
    finally:
        for p in (server, tunnel):
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
