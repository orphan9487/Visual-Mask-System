#!/usr/bin/env python3
"""Derive a standalone *thesis* project from the unified Visual Mask System repo.

The thesis variant is the self-hosted WebSocket chatroom only: no LINE webhook,
no talking-face video, no TTS.  Instead of maintaining a second repo by hand
(which would drift), this script generates that clean variant on demand from a
single source of truth - the unified `main`.  Run it whenever you need a fresh
export to hand in or push as a separate project.

    python scripts/export_thesis.py [output_dir] [--ref main] [--git-init]

Defaults: output_dir = ../vms-thesis , ref = main.

What it does:
  1. Exports the tracked files of <ref> into a clean output directory.
  2. Deletes every LINE / talking-face / TTS specific file.
  3. Best-effort strips the (already inert) LINE hook from the WebSocket app so
     the code reads clean - correctness does not depend on this step.
  4. Makes the thesis `.env.example` the only env template and rewrites README.
  5. Optionally `git init`s the export so it is ready to push as its own repo.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# LINE / talking-face / TTS specific paths. Everything here is competition-only
# and is removed from the thesis export. Directories are removed recursively.
REMOVE: list[str] = [
    # --- LINE ---
    "src/interfaces/line_webhook.py",
    "tests/test_line_webhook.py",
    # --- Talking face ---
    "docs/talking-face-adapter.md",
    "src/config/talking_face.py",
    "src/interfaces/talking_face_sidecar.py",
    "src/services/talking_face_pipeline.py",
    "tests/test_talking_face_adapter.py",
    "tests/test_talking_face_pipeline.py",
    "tests/test_talking_face_sidecar.py",
    # --- TTS ---
    "src/config/tts.py",
    "tests/test_tts.py",
    # --- whole integrations package is LINE/TF/TTS only ---
    "src/integrations",
    # --- competition env template ---
    ".env.competition.example",
]

# The LINE hook inside websocket_app.create_app(). It is guarded by
# features.LINE_ENABLED (False in thesis) so it never runs, but we strip it for
# a clean read. If the source changes and this no longer matches, we warn and
# leave it - the export still works because the guard stays False.
WS_APP = "src/interfaces/websocket_app.py"
LINE_BLOCK = (
    "\n"
    "    # The LINE webhook (and its line-bot-sdk dependency) only loads for the\n"
    "    # competition profile; the thesis profile runs WebSocket-chat-only.\n"
    "    if features.LINE_ENABLED:\n"
    "        from src.interfaces.line_webhook import router as line_router\n"
    "\n"
    "        app.include_router(line_router)\n"
)
FEATURES_IMPORT = "from src.config import features\n"

THESIS_README = """# Visual Mask System - 校內專題版

自製 WebSocket 聊天室版本。瀏覽器透過 FastAPI WebSocket 傳送訊息，系統即時完成
情緒推論、建立 VisualInstruction，再由 LoRA 生成圖片或影片。**不含 LINE Bot、
talking face、TTS。**

> 本目錄由統一 repo 的 `scripts/export_thesis.py` 自動產生，請勿手動維護；
> 需要更新時回到統一 repo 重新匯出。

## 啟動

```powershell
python -m venv .venv && .venv\\Scripts\\activate
pip install -r requirements-cuda.txt
copy .env.example .env        # 填入 VMS_DB_* 資料庫連線
python scripts/database/init_db.py
python chat_server.py
```

開啟 http://127.0.0.1:8000 。健康檢查在 /health 。

## 測試

```powershell
python -m pytest tests/ -q
```

## 架構

```text
index.html ──WS──▶ src/interfaces/websocket_app.py
                          │
                          ▼
                 src/services/pipeline_service.py
                   ├─ perception/  情緒感知
                   ├─ reasoning/   ERC 推理
                   └─ generation/  LoRA 圖片／影片
```
"""


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def export_tree(ref: str, out: Path) -> None:
    """Extract the tracked files of <ref> into <out> (clean, no models/data)."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "tree.tar"
        run(["git", "-C", str(REPO), "archive", "--format=tar", "-o", str(archive), ref])
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(out, filter="data")  # py>=3.12: safe extraction
            except TypeError:
                tar.extractall(out)  # older Python without the filter arg


def remove_paths(out: Path) -> list[str]:
    removed = []
    for rel in REMOVE:
        target = out / rel
        if target.is_dir():
            _rmtree(target)
            removed.append(rel + "/")
        elif target.exists():
            target.unlink()
            removed.append(rel)
    return removed


def _rmtree(p: Path) -> None:
    for child in p.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink()
    p.rmdir()


def strip_line_hook(out: Path) -> bool:
    """Remove the inert LINE block + now-unused features import. Best effort."""
    app = out / WS_APP
    if not app.exists():
        return False
    text = app.read_text(encoding="utf-8")
    if LINE_BLOCK not in text:
        return False
    text = text.replace(LINE_BLOCK, "\n")
    text = text.replace(FEATURES_IMPORT, "")  # import now unused
    app.write_text(text, encoding="utf-8")
    return True


def fix_env_and_readme(out: Path) -> None:
    thesis_env = out / ".env.thesis.example"
    env_example = out / ".env.example"
    if thesis_env.exists():
        env_example.write_text(thesis_env.read_text(encoding="utf-8"), encoding="utf-8")
        thesis_env.unlink()
    (out / "README.md").write_text(THESIS_README, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", nargs="?", default=str(REPO.parent / "vms-thesis"))
    ap.add_argument("--ref", default="main", help="git ref to export (default: main)")
    ap.add_argument("--git-init", action="store_true",
                    help="git init + initial commit in the export")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    if out == REPO:
        print("refusing to export into the repo itself", file=sys.stderr)
        return 2
    if out.exists():
        _rmtree(out)
    out.mkdir(parents=True)

    print(f"exporting {args.ref} -> {out}")
    export_tree(args.ref, out)
    removed = remove_paths(out)
    stripped = strip_line_hook(out)
    fix_env_and_readme(out)

    print(f"removed {len(removed)} LINE/talking-face/TTS paths")
    print("cleaned LINE hook in websocket_app.py" if stripped
          else "note: LINE hook pattern not found (guard keeps it inert anyway)")

    if args.git_init:
        run(["git", "-C", str(out), "init", "-q"])
        run(["git", "-C", str(out), "add", "-A"])
        run(["git", "-C", str(out), "commit", "-q", "-m",
             "Thesis export: self-hosted chatroom only (no LINE/talking-face/TTS)"])
        print("initialised git repo in export")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
