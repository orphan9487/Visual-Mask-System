"""Compatibility launcher for the pure WebSocket chat application."""

from src.interfaces.multipage_app import app


if __name__ == "__main__":
    import uvicorn

    from src.security.tokens import require_session_secret

    require_session_secret()   # 金鑰不合格就不啟動，避免帶著可偽造的登入上線

    import os

    # 埠號可用 VMS_PORT 調整（部分 Windows 的 8000 埠被系統服務 HTTP.sys 占用）
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("VMS_PORT", "8000")))
