"""Compatibility launcher for the pure WebSocket chat application."""

from src.interfaces.multipage_app import app


if __name__ == "__main__":
    import uvicorn

    from src.security.tokens import require_session_secret

    require_session_secret()   # 金鑰不合格就不啟動，避免帶著可偽造的登入上線

    uvicorn.run(app, host="127.0.0.1", port=8000)
