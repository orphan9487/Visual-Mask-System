"""Compatibility launcher for the pure WebSocket chat application."""

from src.interfaces.websocket_app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
