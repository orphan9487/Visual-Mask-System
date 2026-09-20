"""Launcher for the dependency-isolated Breeze2 emotion service."""

from src.interfaces.emotion_inference_app import app
from src.services.emotion_service import warmup


if __name__ == "__main__":
    import uvicorn

    # Load Breeze2 before accepting traffic. A dependency, CUDA, or checkpoint
    # failure is then visible immediately instead of silently serving fallback
    # keyword predictions.
    warmup()
    uvicorn.run(app, host="127.0.0.1", port=8010)
