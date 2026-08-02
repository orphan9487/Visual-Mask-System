"""Create a labelled v2/v3 contact sheet from controlled Henry test outputs."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EMOTIONS = ("anger", "joy", "sadness", "surprise", "fear", "disgust", "neutral")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output" / "current" / "henry_lora_comparison"
CELL = 256
LABEL_HEIGHT = 34


def latest(emotion: str, version_start: str) -> Path:
    matches = sorted(OUTPUT_DIR.glob(f"sticker_{emotion}_lw0p80_*.png"))
    if version_start == "v2":
        return matches[0]
    return matches[1]


def main() -> None:
    font = ImageFont.load_default()
    width = CELL * 2
    height = LABEL_HEIGHT + len(EMOTIONS) * (CELL + LABEL_HEIGHT)
    sheet = Image.new("RGB", (width, height), "#181818")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "v2 — U-Net only", fill="white", font=font)
    draw.text((CELL + 8, 10), "v3 — U-Net + Text Encoder", fill="white", font=font)

    for row, emotion in enumerate(EMOTIONS):
        y = LABEL_HEIGHT + row * (CELL + LABEL_HEIGHT)
        for col, version in enumerate(("v2", "v3")):
            image = Image.open(latest(emotion, version)).convert("RGB")
            image.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
            x = col * CELL + (CELL - image.width) // 2
            sheet.paste(image, (x, y))
        draw.text((8, y + CELL + 8), emotion, fill="white", font=font)

    path = OUTPUT_DIR / "v2_v3_comparison.png"
    sheet.save(path)
    print(path)


if __name__ == "__main__":
    main()
