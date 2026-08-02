"""
Stable Diffusion Principle Diagram
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

BG       = "#0d1117"
C_VAE    = "#238636"
C_CLIP   = "#1f6feb"
C_UNET   = "#9a3fb5"
C_NOISE  = "#e36209"
C_LATENT = "#0e7490"
C_ARROW  = "#8b949e"
C_WHITE  = "#e6edf3"
C_GRAY   = "#8b949e"
C_GOLD   = "#d4a017"

fig = plt.figure(figsize=(22, 12), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_facecolor(BG)


def box(ax, x, y, w, h, color, alpha=0.18, radius=0.25, lw=1.8, ls="-"):
    rect = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, linestyle=ls,
        edgecolor=color, facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(rect)
    rect2 = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, linestyle=ls,
        edgecolor=color, facecolor="none", alpha=0.85, zorder=3)
    ax.add_patch(rect2)


def lbl(ax, x, y, text, color=C_WHITE, size=9, bold=False, ha="center", va="center"):
    w = "bold" if bold else "normal"
    ax.text(x, y, text, color=color, fontsize=size, fontweight=w,
            ha=ha, va=va, zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])


def arr(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)


def noise_patch(ax, cx, cy, size=0.5, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((12, 12))
    ext = [cx-size, cx+size, cy-size, cy+size]
    ax.imshow(data, extent=ext, cmap="gray", vmin=0, vmax=1,
              aspect="auto", zorder=4, alpha=0.9)
    ax.add_patch(plt.Rectangle((cx-size, cy-size), 2*size, 2*size,
                 fill=False, edgecolor=C_NOISE, lw=1.5, zorder=5))


def latent_patch(ax, cx, cy, w=0.8, h=0.4):
    rng = np.random.default_rng(7)
    data = rng.random((8, 8))
    ext = [cx-w, cx+w, cy-h, cy+h]
    ax.imshow(data, extent=ext, cmap="viridis", aspect="auto", zorder=4, alpha=0.9)
    ax.add_patch(plt.Rectangle((cx-w, cy-h), 2*w, 2*h,
                 fill=False, edgecolor=C_LATENT, lw=1.5, zorder=5))


def img_patch(ax, cx, cy, size=0.55, border=C_VAE):
    n = 18
    data = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            data[i, j] = [j/n*0.5, i/n*0.85, 0.45]
    ext = [cx-size, cx+size, cy-size, cy+size]
    ax.imshow(data, extent=ext, aspect="auto", zorder=4, alpha=0.92)
    ax.add_patch(plt.Rectangle((cx-size, cy-size), 2*size, 2*size,
                 fill=False, edgecolor=border, lw=1.8, zorder=5))


# ── Title ─────────────────────────────────────────────────────────────
lbl(ax, 11, 11.45, "Stable Diffusion  —  Latent Diffusion Model (LDM)",
    color=C_WHITE, size=16, bold=True)
lbl(ax, 11, 10.95, "Text-Conditioned Image Generation via Latent Space Diffusion",
    color=C_GRAY, size=10)

# ── Zone backgrounds ──────────────────────────────────────────────────
box(ax, 0.3,  1.0, 3.2, 9.6, C_VAE,    alpha=0.06, radius=0.5, lw=1.0, ls="--")
box(ax, 3.8,  1.0, 14.2, 9.6, C_LATENT, alpha=0.05, radius=0.5, lw=1.0, ls="--")
box(ax, 18.3, 1.0, 3.2, 9.6, C_VAE,    alpha=0.06, radius=0.5, lw=1.0, ls="--")

lbl(ax, 1.9,  10.35, "IMAGE SPACE", color=C_VAE, size=8.5, bold=True)
lbl(ax, 10.9, 10.35, "LATENT SPACE  ·  DIFFUSION PROCESS", color=C_LATENT, size=8.5, bold=True)
lbl(ax, 19.9, 10.35, "IMAGE SPACE", color=C_VAE, size=8.5, bold=True)

# ══════════════════════════════════════════════════════════════════════
# LEFT — Input Image + VAE Encoder
# ══════════════════════════════════════════════════════════════════════
# Input image
box(ax, 0.55, 6.8, 2.7, 2.5, C_VAE, radius=0.2)
img_patch(ax, 1.9, 8.05, size=0.7)
lbl(ax, 1.9, 7.05, "Input Image", color=C_VAE, size=9, bold=True)
lbl(ax, 1.9, 6.72, "512 x 512 x 3", color=C_GRAY, size=8)

arr(ax, 1.9, 6.8, 1.9, 6.38, color=C_VAE)

# VAE Encoder
box(ax, 0.55, 4.8, 2.7, 1.5, C_VAE, radius=0.2)
lbl(ax, 1.9, 5.75, "VAE Encoder", color=C_VAE, size=10, bold=True)
lbl(ax, 1.9, 5.35, "KL-regularized", color=C_WHITE, size=8.5)
lbl(ax, 1.9, 5.0,  "compress x8", color=C_GRAY, size=7.5)

arr(ax, 1.9, 4.8, 1.9, 4.35, color=C_VAE)

# Latent z0 left
box(ax, 0.55, 3.0, 2.7, 1.2, C_LATENT, radius=0.2)
latent_patch(ax, 1.9, 3.6)
lbl(ax, 1.9, 3.1, "Latent  z0   64x64x4", color=C_LATENT, size=8, bold=True)

# ══════════════════════════════════════════════════════════════════════
# FORWARD DIFFUSION  (top row)
# ══════════════════════════════════════════════════════════════════════
lbl(ax, 9.0, 9.82, "Forward Process   q(z_t | z_{t-1})  =  N(z_t ;  sqrt(1-B_t) z_{t-1}, B_t I)",
    color=C_NOISE, size=9, bold=True)

steps_fwd = [(4.3, 8.5, 0, "z_0"),
             (6.5, 8.5, 2, "z_{T/4}"),
             (8.7, 8.5, 4, "z_{T/2}"),
             (10.9, 8.5, 6, "z_T")]

for cx, cy, seed, tag in steps_fwd:
    noise_patch(ax, cx, cy, size=0.5, seed=seed)
    lbl(ax, cx, cy-0.78, tag, color=C_NOISE, size=8.5)

for i in range(len(steps_fwd)-1):
    x1 = steps_fwd[i][0]   + 0.52
    x2 = steps_fwd[i+1][0] - 0.52
    y  = steps_fwd[i][1]
    arr(ax, x1, y, x2, y, color=C_NOISE, lw=1.8)
    lbl(ax, (x1+x2)/2, y+0.22, "+ e", color=C_NOISE, size=9)

# ══════════════════════════════════════════════════════════════════════
# TEXT ENCODER (CLIP)
# ══════════════════════════════════════════════════════════════════════
# Prompt
box(ax, 3.9, 5.5, 5.5, 1.0, C_CLIP, radius=0.2)
lbl(ax, 6.65, 6.15, '"a happy person smiling"', color=C_WHITE, size=9.5)
lbl(ax, 6.65, 5.73, "Text Prompt", color=C_CLIP, size=8.5, bold=True)

arr(ax, 6.65, 6.5, 6.65, 6.98, color=C_CLIP, lw=1.8)

# CLIP
box(ax, 3.9, 7.0, 5.5, 1.6, C_CLIP, radius=0.2)
lbl(ax, 6.65, 8.1,  "CLIP Text Encoder", color=C_CLIP, size=11, bold=True)
lbl(ax, 6.65, 7.72, "ViT-L/14  |  Transformer", color=C_WHITE, size=9)
lbl(ax, 6.65, 7.35, "Tokenize  ->  Embed  ->  Self-Attention", color=C_GRAY, size=8)

arr(ax, 6.65, 8.6, 6.65, 9.08, color=C_CLIP, lw=1.8)

# Embeddings
box(ax, 3.9, 9.1, 5.5, 1.0, C_CLIP, radius=0.2)
lbl(ax, 6.65, 9.72, "Text Embeddings", color=C_CLIP, size=10, bold=True)
lbl(ax, 6.65, 9.34, "77 tokens  x  768 dim", color=C_WHITE, size=9)

# ══════════════════════════════════════════════════════════════════════
# U-NET DENOISER
# ══════════════════════════════════════════════════════════════════════
box(ax, 10.0, 4.2, 7.8, 5.0, C_UNET, radius=0.35, alpha=0.18, lw=2.0)
lbl(ax, 13.9, 9.0, "U-Net Denoiser  e_q(z_t, t, c)", color=C_UNET, size=11.5, bold=True)

# Encoder block
box(ax, 10.3, 7.0, 2.0, 1.8, C_UNET, radius=0.2, alpha=0.35)
lbl(ax, 11.3, 8.5, "Encoder", color=C_WHITE, size=9, bold=True)
lbl(ax, 11.3, 8.1, "Down", color=C_GRAY, size=8)
lbl(ax, 11.3, 7.75, "ResNet", color=C_GRAY, size=8)
lbl(ax, 11.3, 7.38, "+ Attn", color=C_GRAY, size=8)

# Bottleneck
box(ax, 12.6, 7.0, 2.6, 1.8, C_UNET, radius=0.2, alpha=0.5)
lbl(ax, 13.9, 8.55, "Bottleneck", color=C_WHITE, size=9, bold=True)
lbl(ax, 13.9, 8.15, "Cross-Attention", color=C_GOLD, size=9, bold=True)
lbl(ax, 13.9, 7.78, "Q from z_t", color=C_GRAY, size=7.8)
lbl(ax, 13.9, 7.42, "K,V from text", color=C_GRAY, size=7.8)

# Decoder block
box(ax, 15.5, 7.0, 2.0, 1.8, C_UNET, radius=0.2, alpha=0.35)
lbl(ax, 16.5, 8.5, "Decoder", color=C_WHITE, size=9, bold=True)
lbl(ax, 16.5, 8.1, "Up", color=C_GRAY, size=8)
lbl(ax, 16.5, 7.75, "ResNet", color=C_GRAY, size=8)
lbl(ax, 16.5, 7.38, "+ Attn", color=C_GRAY, size=8)

# Skip connections
ax.annotate("", xy=(15.5, 7.5), xytext=(12.3, 7.5),
            arrowprops=dict(arrowstyle="-", color=C_UNET, lw=1.2, ls="dashed",
                            connectionstyle="arc3,rad=-0.35"), zorder=3)
lbl(ax, 13.9, 6.75, "Skip Connections", color=C_UNET, size=8)

# Noise scheduler
box(ax, 10.3, 4.6, 7.3, 1.8, C_NOISE, radius=0.25, alpha=0.22)
lbl(ax, 13.95, 6.15, "Noise Scheduler", color=C_NOISE, size=10, bold=True)
lbl(ax, 13.95, 5.78, "DDPM / DDIM / DPM-Solver++", color=C_WHITE, size=9)
lbl(ax, 13.95, 5.42, "z_{t-1}  =  (z_t - sqrt(1-a_t) * e_q) / sqrt(a_t)  +  sigma_t * noise",
    color=C_GRAY, size=8)
lbl(ax, 13.95, 5.0, "Inference: T steps  ->  T-1  ->  ...  ->  0  (50 steps DDIM)",
    color=C_GRAY, size=7.8)

# Time step
box(ax, 10.3, 4.62, 2.2, 0.7, C_NOISE, radius=0.15, alpha=0.4)
lbl(ax, 11.4, 4.97, "Time Step  t", color=C_NOISE, size=9, bold=True)

# ══════════════════════════════════════════════════════════════════════
# REVERSE DIFFUSION  (bottom row)
# ══════════════════════════════════════════════════════════════════════
lbl(ax, 9.0, 3.75, "Reverse Process   p_q(z_{t-1} | z_t, c)  — U-Net predicts noise  e_q",
    color=C_UNET, size=9, bold=True)

steps_rev = [(10.9, 2.8, 6, "z_T"),
             (8.7,  2.8, 4, "z_{T/2}"),
             (6.5,  2.8, 2, "z_{T/4}"),
             (4.3,  2.8, 0, "z_0'")]

for cx, cy, seed, tag in steps_rev:
    noise_patch(ax, cx, cy, size=0.5, seed=seed)
    lbl(ax, cx, cy-0.78, tag, color=C_UNET, size=8.5)

for i in range(len(steps_rev)-1):
    x1 = steps_rev[i][0]   - 0.52
    x2 = steps_rev[i+1][0] + 0.52
    y  = steps_rev[i][1]
    arr(ax, x1, y, x2, y, color=C_UNET, lw=1.8)
    lbl(ax, (x1+x2)/2, y+0.22, "- e", color=C_UNET, size=9)

# ══════════════════════════════════════════════════════════════════════
# RIGHT — VAE Decoder + Output
# ══════════════════════════════════════════════════════════════════════
# Latent z0' right
box(ax, 18.55, 3.0, 2.7, 1.2, C_LATENT, radius=0.2)
latent_patch(ax, 19.9, 3.6)
lbl(ax, 19.9, 3.1, "Latent  z0'  64x64x4", color=C_LATENT, size=8, bold=True)

arr(ax, 19.9, 4.2, 19.9, 4.65, color=C_VAE)

# VAE Decoder
box(ax, 18.55, 4.65, 2.7, 1.5, C_VAE, radius=0.2)
lbl(ax, 19.9, 5.6,  "VAE Decoder", color=C_VAE, size=10, bold=True)
lbl(ax, 19.9, 5.2,  "KL-regularized", color=C_WHITE, size=8.5)
lbl(ax, 19.9, 4.85, "upsample x8", color=C_GRAY, size=7.5)

arr(ax, 19.9, 6.15, 19.9, 6.65, color=C_VAE)

# Output image
box(ax, 18.55, 6.65, 2.7, 2.5, C_VAE, radius=0.2)
img_patch(ax, 19.9, 7.9, size=0.7, border="#60a5fa")
lbl(ax, 19.9, 6.9, "Generated Image", color=C_VAE, size=9, bold=True)
lbl(ax, 19.9, 6.58, "512 x 512 x 3", color=C_GRAY, size=8)

# ══════════════════════════════════════════════════════════════════════
# CONNECTING ARROWS
# ══════════════════════════════════════════════════════════════════════
# z0 left -> forward step0
arr(ax, 1.9, 3.0,  1.9, 2.35, color=C_LATENT, lw=1.5)
ax.plot([1.9, 3.78], [2.35, 2.35], color=C_LATENT, lw=1.5, zorder=4)
arr(ax, 3.78, 2.35, 3.78, 8.0, color=C_LATENT, lw=1.5)

# z_T -> U-Net top
arr(ax, 10.9, 8.0, 11.3, 8.8, color=C_NOISE, lw=1.5)

# Text embeddings -> cross-attention
arr(ax, 9.4, 9.6, 13.9, 8.8, color=C_CLIP, lw=1.8)
lbl(ax, 11.9, 9.3, "Cross-Attention Conditioning", color=C_GOLD, size=8.5, bold=True)

# U-Net output -> reverse z_T
arr(ax, 13.9, 6.96, 13.9, 6.62, color=C_UNET, lw=1.5)
arr(ax, 13.9, 6.62, 10.9, 3.3, color=C_UNET, lw=1.5)

# z0' -> VAE decoder right
ax.plot([4.3, 18.55], [2.02, 2.02], color=C_LATENT, lw=1.5, zorder=4)
arr(ax, 18.55, 2.02, 18.55, 3.0, color=C_LATENT, lw=1.5)
arr(ax, 18.55, 3.0, 18.55, 3.1, color=C_LATENT, lw=1.5)
ax.plot([4.3, 4.3], [2.3, 2.02], color=C_LATENT, lw=1.5, zorder=4)

# ══════════════════════════════════════════════════════════════════════
# CFG box
# ══════════════════════════════════════════════════════════════════════
box(ax, 3.9, 1.08, 5.5, 1.1, C_GOLD, radius=0.2, alpha=0.15)
lbl(ax, 6.65, 1.82, "Classifier-Free Guidance (CFG)", color=C_GOLD, size=9, bold=True)
lbl(ax, 6.65, 1.42, "e_guided = e_uncond + scale x (e_cond - e_uncond)", color=C_WHITE, size=8.5)

# LoRA note
box(ax, 9.7, 1.08, 4.5, 1.1, "#e05d44", radius=0.2, alpha=0.15)
lbl(ax, 11.95, 1.82, "LoRA Fine-tuning (this project)", color="#e05d44", size=9, bold=True)
lbl(ax, 11.95, 1.42, "Delta_W = alpha/r * B*A   injected into U-Net Attn layers", color=C_WHITE, size=8.5)

# ══════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════
items = [(C_VAE,    "VAE (Encoder/Decoder)"),
         (C_CLIP,   "CLIP Text Encoder"),
         (C_UNET,   "U-Net Denoiser"),
         (C_NOISE,  "Forward Diffusion"),
         (C_LATENT, "Latent Space"),
         (C_GOLD,   "Conditioning / CFG"),
         ("#e05d44","LoRA (this project)")]

for i, (color, text) in enumerate(items):
    bx = 14.6 + (i % 4) * 1.85
    by = 1.82 - (i // 4) * 0.42
    ax.plot(bx, by, "s", color=color, markersize=8, zorder=6)
    lbl(ax, bx+0.12, by, text, color=C_GRAY, size=7.5, ha="left")

# Save
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs" / "generated" / "architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "sd_principle_diagram.png"
plt.savefig(OUT, dpi=155, bbox_inches="tight", facecolor=BG, edgecolor="none")
print(f"Saved: {OUT}")
