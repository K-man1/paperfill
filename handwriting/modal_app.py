"""
Modal GPU service: One-DM handwriting generation for PaperFill.

WHY this exists: the Nest container (2GB RAM, no GPU) cannot run One-DM's
diffusion stack. So inference lives here on a serverless T4, scaling to zero.
PaperFill (on Nest) sends ALL answers in ONE request; we pay the model-load
cost once, loop on the warm GPU, and return a transparent PNG per slot.

This is pinned to One-DM's actual source (commit on main, configs/IAM64.yml):
  * UNetModel(in=4, model_channels=512, out=4, num_res_blocks=1,
              attention_resolutions=(1,1), channel_mult=(1,1),
              num_heads=4, context_dim=512)         -- matches test.py + IAM64
  * Diffusion.ddim_sample(model, vae, n, x, styles, laplace, content, steps, eta)
  * style & laplace normalized to [0,1] (just /255), laplace kernel = ksize=1
  * INFERENCE NEEDS ONLY the One-DM UNet ckpt + an SD VAE. The OCR (vae_HTR138)
    and ResNet18 checkpoints are TRAINING-only losses -- test.py never loads them.

Deploy:   modal deploy handwriting/modal_app.py
Endpoint: POST https://<you>--paperfill-onedm-onedm-generate.modal.run
          body: {"style_b64": "<base64 png>", "items": {"<slot_id>": "<text>"}}
          resp: {"<slot_id>": "<base64 transparent png>"}
"""

import base64
import io

import modal

# ---------------------------------------------------------------------------
# Container image: torch 1.13 + the One-DM repo + its python deps.
# Only the One-DM UNet checkpoint lives on the Volume; the SD VAE is pulled
# from HuggingFace and cached.
# ---------------------------------------------------------------------------
CKPT_DIR = "/ckpts"
ckpt_vol = modal.Volume.from_name("onedm-ckpts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")   # 3.8 dropped by Modal; torch 1.13.1 has cp310 wheels
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==1.13.1", "torchvision==0.14.1",
        "diffusers==0.24.0", "transformers==4.30.2",
        # torch/torchvision 1.13 were built against NumPy 1.x:
        "numpy==1.26.4",
        # diffusers 0.24 uses huggingface_hub.cached_download (removed in >=0.26):
        "huggingface_hub==0.20.3",
        "opencv-python-headless", "pillow",
        "easydict", "einops", "tqdm", "omegaconf==2.3.0", "fastapi[standard]",
        # data_loader/loader.py imports these at module level (we only use
        # ContentData from it, but the whole module must import cleanly):
        "lmdb", "six", "pyyaml", "packaging",
    )
    .run_commands("git clone https://github.com/dailenson/One-DM.git /One-DM")
)

app = modal.App("paperfill-onedm")


# ---------------------------------------------------------------------------
# Style-image preprocessing: a user upload -> what One-DM's encoder expects.
# One-DM's dataset feeds BOTH the grayscale style image AND its Laplacian
# high-frequency map, each as [N,1,64,W] normalized to [0,1]. Their pipeline
# reads a PRE-COMPUTED laplace from disk; we reproduce it at runtime with the
# same 3x3 kernel ([[0,1,0],[1,-4,1],[0,1,0]] == cv2.Laplacian ksize=1).
# ---------------------------------------------------------------------------
def preprocess_style(png_bytes: bytes):
    """Return (style_tensor, laplace_tensor), both [1,1,64,W] in [0,1]."""
    import cv2
    import numpy as np
    import torch

    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)         # H x W uint8
    h, w = img.shape
    new_w = max(8, int(round(w * (64.0 / h))))
    img = cv2.resize(img, (new_w, 64), interpolation=cv2.INTER_CUBIC)

    laplace = cv2.Laplacian(img, cv2.CV_8U, ksize=1)      # matches their kernel

    def to_tensor(x):
        t = torch.from_numpy(x).float() / 255.0           # -> [0,1] (no -1..1!)
        return t.unsqueeze(0).unsqueeze(0)                # [1,1,64,W]

    return to_tensor(img), to_tensor(laplace)


def stitch_words_to_line(word_imgs, space_px: int = 18):
    """Composite per-word PIL 'L' images (height ~64) into one baseline-aligned
    line on a white canvas (white -> transparent later)."""
    from PIL import Image

    if not word_imgs:
        return Image.new("L", (1, 64), 255)

    height = max(im.size[1] for im in word_imgs)
    total_w = sum(im.size[0] for im in word_imgs) + space_px * (len(word_imgs) - 1)
    canvas = Image.new("L", (total_w, height), 255)

    x = 0
    for im in word_imgs:
        y = height - im.size[1]          # bottom-align: shared baseline
        canvas.paste(im, (x, y))
        x += im.size[0] + space_px
    return canvas


def threshold_to_transparent(line_img):
    """Gray line image (dark ink on white) -> RGBA PNG bytes; white paper
    becomes fully transparent so only the ink overlays the worksheet."""
    from PIL import Image
    import numpy as np

    g = np.asarray(line_img.convert("L"))
    rgba = np.zeros((*g.shape, 4), dtype=np.uint8)
    rgba[..., 3] = np.clip(255 - g, 0, 255).astype(np.uint8)   # alpha from ink darkness
    out = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The GPU worker. @enter loads the UNet + VAE once per warm container.
# ---------------------------------------------------------------------------
@app.cls(image=image, gpu="T4", volumes={CKPT_DIR: ckpt_vol},
         scaledown_window=180,
         secrets=[modal.Secret.from_name("paperfill-onedm-auth")])
class OneDM:
    @modal.enter()
    def load(self):
        import os, sys, shutil, torch
        os.chdir("/One-DM")                       # ContentData reads data/unifont.pickle (relative)
        sys.path.insert(0, "/One-DM")

        # The content-glyph dictionary isn't in the repo (.gitignore excludes
        # data/*); it lives on the Volume. Put it where ContentData expects it.
        os.makedirs("/One-DM/data", exist_ok=True)
        if not os.path.exists("/One-DM/data/unifont.pickle"):
            shutil.copy(f"{CKPT_DIR}/unifont.pickle", "/One-DM/data/unifont.pickle")

        from models.unet import UNetModel
        from models.diffusion import Diffusion
        from data_loader.loader import ContentData
        from diffusers import AutoencoderKL

        self.device = "cuda"
        self.torch = torch
        self.content = ContentData()              # text -> unifont content tensor

        # IAM64.yml config values, mapped exactly as test.py constructs them.
        unet = UNetModel(
            in_channels=4, model_channels=512, out_channels=4,
            num_res_blocks=1, attention_resolutions=(1, 1),
            channel_mult=(1, 1), num_heads=4, context_dim=512,
        ).to(self.device)
        ckpt = torch.load(f"{CKPT_DIR}/One-DM-ckpt.pt", map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]       # tolerate wrapped checkpoints
        unet.load_state_dict(ckpt, strict=False)
        self.unet = unet.eval()

        # test.py uses runwayml/stable-diffusion-v1-5 (subfolder vae), which was
        # delisted from HF — this is the maintained community re-host of the
        # SAME weights, so output matches what One-DM trained against.
        self.vae = AutoencoderKL.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            subfolder="vae").to(self.device).eval()
        self.diff = Diffusion(device=self.device)

        # Only characters One-DM has content glyphs for are renderable.
        self.vocab = set(self.content.letters)

    def _generate_word(self, word, style_t, laplace_t):
        """Diffusion-sample a single word -> PIL 'L' image, or None if the word
        has no renderable characters."""
        import unicodedata
        import torchvision
        torch = self.torch

        # The IAM vocab is plain ASCII — it has no á é í ó ú ñ ü. Simply dropping
        # those codepoints doesn't just lose the accent, it CORRUPTS the whole
        # word (e.g. "tenía" -> garbled "aena"). Transliterate accented Latin
        # letters to their base form first (NFKD splits "í" -> "i" + combining
        # accent; we keep the base, drop the combining mark), so "tenía"->"tenia",
        # "destruyó"->"destruyo", "anduvió"->"anduvio", "ñ"->"n".
        word = unicodedata.normalize("NFKD", word)
        word = "".join(c for c in word if not unicodedata.combining(c))
        word = "".join(ch for ch in word if ch in self.vocab)
        if not word:
            return None

        with torch.no_grad():
            content_t = self.content.get_content(word).to(self.device)   # [1, nchars, h, w]
            n = style_t.shape[0]
            # x shape mirrors test.py exactly: (N, 4, H//8, (nchars*32)//8)
            x = torch.randn(
                (n, 4, style_t.shape[2] // 8, (content_t.shape[1] * 32) // 8),
                device=self.device,
            )
            content_t = content_t.repeat(n, 1, 1, 1)
            sampled = self.diff.ddim_sample(
                self.unet, self.vae, n, x,
                style_t.to(self.device), laplace_t.to(self.device), content_t,
                50, 0,                              # sampling_timesteps, eta
            )
            return torchvision.transforms.ToPILImage()(sampled[0].cpu()).convert("L")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, payload: dict):
        """{auth_token, style_b64, items:{slot_id:text}} -> {slot_id: transparent_png_b64}.

        If ONEDM_AUTH_TOKEN is set (via the paperfill-onedm-auth Modal Secret),
        requests must carry a matching `auth_token` or get a 401. This stops
        anyone who finds the public URL from spending your GPU budget."""
        import os
        from fastapi import HTTPException

        expected = os.environ.get("ONEDM_AUTH_TOKEN", "")
        if expected and payload.get("auth_token") != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

        style_t, laplace_t = preprocess_style(base64.b64decode(payload["style_b64"]))
        items = payload.get("items", {})

        word_cache = {}                             # dedup identical words per worksheet
        result = {}
        for slot_id, text in items.items():
            line_words = []
            for word in str(text).split():
                if word not in word_cache:
                    word_cache[word] = self._generate_word(word, style_t, laplace_t)
                img = word_cache[word]
                if img is not None:
                    line_words.append(img)
            if not line_words:
                continue
            png = threshold_to_transparent(stitch_words_to_line(line_words))
            result[slot_id] = base64.b64encode(png).decode()
        return result
