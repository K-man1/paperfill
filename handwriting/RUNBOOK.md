# One-DM handwriting — setup runbook

Goal: PaperFill renders answers in a user's handwriting instead of the `helv`
font. Diffusion runs on a serverless Modal GPU; Nest stays the orchestrator.

The integration is **off by default** — until `ONEDM_MODAL_URL` is set, every
fill renders typeset text exactly as before. Turn it on by completing the steps
below.

## Architecture recap

```
Nest (Flask)                         Modal (T4 GPU, scale-to-zero)
  POST /api/style  ── attach sample ──►  (stored on the job)
  POST /api/fill   ── all answers ─────►  One-DM per word → stitch → transparent PNG
  render.py insert_image(slot bbox) ◄──── {overlay_id: png}
```

One batched call per worksheet = one model-load cold start (~20s) + ~0.4s/word.
80 words ≈ under a minute, inside your 2-min/20-question budget.

---

> ✅ **DEPLOYED & VERIFIED** (workspace `k-man1`). Live endpoint:
> `https://k-man1--paperfill-onedm-onedm-generate.modal.run`
> A smoke test ("hello world") returned legible cursive in ~35s cold.
>
> `modal_app.py` is pinned to One-DM's real source (UNet config from
> `configs/IAM64.yml`, the exact `ddim_sample` signature, [0,1] normalization,
> the `ksize=1` Laplacian) plus the dependency pins below.
>
> **Dependency pins that were required** (2024-era repo on Modal's 2025 index):
> - `numpy==1.26.4` — torch/torchvision 1.13 were built against NumPy 1.x
> - `huggingface_hub==0.20.3` — diffusers 0.24 uses `cached_download` (gone in ≥0.26)
> - `omegaconf==2.3.0` — imported by `models/unet.py`
> - Python `3.10` (Modal dropped 3.8; torch 1.13.1 has cp310 wheels)

## Step 1 — Get the two required assets (free)

Inference needs exactly **two** files. The `vae_HTR138.pth` (OCR) and
`RN18_class_10400.pth` (ResNet18) in the README are **training-only losses** —
`test.py` never loads them, and neither do we. The SD VAE is pulled from HF
automatically.

1. **`One-DM-ckpt.pt`** — the UNet checkpoint.
   From the README "Model Zoo" Google Drive folder:
   https://drive.google.com/drive/folders/10KOQ05HeN2kaR2_OCZNl9D_Kh1p8BDaa
   (or ShiZhi/wisemodel direct:
   https://wisemodel.cn/models/SCUT-MMPR/One-DM/blob/main/One-DM-ckpt.pt)

2. **`unifont.pickle`** — the content-glyph dictionary that tells the model
   *which letters* to write. It is NOT in the GitHub repo (`.gitignore` excludes
   `data/*`); it ships inside the English dataset. From the README "Datasets"
   Google Drive folder, download `unifont.pickle` (tiny) if browsable
   individually, otherwise download `English_data.zip`, unzip, and grab
   `data/unifont.pickle`:
   https://drive.google.com/drive/folders/108TB-z2ytAZSIEzND94dyufybjpqVyn6

## Step 2 — Put both files on a Modal Volume (once)

```bash
modal volume create onedm-ckpts
modal volume put onedm-ckpts /path/to/One-DM-ckpt.pt /One-DM-ckpt.pt
modal volume put onedm-ckpts /path/to/unifont.pickle /unifont.pickle
```

## Step 3 — Deploy the service

1. `modal deploy handwriting/modal_app.py`
2. Modal prints the endpoint URL (form:
   `https://<you>--paperfill-onedm-onedm-generate.modal.run`).
3. Smoke-test it (style_b64 = base64 of any ~64px-tall word image):
   ```bash
   curl -X POST <url> -H 'content-type: application/json' \
     -d '{"style_b64":"<base64 png>","items":{"t":"hello world"}}'
   ```
   You should get back `{"t": "<base64 transparent png>"}`.

## Step 4 — Point the app at it (two env vars, via .env on Nest)

The endpoint is token-protected: the token lives in the `paperfill-onedm-auth`
Modal Secret (key `ONEDM_AUTH_TOKEN`) and must match the app's env var. Wrong/
missing token → 401.

The app deploys to **Hack Club Nest**, not Fly (the `fly.toml` is leftover and
unused). `app.py` loads `.env` itself at startup, and `.env` is gitignored, so it
does NOT sync via `git push` — edit it on the Nest container directly:

```bash
# on the Nest box, in the project dir:
cat >> .env <<'EOF'
ONEDM_MODAL_URL=https://k-man1--paperfill-onedm-onedm-generate.modal.run
ONEDM_AUTH_TOKEN=4KvVkDLiHRSIWjvWIBv1jqGLCiVS5n3L
EOF
pip install -r requirements.txt   # picks up new requests + pillow
# then restart however you run the app on Nest (systemd service / pm2 / tmux)
```

`handwriting_enabled()` now returns True (it only checks `ONEDM_MODAL_URL`; the
client sends `ONEDM_AUTH_TOKEN` automatically). Flow:
1. Upload a PDF as usual → get `job_id`.
2. `POST /api/style` with the user's handwriting sample (multipart `style` file
   or JSON `style_b64`) for that `job_id`.
3. `POST /api/fill` → answers are generated, sent to Modal in one batch, and the
   returned PNGs are stamped into the slot bboxes by `render.py`.

If Modal is down or errors, `_generate_hw_for_job` logs and returns — the fill
falls back to normal text. Nothing breaks.

---

## Where the code lives

| File | Role |
|------|------|
| `handwriting/modal_app.py` | GPU service: preprocess → per-word diffusion → stitch → transparent PNG |
| `handwriting/client.py`    | Nest-side batched HTTP client (`ONEDM_MODAL_URL`, `ONEDM_TIMEOUT`) |
| `render.py`                | `render_overlays_pdf(..., images=)` stamps PNGs via `insert_image` |
| `app.py`                   | `/api/style` route, `_generate_hw_for_job`, hw PNG cache under `outputs/<job>/hw/` |

## Tuning knobs

- **Speed**: `scaledown_window` in `modal_app.py` keeps the container warm longer
  (avoids repeat cold starts during a session). Caching identical words across a
  worksheet is already done in `generate()`.
- **Look**: `space_px` in `stitch_words_to_line` controls word spacing; the
  alpha curve in `threshold_to_transparent` controls ink darkness/anti-aliasing.
- **Quality vs. cost**: switch DDIM→DDPM or raise sampling steps for cleaner
  glyphs at higher latency.
