# Deploying PaperFill to Hack Club Nest + ww2explained.com

This is the production deploy. The whole repo is the deploy unit — you push it
to GitHub, `git clone` it on Nest, install deps into a venv, run gunicorn, and
point Caddy (Nest's reverse proxy) at it. Your Cloudflare domain
`ww2explained.com` then resolves to Nest.

```
ww2explained.com  ──Cloudflare DNS (grey cloud)──►  Nest box
                                                       │
                                               Caddy (TLS, 443)
                                                       │ reverse_proxy
                                                       ▼
                                          gunicorn → app:app  (PORT)
```

---

## 1. Push to GitHub (from your laptop)

```bash
cd ~/Desktop/PaperFill
git add -A
git commit -m "Add Nest production deploy config"
git push        # or `gh repo create` first if there's no remote yet
```

`.env`, `uploads/`, `outputs/`, and `signin_log.json` are gitignored — secrets
and runtime data never leave your machine.

## 2. Clone + set up on Nest

SSH into Nest, then:

```bash
git clone <your-repo-url> paperfill
cd paperfill

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Get your port and create `.env`

```bash
nest get-port          # prints a free port, e.g. 38291
cp .env.example .env
nano .env
```

In `.env` set:

- `PORT=` → the number `nest get-port` gave you
- `SECRET_KEY=` → run `python -c "import secrets;print(secrets.token_hex(32))"` and paste it
- `HCAI_API_KEY=` → your Hack Club AI key (free, no card)
- `ONEDM_MODAL_URL` / `ONEDM_AUTH_TOKEN` → only if you want handwriting; otherwise leave blank

> These are the same three secrets that were on the old server. Recreate them
> here — they were never committed.

## 4. Run it (keep it alive)

Quick test in the foreground:

```bash
./run.sh
# visit http://localhost:PORT on the box, Ctrl-C to stop
```

To keep it running after you log out, use a Nest-managed service. The simplest
is a systemd **user** service (no sudo needed on Nest):

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/paperfill.service <<'EOF'
[Unit]
Description=PaperFill (gunicorn)
After=network.target

[Service]
WorkingDirectory=%h/paperfill
ExecStart=%h/paperfill/run.sh
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now paperfill
loginctl enable-linger "$USER"      # keep the service up after you disconnect

systemctl --user status paperfill   # check it's running
journalctl --user -u paperfill -f   # tail logs
```

(`run.sh` activates the venv and loads `.env` via app.py, so the service picks
up your PORT and keys automatically.)

## 5. Wire up Caddy (the reverse proxy)

The `Caddyfile` in this repo has the block you need. Add it to your Nest Caddy
config the way Nest documents (the `nest` CLI / your user Caddyfile — see
https://guides.hackclub.app), then reload Caddy.

The block routes both `ww2explained.com` (+ `www`) and `karman.hackclub.app` to
`localhost:$PORT`. Make sure the `PORT` Caddy sees matches your `.env`, or
hardcode the number in the Caddyfile instead of `{$PORT:8080}`.

`karman.hackclub.app` will work immediately (Nest already owns that cert).

## 6. Point ww2explained.com at Nest (Cloudflare)

In the Cloudflare dashboard for `ww2explained.com` → **DNS**:

| Type  | Name | Target                | Proxy status            |
|-------|------|-----------------------|-------------------------|
| CNAME | `@`  | `karman.hackclub.app` | **DNS only** (grey ☁️)  |
| CNAME | `www`| `karman.hackclub.app` | **DNS only** (grey ☁️)  |

(Cloudflare flattens the apex CNAME automatically.)

> ⚠️ **Keep these grey-cloud / "DNS only" at least until HTTPS works.** If you
> turn on Cloudflare's orange-cloud proxy, Caddy on Nest can't complete the
> Let's Encrypt challenge and you'll get a cert error. Once Caddy has issued the
> cert (visit https://ww2explained.com and confirm the padlock), you *may*
> switch to orange-cloud with SSL/TLS mode set to **Full (strict)** — but
> grey-cloud is the no-surprises default.

DNS can take a few minutes to propagate. Then visit **https://ww2explained.com**
— you'll hit the PaperFill login page.

---

## Updating after the first deploy

```bash
cd ~/paperfill
git pull
source venv/bin/activate && pip install -r requirements.txt   # if deps changed
systemctl --user restart paperfill
```

## Login

The app gates behind an access code (set in `app.py`): `spurs` = user,
`alien` = admin (`/admin` shows the sign-in log). Change these before sharing
the URL widely.

## Notes

- **Runtime data** (`uploads/`, `outputs/`, `signin_log.json`) lives on the
  Nest disk and persists across restarts. The in-memory job cache is rebuilt
  from disk on demand, so a restart won't lose filled PDFs.
- **Handwriting** is optional. With `ONEDM_MODAL_URL` unset the app renders
  typed answers; set it to enable One-DM handwriting (see `handwriting/RUNBOOK.md`).
- **Other hosts**: the included `Dockerfile` still works on any container host
  (Render, Fly, etc.) — it runs the same gunicorn config. Nest is the free path.
