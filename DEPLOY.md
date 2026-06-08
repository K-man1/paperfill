# Deploying PaperFill on Hack Club Nest + ww2explained.com

## How this is actually set up (verified on the server)

PaperFill runs in your Nest **LXC container** (`karman`, reachable via
`ssh karman@hackclub.app`, root home `/root/paperfill`). Inside the container:

- A **systemd system service** `paperfill.service` runs gunicorn:
  `ExecStart=/root/paperfill/.venv/bin/gunicorn --bind [::]:8080 --workers 2 --timeout 300 app:app`
- The venv is **`.venv`**, the LLM/handwriting secrets live in `/root/paperfill/.env`.
- gunicorn listens on **port 8080** inside the container. That port is **not**
  exposed to the public internet.

In front of the container, Nest runs a **central Caddy gateway** (`hackclub.app`,
`65.108.74.29`) that terminates HTTPS and reverse-proxies your hostnames to the
container's `:8080`. That is why **https://karman.hackclub.app already works**
(it 302-redirects to `/login`, served by gunicorn).

```
ww2explained.com ──Cloudflare DNS (grey cloud)──► Nest gateway (Caddy, TLS)
                                                        │ reverse_proxy
                                                        ▼
                                       LXC container "karman": gunicorn :8080
                                                        │
                                              paperfill.service (systemd)
```

> The earlier "`nest get-port` + install Caddy in the container" idea does **not**
> apply: there is no `nest` CLI or Caddy inside this container, and adding a
> reverse proxy here would do nothing because the gateway is what faces the
> internet. TLS and routing are the gateway's job.

---

## Updating the running app

```bash
ssh karman@hackclub.app
cd /root/paperfill
git pull
source .venv/bin/activate && pip install -r requirements.txt   # only if deps changed
systemctl restart paperfill
systemctl status paperfill          # confirm active (running)
journalctl -u paperfill -f          # tail logs
```

Runtime data (`uploads/`, `outputs/`, `signin_log.json`) lives on the container
disk and survives restarts. The in-memory job cache is rebuilt from disk.

## Login

Access codes are in `app.py`: `spurs` = user, `alien` = admin (`/admin` shows the
sign-in log). Change them before sharing the URL.

---

## Adding the custom domain ww2explained.com

This is the part that is **not** done from inside the app container — it needs
two things: Cloudflare DNS, and a Nest gateway (Caddy) entry. Source:
https://guides.hackclub.app/index.php/Subdomains_and_Custom_Domains

### Step 1 — Cloudflare DNS for ww2explained.com

Since it's an apex/root domain, use one of these (Cloudflare supports apex
CNAME flattening, which is the simplest):

| Type  | Name | Value                 | Proxy                  |
|-------|------|-----------------------|------------------------|
| CNAME | `@`  | `karman.hackclub.app` | **DNS only** (grey ☁️) |
| CNAME | `www`| `karman.hackclub.app` | **DNS only** (grey ☁️) |

If you ever can't use a CNAME at the apex, the Nest docs' fallback is:
`A → 37.27.51.34`, `AAAA → 2a01:4f9:3081:399c::4`, **plus** a TXT record
`domain-verification=karman`.

> ⚠️ **Grey cloud / "DNS only" is required.** If Cloudflare proxies (orange
> cloud), the Nest gateway can't complete the Let's Encrypt challenge and you'll
> get a cert error. Leave it grey until HTTPS works; only then *optionally*
> switch to orange with SSL/TLS mode **Full (strict)**.

### Step 2 — Register the domain on the Nest gateway

Per the Nest guide, after DNS is set you add the domain via the **`nest` CLI**
and your **Nest Caddyfile** — these live in your **Nest shell account on the
gateway** (`hackclub.app`), *not* in this app container. The Caddy block routes
`ww2explained.com` to the same backend that already serves
`karman.hackclub.app`. The reference block is in `Caddyfile` in this repo.

If you're unsure where your Nest `nest` CLI / Caddyfile live (this app runs in an
LXC container that doesn't have them), ask in the Hack Club `#nest` Slack — they
can point you at, or add, the gateway route for `ww2explained.com`.

### Step 3 — Verify

```bash
curl -I https://ww2explained.com      # expect HTTP 302 -> /login, server: gunicorn
```

Cert issuance can take a minute after DNS propagates.

---

## Other hosts (portable path)

The repo also ships `Dockerfile`, `gunicorn.conf.py` (binds `$PORT`, 180s
timeout), `run.sh`, and `Procfile`, so it runs unchanged on any container host
(Render/Fly/etc.) if you ever move off Nest. Those are the free-but-not-Nest
options; Nest is where it lives today.
