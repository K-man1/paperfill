"""
Re-download the disposable-email blocklist that gates signup.

    python tools/refresh_disposable_domains.py

Temp-mail services register new domains constantly, so the bundled snapshot is
only as good as the last time someone ran this. Commit the result; nothing
fetches it at runtime.

The write is guarded: a truncated or hijacked response that dropped the list to
a handful of entries, or that swept in a real provider, would either stop
blocking burners or lock out everyone on gmail. Both fail loudly here instead.
"""

import sys
import urllib.request
from pathlib import Path

SOURCE = ("https://raw.githubusercontent.com/disposable-email-domains/"
          "disposable-email-domains/main/disposable_email_blocklist.conf")

DEST = (Path(__file__).resolve().parents[1]
        / "src" / "paperfill" / "utils" / "disposable_domains.txt")

# Providers our users actually sign up with. If the upstream list ever names
# one, we don't take the update.
MUST_NOT_BLOCK = ["gmail.com", "googlemail.com", "outlook.com", "hotmail.com",
                  "yahoo.com", "icloud.com", "me.com", "proton.me",
                  "protonmail.com", "aol.com", "live.com", "gmx.com",
                  "zoho.com", "fastmail.com", "hackclub.com"]

MIN_DOMAINS = 5000


def main() -> int:
    with urllib.request.urlopen(SOURCE, timeout=30) as resp:
        body = resp.read().decode()

    domains = sorted({ln.strip().lower() for ln in body.splitlines()
                      if ln.strip() and not ln.startswith("#")})

    if len(domains) < MIN_DOMAINS:
        print(f"refusing to write: only {len(domains)} domains, expected "
              f"at least {MIN_DOMAINS}")
        return 1
    caught = sorted(set(domains) & set(MUST_NOT_BLOCK))
    if caught:
        print(f"refusing to write: upstream list blocks {', '.join(caught)}")
        return 1

    previous = len(DEST.read_text().splitlines()) if DEST.exists() else 0
    DEST.write_text("\n".join(domains) + "\n")
    print(f"wrote {len(domains)} domains to {DEST.relative_to(Path.cwd())} "
          f"({len(domains) - previous:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
