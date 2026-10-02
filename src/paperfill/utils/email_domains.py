"""
Whether an address may open an account: real syntax, a domain that can receive
mail, and not a throwaway inbox.

Verification email alone doesn't hold the line. Temp-mail services hand out a
working inbox for sixty seconds, so a link sent to one gets clicked and the
account is "verified" while the address it belongs to no longer exists. That
costs us the free daily credits on every fresh burner, and leaves accounts we
can never reach again.

The blocklist is a snapshot of the community disposable-email-domains list,
refreshed by tools/refresh_disposable_domains.py. It goes stale as services
register new domains, which is why the MX check sits behind it rather than
instead of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import dns.exception
import dns.resolver

_LIST_PATH = Path(__file__).with_name("disposable_domains.txt")

# Deliberately loose. The address only has to be plausible enough to be worth a
# DNS lookup; whether it exists is settled by the verification email, not by a
# regex trying to reimplement RFC 5322.
_ADDRESS = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)

# A blank DNS answer must not read as "no mail server here", so a lookup that
# times out or finds no nameserver lets the signup through. The verification
# email is still unsendable to a domain that genuinely can't receive it, so
# failing open costs us a dead row, while failing closed would lock out real
# users every time our resolver hiccuped.
_DNS_TIMEOUT = 4.0


def _load_blocklist() -> frozenset[str]:
    with _LIST_PATH.open() as f:
        return frozenset(
            line.strip().lower() for line in f
            if line.strip() and not line.startswith("#")
        )


DISPOSABLE_DOMAINS = _load_blocklist()


def is_disposable(domain: str) -> bool:
    """True if the domain, or any domain it sits under, is a known throwaway.

    Walking the parents is what catches the wildcard services: mailinator hands
    out anything.mailinator.com, and only the registered domain is on the list.
    """
    labels = domain.lower().strip(".").split(".")
    return any(
        ".".join(labels[i:]) in DISPOSABLE_DOMAINS
        for i in range(len(labels) - 1)
    )


def _accepts_mail(domain: str) -> bool:
    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = _DNS_TIMEOUT
        try:
            return bool(resolver.resolve(domain, "MX"))
        except dns.resolver.NoAnswer:
            # No MX is still deliverable: RFC 5321 falls back to the address
            # record. Rare, but rejecting it would turn a small self-hosted
            # domain into an unexplainable signup failure.
            return bool(resolver.resolve(domain, "A"))
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return False
    except (dns.exception.DNSException, OSError) as e:
        print(f"[auth] MX lookup for {domain} failed open: {e}")
        return True


def rejection_reason(email: str) -> str | None:
    """The message to show the user, or None if this address may sign up.

    Ordered cheapest first: a regex, then a set lookup, then the one DNS round
    trip, so a junk address never costs a network call.
    """
    email = (email or "").strip().lower()
    if not _ADDRESS.match(email):
        return "Enter a valid email address."
    domain = email.rsplit("@", 1)[1]
    if is_disposable(domain):
        return ("Temporary and disposable email addresses aren't accepted. "
                "Please sign up with a permanent address.")
    if not _accepts_mail(domain):
        return (f"We can't send email to {domain}. Check the address for a "
                "typo and try again.")
    return None
