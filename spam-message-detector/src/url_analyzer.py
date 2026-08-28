"""
URL Security Analysis Module.

This module inspects URL strings for suspicious patterns WITHOUT
visiting, downloading, or executing anything from the URL. It performs
purely static analysis on the URL string itself.

Checks performed:
- HTTPS vs HTTP
- IP-address-based URLs (e.g. http://192.168.1.1/login)
- URL shortening services (bit.ly, tinyurl.com, etc.)
- Excessively long URLs
- Suspicious characters (@, excessive hyphens/digits in domain)
- Unusual number of subdomains
- Punycode / IDN domains (xn-- prefix)
- Phishing-style patterns (login/verify/secure/bank keywords in path)
- Suspicious TLDs commonly associated with spam

The module returns a structured dict with a risk_score (0-100).
"""

import re
from urllib.parse import urlparse, unquote
import ipaddress
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "trusted_domains.json")

# Known URL shortening services (static list — no network calls)
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "shorte.st", "su.pr",
    "bitly.com", "tiny.cc", "shorturl.at", "rebrand.ly",
    "cutt.ly", "rb.gy", "short.link", "lnkd.in", "po.st",
    "soo.gd", "qr.ae", "v.gd", "shrtco.de", "lnk.to",
    "t.ly", "snip.ly", "clck.ru", "vk.cc",
}

# TLDs frequently seen in spam/phishing (not exhaustive, not definitive)
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click",
    "country", "stream", "download", "loan", "work",
    "men", "racing", "review", "party", "trade", "science",
    "date", "accountant", "cricket", "faith", "win",
    "bid", "kim", "rsvp",
}

# Phishing keywords often found in URL paths
PHISHING_PATH_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "confirm",
    "password", "banking", "wallet", "signin", "activate",
    "suspended", "unlock", "claim", "prize", "gift",
]


def load_trusted_domains(path: str = CONFIG_PATH) -> set:
    """Load the set of trusted domains from the config file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return set(d.lower() for d in data.get("trusted_domains", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def extract_urls_from_text(text: str) -> list:
    """
    Extract URLs from a block of text using regex.

    Matches:
    - http://... and https://...
    - www.example.com paths without scheme
    """
    if not text:
        return []

    # Match full URLs with scheme
    url_pattern = re.compile(
        r'https?://[^\s<>"\']+', re.IGNORECASE
    )
    urls = url_pattern.findall(text)

    # Match bare www.example.com (without scheme)
    www_pattern = re.compile(
        r'(?<!\w)www\.[a-z0-9\-]+\.[a-z]{2,}[^\s<>"\']*',
        re.IGNORECASE,
    )
    for m in www_pattern.findall(text):
        if not any(m in u for u in urls):
            urls.append("http://" + m)

    # Strip trailing punctuation that likely isn't part of the URL
    cleaned = []
    for u in urls:
        u = u.rstrip(".,;:!?)]}>'\"")
        cleaned.append(u)

    return cleaned


def _is_ip_address(host: str) -> bool:
    """Check if the host string is an IP address (v4 or v6)."""
    host = host.strip("[]")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _get_domain_parts(host: str) -> list:
    """Split a hostname into its parts (e.g. mail.google.com -> ['mail','google','com'])."""
    if not host:
        return []
    return host.split(".")


def _get_registrable_domain(host: str) -> str:
    """
    Get the main registrable domain from a hostname.

    This is a simplified heuristic: take the last two labels
    (e.g. google.com) unless the TLD is known to be multi-part.
    """
    parts = _get_domain_parts(host)
    if len(parts) < 2:
        return host
    return ".".join(parts[-2:])


def _count_subdomains(host: str) -> int:
    """Count the number of subdomain levels (excluding the registrable domain)."""
    parts = _get_domain_parts(host)
    if len(parts) <= 2:
        return 0
    return len(parts) - 2


def analyze_url(url: str, trusted_domains: set = None) -> dict:
    """
    Analyze a single URL string for suspicious patterns.

    Returns a structured dict:
    {
        "url": str,
        "url_found": bool,
        "scheme": str,
        "domain": str,
        "registrable_domain": str,
        "https": bool,
        "shortened_url": bool,
        "ip_based": bool,
        "punycode": bool,
        "suspicious_patterns": list[str],
        "risk_score": int (0-100),
        "domain_status": str,   # "recognized" | "unknown" | "suspicious"
    }
    """
    if trusted_domains is None:
        trusted_domains = load_trusted_domains()

    result = {
        "url": url,
        "url_found": True,
        "scheme": "",
        "domain": "",
        "registrable_domain": "",
        "https": False,
        "shortened_url": False,
        "ip_based": False,
        "punycode": False,
        "suspicious_patterns": [],
        "risk_score": 0,
        "domain_status": "unknown",
    }

    if not url or not url.strip():
        result["url_found"] = False
        return result

    url = url.strip()

    # Prepend a scheme if missing so urlparse works correctly
    if not url.lower().startswith(("http://", "https://")):
        if url.lower().startswith("www."):
            url = "http://" + url
        else:
            url = "http://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        result["suspicious_patterns"].append("URL could not be parsed")
        result["risk_score"] = 50
        return result

    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path or "").lower()
    query = unquote(parsed.query or "").lower()
    full_url_lower = url.lower()

    result["scheme"] = scheme
    result["domain"] = host
    result["https"] = scheme == "https"
    result["registrable_domain"] = _get_registrable_domain(host)

    risk = 0
    patterns = []

    # --- Check 1: HTTPS vs HTTP ---
    if not result["https"]:
        patterns.append("URL uses HTTP instead of HTTPS (not encrypted)")
        risk += 10

    # --- Check 2: IP-address-based URL ---
    if _is_ip_address(host):
        result["ip_based"] = True
        patterns.append("URL uses an IP address instead of a domain name")
        risk += 20

    # --- Check 3: URL shortening service ---
    if host in URL_SHORTENERS:
        result["shortened_url"] = True
        patterns.append(f"URL uses a shortening service ({host}) — destination is hidden")
        risk += 15

    # --- Check 4: Punycode / IDN domains ---
    if "xn--" in host:
        result["punycode"] = True
        patterns.append("Domain uses Punycode/IDN (internationalized domain) — can be used for homograph attacks")
        risk += 20

    # --- Check 5: Excessively long URL ---
    if len(url) > 100:
        patterns.append(f"URL is unusually long ({len(url)} characters)")
        risk += 10
    if len(url) > 200:
        patterns.append(f"URL is extremely long ({len(url)} characters) — may be trying to hide the real destination")
        risk += 10

    # --- Check 6: @ symbol in URL (can be used to obscure the real destination) ---
    if "@" in url.split("://")[-1].split("/")[0]:
        patterns.append("URL contains '@' in the authority section — can obscure the real destination")
        risk += 15

    # --- Check 7: Excessive subdomains ---
    subdomain_count = _count_subdomains(host)
    if subdomain_count >= 3:
        patterns.append(f"Unusually high number of subdomains ({subdomain_count})")
        risk += 10

    # --- Check 8: Excessive hyphens in domain ---
    if host.count("-") >= 3:
        patterns.append(f"Domain contains many hyphens ({host.count('-')}) — common in phishing domains")
        risk += 10

    # --- Check 9: Many digits in domain ---
    digit_ratio = sum(c.isdigit() for c in host.replace(".", "")) / max(len(host.replace(".", "")), 1)
    if digit_ratio > 0.3:
        patterns.append(f"Domain has a high proportion of digits ({digit_ratio:.0%}) — can be a sign of auto-generated domains")
        risk += 8

    # --- Check 10: Suspicious TLD ---
    tld = host.split(".")[-1] if "." in host else ""
    if tld in SUSPICIOUS_TLDS:
        patterns.append(f"Top-level domain '.{tld}' is frequently associated with spam")
        risk += 12

    # --- Check 11: Phishing keywords in path ---
    path_lower = path + query
    matched_keywords = [kw for kw in PHISHING_PATH_KEYWORDS if kw in path_lower]
    if matched_keywords:
        patterns.append(f"URL path contains sensitive keywords: {', '.join(matched_keywords)}")
        risk += 8

    # --- Check 12: Multiple redirects or port numbers ---
    if parsed.port and parsed.port not in (80, 443):
        patterns.append(f"URL uses a non-standard port ({parsed.port})")
        risk += 10

    # --- Check 13: Embedded credentials in URL ---
    if parsed.username or parsed.password:
        patterns.append("URL contains embedded credentials — extremely suspicious")
        risk += 20

    # --- Domain status check ---
    registrable = result["registrable_domain"]
    if registrable and registrable in trusted_domains:
        result["domain_status"] = "recognized"
        # Slightly reduce risk if the domain is recognized, but don't
        # zero it out — a trusted domain can still be spoofed
        risk = max(0, risk - 5)
    elif result["ip_based"]:
        result["domain_status"] = "suspicious"
    elif result["punycode"]:
        result["domain_status"] = "suspicious"
    else:
        result["domain_status"] = "unknown"

    # Cap at 100
    result["suspicious_patterns"] = patterns
    result["risk_score"] = min(risk, 100)

    return result


def analyze_urls(urls: list, trusted_domains: set = None) -> list:
    """Analyze a list of URLs and return a list of analysis dicts."""
    if not urls:
        return []
    return [analyze_url(u, trusted_domains) for u in urls]
