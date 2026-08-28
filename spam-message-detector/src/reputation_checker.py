"""
External Reputation Checker Module.

This module provides an architecture for integrating a real URL/domain
reputation API (such as Google Safe Browsing, VirusTotal, or URLVoid).

DESIGN PRINCIPLES:
- API keys are read from environment variables, NEVER from source code.
- If no API key is configured, the module reports that external
  verification is not configured — it does NOT fabricate results.
- If an API is configured and returns a result, the real result is
  returned and clearly labeled as external reputation data.
- The module never visits, downloads, or executes anything from the URL.

ENVIRONMENT VARIABLES:
- REPUTATION_API_KEY: API key for the reputation service
- REPUTATION_API_PROVIDER: which provider to use ("google_safe_browsing" | "virustotal")
  (defaults to "google_safe_browsing")

HOW TO ENABLE:
1. Obtain an API key from a reputation service provider.
2. Set the environment variable:
   export REPUTATION_API_KEY="your-api-key-here"
3. Optionally set the provider:
   export REPUTATION_API_PROVIDER="google_safe_browsing"
4. Restart the Streamlit application.

Without these environment variables, the module will report that
external verification is not configured.
"""

import os
import json
import urllib.request
import urllib.error


def is_configured() -> bool:
    """Check whether a reputation API key is configured."""
    return bool(os.environ.get("REPUTATION_API_KEY"))


def get_provider() -> str:
    """Get the configured reputation API provider name."""
    return os.environ.get("REPUTATION_API_PROVIDER", "google_safe_browsing")


def check_reputation(url: str) -> dict:
    """
    Check URL/domain reputation via an external API.

    If no API key is configured, returns a dict indicating that
    external verification is not available. No fabricated results.

    If an API key IS configured, this function makes a real HTTP
    request to the configured provider's API and returns the result.

    Returns:
    {
        "configured": bool,
        "provider": str,
        "checked": bool,       # whether an actual API call was made
        "result": str,         # human-readable result
        "details": dict,       # raw API response (if available)
        "error": str | None,   # error message if the call failed
    }
    """
    api_key = os.environ.get("REPUTATION_API_KEY")
    provider = get_provider()

    result = {
        "configured": False,
        "provider": provider,
        "checked": False,
        "result": "",
        "details": {},
        "error": None,
    }

    if not api_key:
        result["configured"] = False
        result["result"] = "External reputation verification is not configured."
        return result

    result["configured"] = True

    if not url or not url.strip():
        result["result"] = "No URL provided for reputation check."
        return result

    # --- Google Safe Browsing API v4 ---
    if provider == "google_safe_browsing":
        return _check_google_safe_browsing(url, api_key)

    # --- VirusTotal API v3 ---
    elif provider == "virustotal":
        return _check_virustotal(url, api_key)

    else:
        result["error"] = f"Unknown reputation API provider: {provider}"
        result["result"] = f"Configured provider '{provider}' is not supported."
        return result


def _check_google_safe_browsing(url: str, api_key: str) -> dict:
    """
    Query the Google Safe Browsing API v4.

    API docs: https://developers.google.com/safe-browsing/v4/lookup-api
    """
    result = {
        "configured": True,
        "provider": "google_safe_browsing",
        "checked": True,
        "result": "",
        "details": {},
        "error": None,
    }

    api_url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"

    payload = json.dumps({
        "client": {
            "clientId": "spam-scam-risk-analyzer",
            "clientVersion": "1.0",
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        result["details"] = data

        matches = data.get("matches", [])
        if matches:
            threat_types = [m.get("threatType", "UNKNOWN") for m in matches]
            result["result"] = (
                f"Google Safe Browsing flagged this URL as: {', '.join(threat_types)}. "
                f"This is external reputation data from Google Safe Browsing."
            )
        else:
            result["result"] = (
                "Google Safe Browsing found no threats for this URL. "
                "This is external reputation data from Google Safe Browsing. "
                "Note: a clean result does not guarantee the URL is safe."
            )

    except urllib.error.HTTPError as e:
        result["error"] = f"API returned HTTP {e.code}: {e.reason}"
        result["result"] = f"External reputation check failed (HTTP {e.code})."
    except urllib.error.URLError as e:
        result["error"] = f"Network error: {e.reason}"
        result["result"] = "External reputation check failed (network error)."
    except Exception as e:
        result["error"] = str(e)
        result["result"] = f"External reputation check failed: {e}"

    return result


def _check_virustotal(url: str, api_key: str) -> dict:
    """
    Query the VirusTotal API v3 for URL analysis.

    API docs: https://developers.virustotal.com/reference/url-info
    """
    import base64

    result = {
        "configured": True,
        "provider": "virustotal",
        "checked": True,
        "result": "",
        "details": {},
        "error": None,
    }

    # VirusTotal v3 requires base64url-encoded URLs
    url_id = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").rstrip("=")
    api_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"

    try:
        req = urllib.request.Request(
            api_url,
            headers={"x-apikey": api_key},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        result["details"] = data

        attrs = data.get("data", {}).get("attributes", {})
        last_analysis = attrs.get("last_analysis_stats", {})
        malicious = last_analysis.get("malicious", 0)
        suspicious = last_analysis.get("suspicious", 0)
        harmless = last_analysis.get("harmless", 0)

        if malicious > 0 or suspicious > 0:
            result["result"] = (
                f"VirusTotal reports {malicious} malicious and {suspicious} suspicious "
                f"detections out of {malicious + suspicious + harmless} engines. "
                f"This is external reputation data from VirusTotal."
            )
        else:
            result["result"] = (
                f"VirusTotal reports 0 malicious detections from "
                f"{malicious + suspicious + harmless} engines. "
                f"This is external reputation data from VirusTotal. "
                f"Note: a clean result does not guarantee the URL is safe."
            )

    except urllib.error.HTTPError as e:
        result["error"] = f"API returned HTTP {e.code}: {e.reason}"
        result["result"] = f"External reputation check failed (HTTP {e.code})."
    except urllib.error.URLError as e:
        result["error"] = f"Network error: {e.reason}"
        result["result"] = "External reputation check failed (network error)."
    except Exception as e:
        result["error"] = str(e)
        result["result"] = f"External reputation check failed: {e}"

    return result
