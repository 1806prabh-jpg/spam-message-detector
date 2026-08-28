"""
Sender Analysis Module.

Analyzes optional sender information (email address or phone number)
provided by the user. This module performs:

- Email format validation and domain extraction
- Domain comparison against the trusted-domain configuration
- Free email provider detection
- Phone number basic format validation
- Clear statements about what can and cannot be verified

IMPORTANT: This module does NOT claim to verify that a sender is a
real person or that a phone number belongs to anyone. It only checks
the structure of the provided information and compares domains against
a configuration file. Independent verification of identity requires
an external service that is not part of this project.
"""

import re
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "trusted_domains.json")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load the trusted-domain and free-email-domain configuration."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"trusted_domains": [], "free_email_domains": []}


# Basic email regex (RFC 5322 simplified — good enough for validation, not exhaustive)
EMAIL_REGEX = re.compile(
    r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
)

# Basic phone number regex: accepts +, spaces, dashes, parentheses, 7-15 digits
PHONE_REGEX = re.compile(
    r'^[+]?[\s\-()]*(?:\d[\s\-()]*){7,15}$'
)


def is_email(text: str) -> bool:
    """Check if the text looks like an email address."""
    if not text:
        return False
    return bool(EMAIL_REGEX.match(text.strip()))


def is_phone_number(text: str) -> bool:
    """Check if the text looks like a phone number."""
    if not text:
        return False
    text = text.strip()
    # Must contain at least 7 digits to be a plausible phone number
    digit_count = sum(c.isdigit() for c in text)
    if digit_count < 7 or digit_count > 15:
        return False
    return bool(PHONE_REGEX.match(text))


def extract_email_domain(email: str) -> str:
    """Extract the domain part from an email address."""
    if "@" not in email:
        return ""
    return email.split("@")[-1].lower().strip()


def analyze_sender(sender: str, config: dict = None) -> dict:
    """
    Analyze sender information.

    Returns a structured dict:
    {
        "provided": bool,
        "type": "email" | "phone" | "unknown",
        "raw": str,
        "domain": str,            # extracted domain (email only)
        "domain_status": str,     # "recognized" | "unknown" | "suspicious" | "free_email" | "n/a"
        "free_email": bool,
        "format_valid": bool,
        "verification_status": str,  # human-readable status
        "risk_contribution": int,    # 0-15 risk points to add to total
        "notes": list[str],
    }
    """
    if config is None:
        config = load_config()

    result = {
        "provided": False,
        "type": "unknown",
        "raw": "",
        "domain": "",
        "domain_status": "n/a",
        "free_email": False,
        "format_valid": False,
        "verification_status": "",
        "risk_contribution": 0,
        "notes": [],
    }

    if not sender or not sender.strip():
        result["verification_status"] = "No sender information provided."
        return result

    sender = sender.strip()
    result["provided"] = True
    result["raw"] = sender

    trusted = set(d.lower() for d in config.get("trusted_domains", []))
    free_domains = set(d.lower() for d in config.get("free_email_domains", []))

    # --- Email analysis ---
    if is_email(sender):
        result["type"] = "email"
        result["format_valid"] = True
        domain = extract_email_domain(sender)
        result["domain"] = domain

        if domain in trusted:
            result["domain_status"] = "recognized"
            result["verification_status"] = (
                f"Domain '{domain}' is recognized in the trusted-domain configuration. "
                f"Note: this confirms the domain is known, not that this specific sender is verified."
            )
            result["risk_contribution"] = 0
        elif domain in free_domains:
            result["domain_status"] = "free_email"
            result["free_email"] = True
            result["verification_status"] = (
                f"Domain '{domain}' is a free email provider. Free email accounts are not "
                f"inherently suspicious but are commonly used in spam. "
                f"Sender identity could not be independently verified."
            )
            result["risk_contribution"] = 3
        else:
            result["domain_status"] = "unknown"
            result["verification_status"] = (
                f"Domain '{domain}' was not found in the trusted-domain configuration. "
                f"Sender identity could not be independently verified."
            )
            result["risk_contribution"] = 5

    # --- Phone number analysis ---
    elif is_phone_number(sender):
        result["type"] = "phone"
        result["format_valid"] = True
        result["domain_status"] = "n/a"
        result["verification_status"] = (
            f"The input '{sender}' appears to be a phone number. "
            f"Phone number identity cannot be independently verified without an "
            f"appropriate external service. Do not assume the number belongs to "
            f"the claimed person or organization."
        )
        result["risk_contribution"] = 5
        result["notes"].append(
            "Phone number ownership verification requires an external telecom/HLR lookup service."
        )

    # --- Unknown format ---
    else:
        result["type"] = "unknown"
        result["format_valid"] = False
        result["verification_status"] = (
            f"The input '{sender}' is not a recognized email address or phone number format. "
            f"Sender identity could not be independently verified."
        )
        result["risk_contribution"] = 8
        result["notes"].append(
            "Input did not match known email or phone number patterns."
        )

    return result
