"""
Risk Scoring Module.

Combines evidence from all analysis modules into a transparent,
weighted risk score (0-100) and a final risk category.

SCORE WEIGHTING (documented and shown in the UI):
- ML text risk:      up to 60 points
- URL risk:          up to 25 points
- Sender/domain risk: up to 15 points

The final score is NOT a mathematical guarantee of fraud. It is a
transparent aggregation of available signals to help the user make
an informed decision.

Risk categories:
- LOW RISK (0-33):    Message appears low risk based on available evidence.
- MEDIUM RISK (34-66): Some suspicious indicators detected. Verify independently.
- HIGH RISK (67-100):  Multiple suspicious indicators detected. Avoid interaction.
"""

from typing import Optional


def compute_ml_risk(ml_result: dict) -> dict:
    """
    Convert ML prediction + confidence into a risk score (0-60).

    If the model predicts SPAM with high confidence, the risk is high.
    If the model predicts NOT SPAM with high confidence, the risk is low.
    If confidence is low, the risk is moderate regardless of prediction.

    Returns:
    {
        "score": int,           # 0-60
        "prediction": str,      # "SPAM" or "NOT SPAM"
        "confidence": float,    # 0.0-1.0
        "spam_probability": float,
        "explanation": str,
    }
    """
    label = ml_result.get("label", "UNKNOWN")
    confidence = ml_result.get("confidence", 0.0) or 0.0
    probabilities = ml_result.get("probabilities", {})

    spam_prob = probabilities.get("SPAM", 0.0) if probabilities else 0.0
    ham_prob = probabilities.get("NOT SPAM", 1.0 - spam_prob) if probabilities else (1.0 - spam_prob)

    # The spam probability directly drives the ML risk score.
    # spam_prob of 1.0 -> 60 points, spam_prob of 0.0 -> 0 points.
    score = int(spam_prob * 60)

    if label == "SPAM":
        explanation = (
            f"ML model predicted SPAM with {confidence*100:.1f}% confidence "
            f"(spam probability: {spam_prob*100:.1f}%)."
        )
    else:
        explanation = (
            f"ML model predicted NOT SPAM with {confidence*100:.1f}% confidence "
            f"(spam probability: {spam_prob*100:.1f}%)."
        )

    return {
        "score": score,
        "prediction": label,
        "confidence": confidence,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "explanation": explanation,
    }


def compute_url_risk(url_analyses: list) -> dict:
    """
    Convert URL analysis results into a risk score (0-25).

    If multiple URLs are present, take the maximum risk.
    Also collects all suspicious patterns across all URLs.

    Returns:
    {
        "score": int,           # 0-25
        "urls_found": int,
        "analyses": list,
        "all_patterns": list[str],
        "explanation": str,
    }
    """
    if not url_analyses:
        return {
            "score": 0,
            "urls_found": 0,
            "analyses": [],
            "all_patterns": [],
            "explanation": "No URLs detected in the message.",
        }

    # Scale each URL's 0-100 risk score to 0-25
    max_raw_risk = max(u["risk_score"] for u in url_analyses)
    score = int(max_raw_risk / 100 * 25)

    all_patterns = []
    for u in url_analyses:
        all_patterns.extend(u.get("suspicious_patterns", []))

    urls_found = len(url_analyses)
    if score == 0:
        explanation = f"{urls_found} URL(s) found — no suspicious URL patterns detected."
    else:
        explanation = (
            f"{urls_found} URL(s) found with suspicious indicators "
            f"(max URL risk: {max_raw_risk}/100, scaled to {score}/25)."
        )

    return {
        "score": score,
        "urls_found": urls_found,
        "analyses": url_analyses,
        "all_patterns": all_patterns,
        "explanation": explanation,
    }


def compute_sender_risk(sender_result: dict) -> dict:
    """
    Convert sender analysis into a risk score (0-15).

    Returns:
    {
        "score": int,          # 0-15
        "provided": bool,
        "type": str,
        "domain": str,
        "domain_status": str,
        "explanation": str,
    }
    """
    if not sender_result.get("provided", False):
        return {
            "score": 0,
            "provided": False,
            "type": "none",
            "domain": "",
            "domain_status": "n/a",
            "explanation": "No sender information provided — sender risk is neutral (0/15).",
        }

    score = sender_result.get("risk_contribution", 0)
    # Cap at 15
    score = min(score, 15)

    return {
        "score": score,
        "provided": True,
        "type": sender_result.get("type", "unknown"),
        "domain": sender_result.get("domain", ""),
        "domain_status": sender_result.get("domain_status", "unknown"),
        "verification_status": sender_result.get("verification_status", ""),
        "explanation": f"Sender risk contribution: {score}/15. {sender_result.get('verification_status', '')}",
    }


def compute_reputation_risk(reputation_result: dict) -> dict:
    """
    Factor in external reputation data if available.

    External reputation can add up to 10 bonus risk points on top of
    the base 100 (but the final score is capped at 100). If no API is
    configured, this adds 0 points.

    Returns:
    {
        "score": int,       # bonus points (0-10)
        "configured": bool,
        "result": str,
        "explanation": str,
    }
    """
    configured = reputation_result.get("configured", False)
    checked = reputation_result.get("checked", False)

    if not configured:
        return {
            "score": 0,
            "configured": False,
            "result": reputation_result.get("result", ""),
            "explanation": "External reputation API not configured — no additional risk from reputation data.",
        }

    if not checked:
        return {
            "score": 0,
            "configured": True,
            "result": reputation_result.get("result", ""),
            "explanation": "External reputation API configured but no check was performed.",
        }

    result_text = reputation_result.get("result", "")
    error = reputation_result.get("error")

    if error:
        return {
            "score": 0,
            "configured": True,
            "result": result_text,
            "explanation": f"External reputation check encountered an error: {error}",
        }

    # If the API flagged the URL, add bonus risk
    result_lower = result_text.lower()
    if any(word in result_lower for word in ["malicious", "flagged", "threat", "suspicious", "malware", "phishing"]):
        return {
            "score": 10,
            "configured": True,
            "result": result_text,
            "explanation": "External reputation API flagged this URL — +10 risk points.",
        }
    elif "no threats" in result_lower or "0 malicious" in result_lower or "clean" in result_lower:
        return {
            "score": 0,
            "configured": True,
            "result": result_text,
            "explanation": "External reputation API found no threats — no additional risk points.",
        }
    else:
        return {
            "score": 0,
            "configured": True,
            "result": result_text,
            "explanation": "External reputation API returned an inconclusive result.",
        }


def get_risk_category(score: int) -> dict:
    """
    Map a 0-100 risk score to a risk category.

    Returns:
    {
        "level": str,       # "LOW RISK" | "MEDIUM RISK" | "HIGH RISK"
        "description": str,
        "color": str,       # hex color for UI
    }
    """
    if score <= 33:
        return {
            "level": "LOW RISK",
            "description": "Message appears low risk based on the available evidence.",
            "color": "#27ae60",
        }
    elif score <= 66:
        return {
            "level": "MEDIUM RISK",
            "description": (
                "Some suspicious indicators were detected. Verify the sender "
                "independently before clicking links or providing information."
            ),
            "color": "#f39c12",
        }
    else:
        return {
            "level": "HIGH RISK",
            "description": (
                "Multiple suspicious indicators were detected. Avoid clicking "
                "links or sharing sensitive information until independently verified."
            ),
            "color": "#e74c3c",
        }


def compute_final_risk(
    ml_result: dict,
    url_analyses: list,
    sender_result: dict,
    reputation_result: Optional[dict] = None,
) -> dict:
    """
    Combine all evidence sources into a final risk assessment.

    Weighting:
        ML text risk:       up to 60 points
        URL risk:           up to 25 points
        Sender/domain risk: up to 15 points
        External reputation: up to 10 bonus points (capped at 100 total)

    Returns a comprehensive dict with all sub-scores and the final assessment.
    """
    ml_risk = compute_ml_risk(ml_result)
    url_risk = compute_url_risk(url_analyses)
    sender_risk = compute_sender_risk(sender_result)

    reputation_risk = {"score": 0, "configured": False, "result": "", "explanation": ""}
    if reputation_result:
        reputation_risk = compute_reputation_risk(reputation_result)

    base_score = ml_risk["score"] + url_risk["score"] + sender_risk["score"]
    total_score = min(base_score + reputation_risk["score"], 100)

    category = get_risk_category(total_score)

    # Build recommendation
    if total_score <= 33:
        recommendation = (
            "This message appears to be low risk based on available evidence. "
            "As always, use normal caution with any message. If something seems "
            "off, verify through official channels."
        )
    elif total_score <= 66:
        recommendation = (
            "Some suspicious indicators were detected. Do not rely solely on this "
            "assessment. Independently verify the sender through its official website "
            "or a known contact method before clicking any links or providing information."
        )
    else:
        recommendation = (
            "Multiple suspicious indicators were detected. Avoid clicking links, "
            "sharing sensitive information, or responding to this message. If you "
            "believe it may be important, contact the organization directly through "
            "an officially known channel — not through the contact details in this message."
        )

    return {
        "total_score": total_score,
        "category": category,
        "recommendation": recommendation,
        "breakdown": {
            "ml": ml_risk,
            "url": url_risk,
            "sender": sender_risk,
            "reputation": reputation_risk,
        },
        "weights": {
            "ml": "up to 60 points",
            "url": "up to 25 points",
            "sender": "up to 15 points",
            "reputation": "up to 10 bonus points (capped at 100)",
        },
    }
