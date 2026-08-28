"""
AI-Powered Spam & Scam Message Risk Analyzer — Streamlit Application.

Run:  streamlit run app.py

This application combines:
1. ML text classification (existing Multinomial Naive Bayes + TF-IDF)
2. URL security analysis (static, no URL visits)
3. Sender/domain verification (trusted-domain config)
4. External reputation API (optional, via environment variable)
5. Transparent risk scoring (weighted evidence aggregation)
"""

import os
import sys

import streamlit as st

# Make the src package importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import SpamDetector  # noqa: E402
from src.url_analyzer import (  # noqa: E402
    extract_urls_from_text,
    analyze_url,
    load_trusted_domains,
)
from src.sender_analyzer import analyze_sender, load_config  # noqa: E402
from src.reputation_checker import check_reputation, is_configured, get_provider  # noqa: E402
from src.risk_scorer import compute_final_risk  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Spam & Scam Message Risk Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header { text-align: center; padding: 1rem 0 0.5rem 0; }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #856404;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .risk-score {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .info-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #1e8449; font-weight: 600; }
    .status-warn { color: #b9770e; font-weight: 600; }
    .status-bad  { color: #c0392b; font-weight: 600; }
    .status-info { color: #2471a3; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector():
    try:
        detector = SpamDetector()
        detector.load()
        return detector, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def get_trusted_domains():
    return load_trusted_domains()


@st.cache_data
def get_sender_config():
    return load_config()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### AI-Powered Spam & Scam\n### Message Risk Analyzer")
    st.markdown("---")

    st.markdown("#### Model Information")
    detector, load_error = load_detector()
    if detector and detector.loaded:
        st.success("ML Model: Loaded")
        st.caption(f"Classifier: {detector.model_name}")
        st.caption("Vectorizer: TF-IDF (unigrams + bigrams)")
    else:
        st.error("ML Model: Not loaded")
        if load_error:
            st.caption(load_error)

    st.markdown("---")
    st.markdown("#### External Reputation API")
    if is_configured():
        provider = get_provider()
        st.success(f"Configured: {provider}")
    else:
        st.info("Not configured")
        st.caption(
            "Set REPUTATION_API_KEY environment variable to enable "
            "external URL/domain reputation checks."
        )

    st.markdown("---")
    st.markdown("#### Tech Stack")
    st.markdown(
        """
        - Scikit-learn (ML)
        - NLTK (NLP)
        - TF-IDF + Naive Bayes
        - URL static analysis
        - Domain verification
        - Streamlit (UI)
        """
    )

    st.markdown("---")
    st.markdown("#### Risk Score Weights")
    st.markdown(
        """
        - ML text risk: **up to 60 pts**
        - URL risk: **up to 25 pts**
        - Sender risk: **up to 15 pts**
        - Reputation: **up to 10 bonus**
        """
    )


# ---------------------------------------------------------------------------
# Main content — Header + Disclaimer
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("AI-Powered Spam & Scam Message Risk Analyzer")
st.markdown(
    "Analyze a message for spam, scam, and security risk indicators using "
    "machine learning, URL analysis, sender verification, and optional "
    "external reputation data."
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">',
    unsafe_allow_html=True,
)
st.markdown(
    "**Disclaimer:** This tool provides an automated risk assessment based on "
    "available message, sender, URL, and reputation signals. It does not guarantee "
    "that a message is safe or fraudulent. Independently verify important messages "
    "through official channels."
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 1: Message Analysis — Input
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">1. Message Analysis</div>', unsafe_allow_html=True)

col_msg, col_ex = st.columns([3, 1])

with col_msg:
    message = st.text_area(
        "Message *",
        height=120,
        placeholder="Type or paste the message you want to analyze...",
        key="message_input",
    )

with col_ex:
    st.markdown("**Quick examples**")
    example = st.selectbox(
        "Choose an example",
        [
            "",
            "spam_prize",
            "normal_meeting",
            "spam_url_http",
            "normal_https",
            "spam_shortened",
            "spam_phishing",
            "normal_delivery",
        ],
        format_func=lambda x: {
            "": "Select...",
            "spam_prize": "Spam: Prize scam",
            "normal_meeting": "Normal: Meeting",
            "spam_url_http": "Spam: HTTP URL",
            "normal_https": "Normal: HTTPS link",
            "spam_shortened": "Spam: Shortened URL",
            "spam_phishing": "Spam: Phishing",
            "normal_delivery": "Normal: Delivery",
        }.get(x, x),
        label_visibility="collapsed",
    )

EXAMPLES = {
    "spam_prize": "Congratulations! You won a cash prize of $50,000. Click this link now to claim: http://192.168.1.5/claim?user=winner&id=99999",
    "normal_meeting": "Hey, are we meeting at 5 PM today? Let me know if the time works for you.",
    "spam_url_http": "URGENT: Your account has been suspended. Verify now at http://secure-bank-login.verify-account.tk/login?token=abc123 to avoid permanent closure.",
    "normal_https": "Hi, I found a great article you might enjoy: https://github.com/scikit-learn/scikit-learn — check it out when you have time.",
    "spam_shortened": "You have been selected for a free gift! Claim it here: http://bit.ly/free-gift-claim before it expires!",
    "spam_phishing": "Dear customer, your Amazon account has unusual activity. Please verify your identity at http://amaz0n-verify.account-update.secure-login.xyz/signin to prevent account lock.",
    "normal_delivery": "Hi mom, I arrived safely at the hostel. I'll call you later tonight. Love you!",
}

if example and not message:
    message = EXAMPLES.get(example, "")

# ---------------------------------------------------------------------------
# Section 2: Sender Verification — Input
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">2. Sender Verification</div>', unsafe_allow_html=True)

col_sender, col_url = st.columns(2)

with col_sender:
    sender = st.text_input(
        "Sender email or phone number (optional)",
        placeholder="e.g. support@google.com or +1-555-123-4567",
        key="sender_input",
    )

with col_url:
    manual_url = st.text_input(
        "URL to analyze (optional)",
        placeholder="e.g. https://example.com/page — leave blank to auto-extract from message",
        key="url_input",
    )

st.markdown("")
analyze_btn = st.button("Analyze Message", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis execution
# ---------------------------------------------------------------------------
if analyze_btn:
    # --- Validate input ---
    if not message or not message.strip():
        st.warning("Please enter a message to analyze.")
    elif detector is None or not detector.loaded:
        st.error(
            "The ML model could not be loaded. Please train the model first:\n\n"
            "`python -m src.train`"
        )
        if load_error:
            st.caption(f"Details: {load_error}")
    else:
        try:
            with st.spinner("Running full risk analysis..."):
                # --- Step 1: ML prediction ---
                ml_result = detector.predict(message)

                # --- Step 2: URL analysis ---
                urls = []
                if manual_url and manual_url.strip():
                    urls.append(manual_url.strip())
                # Auto-extract URLs from message if not manually provided
                # or if user wants additional URLs checked
                extracted = extract_urls_from_text(message)
                for u in extracted:
                    if u not in urls:
                        urls.append(u)

                trusted_domains = get_trusted_domains()
                url_analyses = [analyze_url(u, trusted_domains) for u in urls]

                # --- Step 3: Sender analysis ---
                sender_config = get_sender_config()
                sender_result = analyze_sender(sender, sender_config)

                # --- Step 4: External reputation ---
                reputation_result = None
                if urls and is_configured():
                    # Check the first URL (or the manual one if provided)
                    url_to_check = urls[0]
                    reputation_result = check_reputation(url_to_check)

                # --- Step 5: Combined risk score ---
                final_risk = compute_final_risk(
                    ml_result=ml_result,
                    url_analyses=url_analyses,
                    sender_result=sender_result,
                    reputation_result=reputation_result,
                )

            # ===================================================================
            # DISPLAY RESULTS
            # ===================================================================
            st.markdown("---")

            # --- Final Risk Level ---
            st.markdown('<div class="section-header">3. Risk Assessment</div>', unsafe_allow_html=True)

            category = final_risk["category"]
            total_score = final_risk["total_score"]

            bg_color_map = {
                "LOW RISK": "#e8f8e8",
                "MEDIUM RISK": "#fff4e0",
                "HIGH RISK": "#fde8e8",
            }
            border_color_map = {
                "LOW RISK": "#27ae60",
                "MEDIUM RISK": "#f39c12",
                "HIGH RISK": "#e74c3c",
            }

            bg = bg_color_map.get(category["level"], "#f8f9fa")
            border = border_color_map.get(category["level"], "#dee2e6")
            text_color = category["color"]

            st.markdown(
                f'<div class="risk-box" style="background-color:{bg}; border:2px solid {border};">',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="risk-label" style="color:{text_color};">{category["level"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="risk-score" style="color:{text_color};">{total_score}/100</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#555;">{category["description"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Score breakdown
            st.markdown("#### Score Breakdown")
            breakdown = final_risk["breakdown"]

            col_m, col_u, col_s, col_r = st.columns(4)
            with col_m:
                st.metric("ML Text Risk", f"{breakdown['ml']['score']}/60")
            with col_u:
                st.metric("URL Risk", f"{breakdown['url']['score']}/25")
            with col_s:
                st.metric("Sender Risk", f"{breakdown['sender']['score']}/15")
            with col_r:
                rep_score = breakdown['reputation']['score']
                st.metric("Reputation Bonus", f"+{rep_score}/10")

            st.progress(total_score / 100.0)

            with st.expander("How the score is calculated"):
                st.markdown(
                    """
                    **Risk Score Formula:**
                    ```
                    Total Score = ML Risk (0-60) + URL Risk (0-25) + Sender Risk (0-15) + Reputation Bonus (0-10)
                    ```
                    - **ML Risk (60 pts):** Based on the model's spam probability. If the model says 80% likely spam, that's 0.80 × 60 = 48 points.
                    - **URL Risk (25 pts):** Based on static URL analysis (HTTPS, shorteners, IP addresses, suspicious TLDs, etc.). Scaled from the 0-100 URL risk score.
                    - **Sender Risk (15 pts):** Based on domain verification status. Unknown domains add points; recognized domains add zero.
                    - **Reputation Bonus (10 pts):** Added if an external reputation API flags the URL. Only available when `REPUTATION_API_KEY` is configured.

                    **Categories:**
                    - 0–33: LOW RISK
                    - 34–66: MEDIUM RISK
                    - 67–100: HIGH RISK

                    This score is NOT a mathematical guarantee of fraud — it is a transparent aggregation of available signals.
                    """
                )

            # --- ML Analysis ---
            st.markdown('<div class="section-header">4. Machine Learning Analysis</div>', unsafe_allow_html=True)

            ml_label = ml_result["label"]
            ml_conf = ml_result["confidence"]
            ml_probs = ml_result.get("probabilities", {})

            if ml_label == "SPAM":
                st.markdown(
                    f'<span class="status-bad">Prediction: SPAM</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="status-good">Prediction: NOT SPAM</span>',
                    unsafe_allow_html=True,
                )

            if ml_conf is not None:
                st.markdown(f"**ML Confidence:** {ml_conf*100:.1f}%")

            if ml_probs:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    ham_p = ml_probs.get("NOT SPAM", 0.0)
                    st.metric("P(NOT SPAM)", f"{ham_p*100:.1f}%")
                    st.progress(ham_p)
                with col_p2:
                    spam_p = ml_probs.get("SPAM", 0.0)
                    st.metric("P(SPAM)", f"{spam_p*100:.1f}%")
                    st.progress(spam_p)

                st.caption(
                    "Confidence is the model's estimated probability, not a guarantee. "
                    "The model was trained on SMS data and may not generalize perfectly to all message types."
                )

            with st.expander("See preprocessed message"):
                st.text(ml_result["cleaned"])

            st.caption(f"Model: {detector.model_name} | Vectorizer: TF-IDF (unigrams + bigrams)")

            # --- Sender Analysis ---
            st.markdown('<div class="section-header">5. Sender Verification</div>', unsafe_allow_html=True)

            if sender_result["provided"]:
                st.markdown(f"**Sender:** `{sender_result['raw']}`")
                st.markdown(f"**Type:** {sender_result['type']}")

                if sender_result["domain"]:
                    st.markdown(f"**Extracted domain:** `{sender_result['domain']}`")

                status = sender_result["domain_status"]
                if status == "recognized":
                    st.markdown(
                        f'<span class="status-good">Domain Status: Recognized in trusted-domain configuration</span>',
                        unsafe_allow_html=True,
                    )
                elif status == "free_email":
                    st.markdown(
                        f'<span class="status-info">Domain Status: Free email provider</span>',
                        unsafe_allow_html=True,
                    )
                elif status == "unknown":
                    st.markdown(
                        f'<span class="status-warn">Domain Status: Not found in trusted-domain configuration</span>',
                        unsafe_allow_html=True,
                    )
                elif status == "suspicious":
                    st.markdown(
                        f'<span class="status-bad">Domain Status: Suspicious</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**Domain Status:** {status}")

                st.info(sender_result["verification_status"])

                for note in sender_result.get("notes", []):
                    st.caption(f"Note: {note}")
            else:
                st.info("No sender information provided. Sender identity could not be assessed.")

            st.caption(
                "Domain verification checks the sender's domain against a configuration file. "
                "It does NOT verify that the sender is the person they claim to be."
            )

            # --- URL Analysis ---
            st.markdown('<div class="section-header">6. URL Security Analysis</div>', unsafe_allow_html=True)

            if url_analyses:
                for i, ua in enumerate(url_analyses):
                    st.markdown(f"**URL {i+1}:** `{ua['url']}`")

                    col_https, col_domain, col_status = st.columns(3)
                    with col_https:
                        if ua["https"]:
                            st.markdown(
                                '<span class="status-good">HTTPS: Yes</span>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<span class="status-warn">HTTPS: No (HTTP)</span>',
                                unsafe_allow_html=True,
                            )
                    with col_domain:
                        st.markdown(f"**Domain:** `{ua['domain'] or 'N/A'}`")
                    with col_status:
                        ds = ua["domain_status"]
                        if ds == "recognized":
                            st.markdown(
                                '<span class="status-good">Recognized domain</span>',
                                unsafe_allow_html=True,
                            )
                        elif ds == "unknown":
                            st.markdown(
                                '<span class="status-warn">Unknown domain</span>',
                                unsafe_allow_html=True,
                            )
                        elif ds == "suspicious":
                            st.markdown(
                                '<span class="status-bad">Suspicious domain</span>',
                                unsafe_allow_html=True,
                            )

                    # Indicators
                    indicators = []
                    if ua["ip_based"]:
                        indicators.append("IP-based URL")
                    if ua["shortened_url"]:
                        indicators.append("Shortened URL")
                    if ua["punycode"]:
                        indicators.append("Punycode/IDN domain")
                    if indicators:
                        st.markdown(f"**Indicators:** {', '.join(indicators)}")

                    if ua["suspicious_patterns"]:
                        st.markdown("**Suspicious patterns detected:**")
                        for p in ua["suspicious_patterns"]:
                            st.markdown(f"- {p}")
                    else:
                        st.markdown(
                            '<span class="status-good">No suspicious URL patterns detected.</span>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**URL Risk Score:** {ua['risk_score']}/100")
                    st.markdown("---")
            else:
                st.info("No URLs detected in the message or URL field.")

            st.caption(
                "URL analysis is purely static — the application does not visit, "
                "download, or execute anything from any URL."
            )

            # --- External Verification ---
            st.markdown('<div class="section-header">7. External Verification</div>', unsafe_allow_html=True)

            rep = final_risk["breakdown"]["reputation"]
            if rep["configured"]:
                st.markdown(f"**Provider:** {get_provider()}")
                st.markdown(f"**Result:** {rep['result']}")
                if rep.get("explanation"):
                    st.caption(rep["explanation"])
            else:
                st.info("External reputation verification is not configured.")
                st.caption(
                    "To enable, set the REPUTATION_API_KEY environment variable "
                    "with a key from Google Safe Browsing or VirusTotal."
                )

            # --- Recommendation ---
            st.markdown('<div class="section-header">8. Safety Recommendation</div>', unsafe_allow_html=True)

            rec_bg = bg_color_map.get(category["level"], "#f8f9fa")
            rec_border = border_color_map.get(category["level"], "#dee2e6")

            st.markdown(
                f'<div class="info-card" style="background-color:{rec_bg}; border:2px solid {rec_border};">',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{category['level']}**")
            st.markdown(final_risk["recommendation"])
            st.markdown("</div>", unsafe_allow_html=True)

        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            import traceback
            st.expander("Error details").code(traceback.format_exc())

# ---------------------------------------------------------------------------
# How It Works
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)

col_h1, col_h2, col_h3 = st.columns(3)

with col_h1:
    st.markdown(
        """
        **1. Text Preprocessing & ML**
        
        The message is cleaned (lowercased, punctuation removed, stopwords filtered)
        and converted to numerical features using **TF-IDF**. A trained
        **Multinomial Naive Bayes** classifier predicts SPAM or NOT SPAM with
        a confidence score.
        """
    )

with col_h2:
    st.markdown(
        """
        **2. URL & Sender Analysis**
        
        URLs are extracted from the message and inspected for suspicious patterns
        (HTTP, IP addresses, shorteners, suspicious TLDs, phishing keywords).
        Sender domains are checked against a trusted-domain configuration.
        No URLs are visited.
        """
    )

with col_h3:
    st.markdown(
        """
        **3. Risk Score Aggregation**
        
        All evidence is combined into a transparent 0-100 risk score:
        ML (60 pts) + URL (25 pts) + Sender (15 pts) + Reputation bonus (10 pts).
        The score maps to LOW, MEDIUM, or HIGH RISK.
        """
    )

st.markdown(
    """
    <div class="disclaimer">
    <strong>Reminder:</strong> This tool provides an automated risk assessment based on available
    signals. It does not guarantee that a message is safe or fraudulent. Always independently
    verify important messages through official channels.
    </div>
    """,
    unsafe_allow_html=True,
)
