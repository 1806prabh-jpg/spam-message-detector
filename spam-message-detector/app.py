"""
Streamlit web application for the Spam Message Detector.

Run:  streamlit run app.py
"""

import os
import sys

import streamlit as st

# Make the src package importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import SpamDetector  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .spam-box {
        background-color: #fde8e8;
        border: 2px solid #e74c3c;
    }
    .ham-box {
        background-color: #e8f8e8;
        border: 2px solid #27ae60;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .spam-label {
        color: #c0392b;
    }
    .ham-label {
        color: #1e8449;
    }
    .confidence-bar {
        margin-top: 0.5rem;
    }
    .info-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load model (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector():
    try:
        detector = SpamDetector()
        detector.load()
        return detector, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar — project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About This Project")
    st.markdown(
        """
        This app uses **Natural Language Processing (NLP)** and **Machine Learning**
        to classify text messages as **SPAM** or **NOT SPAM**.

        **How it works:**
        1. Your message is cleaned (lowercased, punctuation removed, stopwords filtered)
        2. The cleaned text is converted to numbers using **TF-IDF**
        3. A trained classifier predicts the category
        4. The result and confidence are displayed
        """
    )

    st.markdown("---")
    st.markdown("### Model Information")
    detector, load_error = load_detector()
    if detector and detector.loaded:
        st.markdown(f"**Model:** {detector.model_name}")
        st.markdown("**Vectorizer:** TF-IDF (unigrams + bigrams)")
        st.markdown("**Status:** Ready")
    else:
        st.error("Model not loaded")
        if load_error:
            st.caption(load_error)

    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown(
        """
        - Python & Scikit-learn
        - NLTK (NLP preprocessing)
        - TF-IDF Vectorization
        - Streamlit (web UI)
        """
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Spam Message Detector")
st.markdown(
    "Enter a text message below and click **Analyze** to find out "
    "whether it is spam or a legitimate message."
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Input section ---
st.markdown("### Enter a Message")
message = st.text_area(
    "Message text",
    height=150,
    placeholder="Type or paste a message here...",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
with col2:
    example = st.selectbox(
        "Try an example",
        [
            "",
            "Congratulations! You won a cash prize. Click this link now!",
            "Hey, are we meeting at 5 PM?",
            "URGENT: Your mobile number has won $5000. Reply WIN to claim.",
            "Can you pick up some milk on your way home?",
            "FREE entry in a weekly competition to win FA Cup tickets!",
            "I'll call you back in 10 minutes.",
        ],
        format_func=lambda x: "Choose an example..." if x == "" else x[:60] + ("..." if len(x) > 60 else ""),
    )

# Use the example if one is selected and the text area is empty
if example and not message:
    message = example

# --- Prediction ---
if analyze_btn:
    if not message or not message.strip():
        st.warning("Please enter a message to analyze.")
    elif detector is None or not detector.loaded:
        st.error(
            "The model could not be loaded. Please train the model first by running:\n\n"
            "`python -m src.train`"
        )
        if load_error:
            st.caption(f"Details: {load_error}")
    else:
        try:
            with st.spinner("Analyzing message..."):
                result = detector.predict(message)

            label = result["label"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            # Display result box
            if label == "SPAM":
                st.markdown(
                    f'<div class="result-box spam-box">'
                    f'<p class="result-label spam-label">SPAM</p>'
                    f'<p style="color:#7b241c; margin:0.5rem 0 0 0;">'
                    f'This message looks like spam.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-box ham-box">'
                    f'<p class="result-label ham-label">NOT SPAM</p>'
                    f'<p style="color:#196f3d; margin:0.5rem 0 0 0;">'
                    f'This message appears to be legitimate.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Confidence display
            if probabilities:
                st.markdown("### Confidence")
                spam_prob = probabilities.get("SPAM", 0.0)
                ham_prob = probabilities.get("NOT SPAM", 0.0)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("NOT SPAM", f"{ham_prob*100:.1f}%")
                    st.progress(ham_prob)
                with col_b:
                    st.metric("SPAM", f"{spam_prob*100:.1f}%")
                    st.progress(spam_prob)

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not a guarantee of correctness."
                )

            # Show preprocessed text in an expander
            with st.expander("See preprocessed message"):
                st.text(result["cleaned"])

        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

# --- How it works section ---
st.markdown("---")
st.markdown("### How the System Works")
st.markdown(
    """
    <div class="info-card">
    <strong>NLP Pipeline:</strong><br>
    Raw Message &rarr; Text Preprocessing &rarr; TF-IDF Vectorization &rarr; Machine Learning Model &rarr; Prediction
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    **Text Preprocessing** cleans the raw message by:
    - Converting to lowercase (so "FREE" and "free" are treated the same)
    - Removing punctuation and special characters
    - Removing stopwords (common words like "the", "is", "at" that don't help classification)
    - Normalizing whitespace

    **TF-IDF** (Term Frequency-Inverse Document Frequency) converts the cleaned
    text into numerical features. It weighs words by how frequently they appear
    in a message versus how common they are across all messages. Words that are
    frequent in spam but rare in ham (like "prize", "claim", "urgent") get high
    TF-IDF values, making them strong signals for the classifier.

    **The classifier** was trained on thousands of labeled SMS messages and
    learned which TF-IDF patterns correspond to spam versus legitimate messages.
    """
)

st.markdown("---")
st.caption("Spam Message Detector — A portfolio NLP & ML project.")
