# AI-Powered Spam & Scam Message Risk Analyzer

## Overview

The **AI-Powered Spam & Scam Message Risk Analyzer** is an advanced NLP and machine learning application that goes beyond simple spam/ham classification. It analyzes a message's text content, any embedded URLs, optional sender information, and (if configured) external reputation data to produce a transparent, evidence-based risk assessment with a 0-100 risk score and a LOW / MEDIUM / HIGH risk category.

The project builds on a trained **Multinomial Naive Bayes + TF-IDF** model (trained on the SMS Spam Collection dataset) and adds URL security analysis, sender/domain verification, an optional external reputation API architecture, and a weighted risk scoring system — all wrapped in a professional Streamlit interface.

This project is designed as a B.Tech CSE (AI/ML) placement portfolio piece demonstrating practical skills in NLP, machine learning, security analysis, software architecture, and deployment.

---

## Problem Statement

Spam and scam messages are not just annoyances — they are vectors for phishing, financial fraud, and identity theft. A simple "SPAM or NOT SPAM" binary classification is insufficient for real-world decision-making because:

1. **A message can be spam-like without being dangerous** (e.g., marketing emails)
2. **A message can be dangerous without looking like spam** (e.g., a well-crafted phishing attempt)
3. **URLs and sender identity** carry critical risk signals that text-only models ignore
4. **Users need actionable guidance**, not just a label — they need to know *why* a message is risky and *what to do*

This project addresses these gaps by combining multiple analysis layers into a transparent risk assessment that helps users make informed decisions while never claiming absolute certainty.

---

## Features

### Core ML Features (preserved from original project)
- **Multinomial Naive Bayes** classifier trained on 5,572 real SMS messages
- **TF-IDF vectorization** with unigrams and bigrams
- **NLP preprocessing pipeline** (lowercasing, punctuation removal, stopword filtering)
- **Model evaluation** with accuracy, precision, recall, F1-score, and confusion matrix

### New Advanced Features
- **URL security analysis** — 13+ static checks (HTTPS, IP-based, shorteners, Punycode, suspicious TLDs, phishing keywords, etc.)
- **Sender/domain verification** — email format validation, domain extraction, trusted-domain comparison, free email detection
- **Phone number validation** — basic format check with clear caveats about identity verification
- **External reputation API architecture** — pluggable support for Google Safe Browsing and VirusTotal via environment variables
- **Transparent risk scoring** — weighted evidence aggregation (ML 60 + URL 25 + Sender 15 + Reputation 10)
- **Three-tier risk categories** — LOW, MEDIUM, HIGH with actionable recommendations
- **Professional Streamlit UI** — 8 clearly labeled sections with visual indicators
- **Auto URL extraction** — URLs are automatically detected in the message text
- **No fabricated results** — the app never claims verification without evidence

---

## Architecture

```
                         ┌─────────────────────┐
                         │   User Input         │
                         │  (Message + Sender   │
                         │   + Optional URL)    │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │  NLP + ML    │ │ URL Analyzer │ │   Sender     │
            │  Pipeline    │ │ (Static)     │ │  Analyzer    │
            │              │ │              │ │              │
            │ Preprocess   │ │ HTTPS check  │ │ Email format │
            │ TF-IDF       │ │ IP-based     │ │ Domain extract│
            │ Naive Bayes  │ │ Shorteners   │ │ Trusted check │
            │ Predict+Proba│ │ Phishing kw  │ │ Phone format  │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                   │                │                │
                   │         ┌──────▼───────┐        │
                   │         │ Reputation   │        │
                   │         │ Checker      │        │
                   │         │ (Optional)   │        │
                   │         │ Env var API  │        │
                   │         └──────┬───────┘        │
                   │                │                │
                   └────────┬───────┴────────────────┘
                            ▼
                   ┌────────────────┐
                   │  Risk Scorer   │
                   │                │
                   │ ML: 0-60       │
                   │ URL: 0-25      │
                   │ Sender: 0-15   │
                   │ Reputation: +10│
                   │                │
                   │ Total: 0-100   │
                   └───────┬────────┘
                           ▼
                   ┌────────────────┐
                   │  Final Result  │
                   │                │
                   │ LOW / MED / HIGH│
                   │ Score: 0-100   │
                   │ Recommendation │
                   └────────────────┘
```

---

## NLP Pipeline

```
Raw Message
    ↓
Text Preprocessing
    (lowercase → remove special characters → normalize whitespace
     → tokenize → remove stopwords → rejoin)
    ↓
TF-IDF Vectorization
    (convert text to numerical feature vectors using fitted vectorizer)
    ↓
Machine Learning Model
    (Multinomial Naive Bayes — trained on 5,572 SMS messages)
    ↓
Prediction + Confidence
    (SPAM / NOT SPAM + probability scores)
```

### Preprocessing Steps

| Step | What it does | Why |
|------|-------------|-----|
| Lowercasing | Converts all text to lowercase | "FREE" and "free" should be treated as the same word |
| Remove special characters | Strips punctuation, symbols, emojis | Punctuation rarely helps classification and adds noise |
| Whitespace normalization | Collapses multiple spaces, trims ends | Ensures clean token boundaries |
| Tokenization | Splits text into individual words | Needed for stopword removal and vectorization |
| Stopword removal | Drops common words ("the", "is", "at") | These words appear in both spam and ham equally and don't help distinguish them |

---

## TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts text into numerical features. It has two components:

- **TF (Term Frequency):** How often a word appears in a specific message
- **IDF (Inverse Document Frequency):** How rare the word is across all messages

The product TF × IDF gives a score that is high for words that are frequent in a specific message but rare across the corpus. This makes spam-associated words like "prize", "claim", "urgent", and "winner" stand out strongly.

### Why TF-IDF is preferable to raw word counts in this project:
1. **Down-weights common words** — "the", "is", "and" appear in both spam and ham; raw counts would give them high weight
2. **Up-weights distinctive words** — words unique to spam or ham get higher scores
3. **Better generalization** — TF-IDF features produce more robust models than raw counts
4. **Sublinear scaling** — we use `1 + log(tf)` to prevent a single repeated word from dominating

### Data leakage prevention:
The TF-IDF vectorizer is fitted **only on the training data** inside a scikit-learn Pipeline. The same fitted vectorizer is then applied to the test set and to new messages at inference time. This ensures no information from the test set leaks into training.

---

## Multinomial Naive Bayes

The selected classifier is **Multinomial Naive Bayes**, a probabilistic model based on Bayes' theorem. It is particularly well-suited for text classification because:

- It works naturally with discrete features like word counts or TF-IDF values
- It is fast to train and predict
- It performs well even with limited training data
- It provides probability estimates that can be used as confidence scores

**Parameters used:** `alpha=0.1` (Laplace smoothing — small value because the vocabulary is large and most words appear in the training data)

The model was compared against **Logistic Regression** (`class_weight="balanced"`, `solver="liblinear"`) and selected based on the highest F1-score.

---

## Sender Verification

When the user provides sender information, the system performs:

### For email addresses:
1. **Format validation** — checks the email matches a valid pattern
2. **Domain extraction** — pulls the domain from the email (e.g. `user@google.com` → `google.com`)
3. **Trusted-domain comparison** — checks the domain against `config/trusted_domains.json`
4. **Free email detection** — identifies known free email providers (Gmail, Yahoo, etc.)

### For phone numbers:
1. **Format validation** — checks for a plausible phone number structure (7-15 digits)
2. **Clear caveat** — displays that phone identity cannot be independently verified without an external telecom service

### Domain verification states:
| State | Meaning |
|-------|---------|
| **Recognized** | Domain found in trusted-domain configuration |
| **Unknown** | Domain not found in the configuration |
| **Suspicious** | Domain uses IP address or Punycode |
| **Free email** | Domain is a known free email provider |
| **Unable to verify** | Sender format not recognized |

**IMPORTANT:** Recognizing a domain does NOT verify that the sender is who they claim to be. Email addresses can be spoofed. The system explicitly states this in the UI.

---

## URL Security Analysis

The URL analyzer performs **purely static analysis** on URL strings. It never visits, downloads, or executes anything from any URL.

### Checks performed (13+ patterns):

| Check | Risk Added | Description |
|-------|-----------|-------------|
| HTTP (not HTTPS) | +10 | URL is not encrypted |
| IP-based URL | +20 | Uses an IP address instead of a domain name |
| URL shortener | +15 | Destination is hidden (bit.ly, tinyurl.com, etc.) |
| Punycode/IDN | +20 | Internationalized domain — can be used for homograph attacks |
| Long URL (>100 chars) | +10 | May be trying to hide the real destination |
| Very long URL (>200 chars) | +10 | Extremely long — suspicious |
| @ in authority | +15 | Can obscure the real destination |
| Excessive subdomains (≥3) | +10 | Unusual subdomain depth |
| Many hyphens in domain (≥3) | +10 | Common in phishing domains |
| High digit ratio (>30%) | +8 | May be auto-generated |
| Suspicious TLD | +12 | TLDs frequently associated with spam (.tk, .xyz, .click, etc.) |
| Phishing keywords in path | +8 | login/verify/secure/account/claim/prize/etc. |
| Non-standard port | +10 | URL uses a port other than 80 or 443 |
| Embedded credentials | +20 | URL contains username:password@ — extremely suspicious |

---

## Risk Scoring

The final risk score (0-100) is a **transparent, weighted aggregation** of all evidence:

```
Total Score = ML Risk (0-60) + URL Risk (0-25) + Sender Risk (0-15) + Reputation Bonus (0-10)
```

| Component | Max Points | How it's calculated |
|-----------|-----------|---------------------|
| **ML text risk** | 60 | Spam probability × 60 (e.g., 80% spam = 48 pts) |
| **URL risk** | 25 | Max URL risk score (0-100) scaled to 25 |
| **Sender/domain risk** | 15 | Based on domain status (0 for recognized, 3-8 for unknown) |
| **Reputation bonus** | +10 | Added if external API flags the URL (capped at 100 total) |

### Risk Categories:
| Score | Category | Description |
|-------|----------|-------------|
| 0-33 | **LOW RISK** | Message appears low risk based on available evidence |
| 34-66 | **MEDIUM RISK** | Some suspicious indicators detected. Verify independently |
| 67-100 | **HIGH RISK** | Multiple suspicious indicators detected. Avoid interaction |

**The score is NOT a mathematical guarantee of fraud.** It is a transparent aggregation of available signals to help users make informed decisions.

---

## External Reputation API Architecture

The project is designed to support real URL/domain reputation APIs via environment variables. No API keys are stored in source code.

### Supported providers:
1. **Google Safe Browsing API v4** — checks URLs against Google's threat database
2. **VirusTotal API v3** — checks URLs against 70+ antivirus engines

### Configuration:
```bash
# Set the API key
export REPUTATION_API_KEY="your-api-key-here"

# Set the provider (optional, defaults to google_safe_browsing)
export REPUTATION_API_PROVIDER="google_safe_browsing"  # or "virustotal"
```

### Behavior:
- **No API key configured:** The UI shows "External reputation verification is not configured." No fabricated results.
- **API key configured + URL present:** A real HTTP request is made to the provider's API. The real result is displayed and clearly labeled as external reputation data.
- **API error:** The error is displayed transparently. No fallback to fabricated results.

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Primary programming language |
| Scikit-learn | ML model (Naive Bayes), TF-IDF, Pipeline, evaluation |
| NLTK | Stopword list for NLP preprocessing |
| Pandas / NumPy | Data loading and numerical operations |
| Matplotlib | Confusion matrix visualization |
| Streamlit | Web interface |
| urllib (stdlib) | External reputation API calls |
| json (stdlib) | Configuration file parsing |
| ipaddress (stdlib) | IP address detection in URLs |

---

## Machine Learning Models

Two classifiers were trained and compared on the same test set:

| Model | Accuracy | Precision | Recall | F1-score | Confusion Matrix |
|-------|----------|-----------|--------|----------|-----------------|
| Multinomial Naive Bayes | 0.9874 | 1.0000 | 0.9060 | **0.9507** | TN=964, FP=0, FN=14, TP=135 |
| Logistic Regression | 0.9704 | 0.8671 | 0.9195 | 0.8925 | TN=943, FP=21, FN=12, TP=137 |

**Selected model:** Multinomial Naive Bayes (highest F1-score: 0.9507)

- Naive Bayes achieved **perfect precision** (zero false positives — no legitimate message was wrongly flagged as spam)
- Logistic Regression achieved higher recall (caught more spam) but with 21 false positives
- F1-score was chosen as the selection metric because it balances precision and recall, both of which matter for spam detection

**Dataset:** SMS Spam Collection (UCI ML Repository) — 5,565 messages after preprocessing (747 spam, 4,818 ham), 80/20 stratified split.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)

### Step 1: Clone and navigate
```bash
cd spam-message-detector
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify the dataset
The dataset should be at `data/spam.csv`. If missing, run:
```bash
python -c "
import urllib.request, zipfile
urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip', 'data/smsspamcollection.zip')
with zipfile.ZipFile('data/smsspamcollection.zip') as z:
    z.extractall('data/')
import pandas as pd
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label','message'])
df.to_csv('data/spam.csv', index=False)
print('Dataset ready:', df.shape)
"
```

### Step 4: Train the model (if not already trained)
```bash
python -m src.train
```

### Step 5: (Optional) Configure external reputation API
```bash
export REPUTATION_API_KEY="your-api-key"
export REPUTATION_API_PROVIDER="google_safe_browsing"  # or "virustotal"
```

---

## How to Run

```bash
streamlit run app.py
```

The app will open in your browser. Enter a message, optionally add sender info and a URL, then click **Analyze Message**.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REPUTATION_API_KEY` | No | API key for external reputation service. If not set, external verification is disabled. |
| `REPUTATION_API_PROVIDER` | No | Which provider to use: `google_safe_browsing` (default) or `virustotal` |

No other environment variables are required. The ML model, dataset, and all analysis modules work without any external configuration.

---

## Project Structure

```
spam-message-detector/
│
├── config/
│   └── trusted_domains.json     # Trusted domain + free email configuration
│
├── data/
│   └── spam.csv                 # SMS Spam Collection dataset
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # NLP text preprocessing pipeline
│   ├── train.py                 # Model training & evaluation
│   ├── predict.py               # ML prediction utilities
│   ├── url_analyzer.py          # Static URL security analysis
│   ├── sender_analyzer.py       # Sender email/phone verification
│   ├── reputation_checker.py    # External reputation API architecture
│   └── risk_scorer.py           # Weighted risk score aggregation
│
├── models/
│   ├── spam_classifier.pkl      # Trained ML pipeline (Naive Bayes + TF-IDF)
│   ├── metrics.txt              # Actual evaluation results
│   ├── confusion_matrix.png     # Best model confusion matrix
│   ├── confusion_matrix_nb.png  # Naive Bayes confusion matrix
│   └── confusion_matrix_lr.png  # Logistic Regression confusion matrix
│
├── app.py                       # Streamlit web interface
├── requirements.txt
├── README.md
└── screenshots/                 # Add screenshots here
```

---

## Limitations

1. **ML model trained on SMS data** — the model may not generalize perfectly to email, social media, or other message formats
2. **No real-time URL visiting** — URL analysis is purely static; it cannot detect whether a page contains malware at runtime
3. **Trusted-domain list is manually maintained** — the configuration file contains a small example list and is not comprehensive
4. **Phone number verification is not possible** — without an external telecom/HLR lookup service, phone ownership cannot be verified
5. **Email spoofing is not detected** — the system checks the domain string but cannot verify email headers or SPF/DKIM records
6. **External reputation API is optional** — without configuration, no external URL reputation data is available
7. **Risk score is heuristic** — the weighting (60/25/15/10) is a reasonable default but not empirically optimized
8. **No multilingual support** — the model and preprocessing are English-only

---

## Future Improvements

1. **Deep learning models** — Train an LSTM or BERT model for better context understanding
2. **Real-time URL scanning** — Integrate with Google Safe Browsing API (architecture is ready, just needs an API key)
3. **Email header analysis** — Parse SPF, DKIM, and DMARC records for email authentication verification
4. **Larger dataset** — Incorporate email spam datasets (Enron, SpamAssassin) for broader coverage
5. **Multilingual support** — Extend to detect spam in multiple languages
6. **Model explainability** — Show which words/features contributed most to the ML prediction
7. **Active learning** — Let users flag incorrect predictions to retrain the model
8. **Batch processing** — Allow uploading a CSV of messages for bulk analysis
9. **Risk score optimization** — Use empirical data to optimize the weighting of evidence components
10. **Deployment** — Deploy as a REST API using FastAPI or on Streamlit Community Cloud

---

## Screenshots

Add screenshots of the Streamlit app here:

1. `screenshots/main_interface.png` — Main input interface
2. `screenshots/low_risk_result.png` — Low risk analysis result
3. `screenshots/high_risk_result.png` — High risk analysis result
4. `screenshots/url_analysis.png` — URL security analysis section
5. `screenshots/sender_verification.png` — Sender verification section

---

## Interview Questions

### 1. What is the NLP pipeline in this project?
The NLP pipeline processes raw text through: lowercasing → removing special characters → whitespace normalization → tokenization → stopword removal → rejoining. The cleaned text is then converted to numerical features using TF-IDF and fed to a Multinomial Naive Bayes classifier. The same pipeline is applied during training and inference via a scikit-learn Pipeline object to ensure consistency and prevent data leakage.

### 2. What is TF-IDF and why is it better than raw word counts?
TF-IDF (Term Frequency-Inverse Document Frequency) weighs words by how frequently they appear in a specific message versus how common they are across all messages. Raw word counts treat all words equally, so common words like "the" and "is" would dominate. TF-IDF down-weights common words and up-words distinctive words, making spam-associated terms like "prize" and "claim" stand out. We also use sublinear TF (1 + log(tf)) to prevent a single repeated word from dominating.

### 3. Why is the TF-IDF vectorizer fitted only on training data?
If the vectorizer sees the test data during fitting, it learns the vocabulary and IDF weights from the test set — this is called data leakage. The model would perform artificially well because it has already "seen" the test data's word distribution. By fitting only on training data inside a Pipeline, we ensure the test set simulates truly unseen messages, giving an honest estimate of real-world performance.

### 4. How does the risk scoring system work?
The final risk score (0-100) combines four evidence sources with transparent weights: ML text risk (up to 60 points, based on spam probability), URL risk (up to 25 points, based on static URL analysis), sender/domain risk (up to 15 points, based on domain verification), and reputation bonus (up to 10 points, from external API if configured). The score maps to LOW (0-33), MEDIUM (34-66), or HIGH (67-100) risk categories. The weighting is documented and shown in the UI — it is not a black box.

### 5. Why not just use the ML model's prediction?
The ML model only analyzes text patterns. It cannot detect whether a URL uses an IP address, whether a domain is spoofed, or whether the sender's email domain is recognized. A well-crafted phishing message might not trigger the spam model but could have a suspicious URL. Combining multiple evidence sources provides a more realistic and useful risk assessment than text classification alone.

### 6. What does the URL analyzer check and what does it NOT do?
The URL analyzer performs 13+ static checks: HTTPS vs HTTP, IP-based URLs, shortening services, Punycode/IDN domains, URL length, @ symbols, subdomain count, hyphens, digit ratio, suspicious TLDs, phishing keywords in paths, non-standard ports, and embedded credentials. It does NOT visit, download, or execute anything from the URL. It purely inspects the URL string. This is a safety principle — the app never interacts with potentially dangerous URLs.

### 7. How does sender/domain verification work and what are its limits?
The system extracts the domain from the sender's email address and checks it against a configuration file (`config/trusted_domains.json`). If the domain is in the list, it's "recognized." If not, it's "unknown." For phone numbers, the system validates the format but cannot verify ownership. The key limitation is that email addresses can be spoofed — recognizing a domain doesn't prove the sender controls that email account. The UI explicitly states this.

### 8. How is the external reputation API designed?
The `reputation_checker.py` module reads the API key from the `REPUTATION_API_KEY` environment variable — never from source code. It supports Google Safe Browsing and VirusTotal. If no key is configured, it reports "not configured" and does not fabricate results. If a key is present, it makes a real HTTP request to the provider's API and returns the actual response, clearly labeled as external data. This architecture allows the app to work without any external dependency while remaining ready for production use.

### 9. Why does the app never claim "100% safe" or "verified"?
Because no single automated system can guarantee safety. The ML model can be wrong, URLs can be newly created, email addresses can be spoofed, and reputation databases have false negatives. Claiming certainty would be misleading and potentially dangerous. Instead, the app uses probabilistic language ("likely," "appears," "unable to verify") and always recommends independent verification for important messages.

### 10. How would you improve this for production?
I would: (1) enable the external reputation API with a real Google Safe Browsing key, (2) add email header analysis (SPF/DKIM/DMARC), (3) train on a larger and more diverse dataset including email, (4) use a transformer model like BERT for better text understanding, (5) implement active learning from user feedback, (6) add real-time monitoring and alerting, (7) optimize the risk score weights using empirical data, (8) deploy as a microservice API with rate limiting, and (9) add multi-language support.
