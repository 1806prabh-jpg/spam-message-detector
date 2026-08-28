# AI-Powered Spam & Scam Message Risk Analyzer

A machine learning and NLP-powered application that analyzes text messages, URLs, and sender information to produce a transparent, evidence-based risk assessment — going beyond simple spam/ham classification to help users make informed decisions about potentially fraudulent messages.

---

## Live Demo

The application is deployed and runnable via Streamlit:

```
YOUR_STREAMLIT_APP_URL
```

> Replace the placeholder above with your actual Streamlit Cloud (or self-hosted) URL.

---

## Project Overview

This project started as a spam/ham SMS classifier built with TF-IDF and Multinomial Naive Bayes, trained on the publicly available SMS Spam Collection dataset (5,572 messages). It has been extended into a multi-signal risk analyzer that combines:

1. **ML text classification** — the original Naive Bayes model predicts SPAM or NOT SPAM and outputs a confidence probability.
2. **URL security analysis** — a static URL inspector checks for 13+ suspicious structural patterns without ever visiting the URL.
3. **Sender/domain verification** — email addresses and phone numbers are validated and checked against a configurable trusted-domain list.
4. **External reputation API** (optional) — a pluggable architecture for Google Safe Browsing or VirusTotal, activated via environment variables.
5. **Transparent risk scoring** — all evidence is aggregated into a 0–100 heuristic risk score with a LOW / MEDIUM / HIGH category and an actionable recommendation.

The application is designed for a B.Tech CSE (AI/ML) placement portfolio and demonstrates practical skills in NLP, feature engineering, model evaluation, security analysis, software architecture, and deployment.

> **Disclaimer:** This application provides an automated risk assessment based on available message, sender, URL, and reputation signals. It does not guarantee that a message is safe or fraudulent. Always independently verify important messages through official channels.

---

## Key Features

| Feature | Description |
|---------|-------------|
| ML classification | Multinomial Naive Bayes with TF-IDF (unigrams + bigrams) trained on 5,572 SMS messages |
| NLP preprocessing | Lowercasing, punctuation removal, stopword filtering, tokenization, whitespace normalization |
| URL auto-extraction | URLs are automatically detected in the message text; manual URL entry also supported |
| Static URL analysis | 13+ checks: HTTPS, IP-based, shorteners, Punycode, suspicious TLDs, phishing keywords, and more — no URLs are visited |
| Sender verification | Email format validation, domain extraction, trusted-domain comparison, free email detection, phone number format validation |
| External reputation API | Optional integration with Google Safe Browsing or VirusTotal via environment variables — no fabricated results |
| Risk scoring | Transparent weighted score (0–100) with LOW / MEDIUM / HIGH categories and recommendations |
| Streamlit UI | Professional, multi-section interface with visual indicators, score breakdown, and disclaimer |

---

## How the System Works

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
            │ TF-IDF       │ │ IP-based     │ │ Domain check │
            │ Naive Bayes  │ │ Shorteners   │ │ Phone format │
            │ Predict+Proba│ │ Phishing kw  │ │ Trusted list │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
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
                   │  ML: 0-60      │
                   │  URL: 0-25     │
                   │  Sender: 0-15  │
                   │  Reputation:   │
                   │  up to +10     │
                   │  Total: 0-100  │
                   └───────┬────────┘
                           ▼
                   ┌────────────────┐
                   │  Final Result  │
                   │  LOW / MED /   │
                   │  HIGH RISK     │
                   │  + Score       │
                   │  + Recommend.  │
                   └────────────────┘
```

**Step-by-step:**

1. The user enters a message (required), optionally a sender email/phone, and optionally a URL.
2. The message is preprocessed (lowercase, remove punctuation, remove stopwords) and passed through the trained TF-IDF + Naive Bayes pipeline to get a SPAM/NOT SPAM prediction with probability.
3. URLs are extracted from the message text (and/or taken from the manual URL field) and analyzed statically for suspicious patterns.
4. Sender information is validated and the domain is checked against the trusted-domain configuration.
5. If an external reputation API key is configured, the URL is submitted to the configured provider (Google Safe Browsing or VirusTotal).
6. All evidence is combined into a transparent 0–100 risk score and a LOW / MEDIUM / HIGH category with a recommendation.

---

## Machine Learning Model

### Model

**Multinomial Naive Bayes** — a probabilistic classifier based on Bayes' theorem, well-suited for text classification with discrete features like TF-IDF values. It was selected over Logistic Regression based on a higher F1-score on the test set.

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features by weighing each word by how frequently it appears in a specific message versus how common it is across all messages. Words frequent in spam but rare in ham (e.g., "prize", "claim", "urgent") receive high TF-IDF values, making them strong signals for the classifier.

**Configuration used:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ngram_range` | (1, 2) | Unigrams + bigrams — captures phrases like "free entry" |
| `max_features` | 5,000 | Keep top 5,000 terms by frequency |
| `min_df` | 2 | Ignore terms appearing in fewer than 2 documents |
| `max_df` | 0.95 | Ignore terms in >95% of documents (corpus-specific) |
| `sublinear_tf` | True | Use 1 + log(tf) instead of raw term frequency |

The TF-IDF vectorizer is fitted **only on the training data** inside a scikit-learn Pipeline to prevent data leakage from the test set.

### Preprocessing Pipeline

| Step | What it does | Why |
|------|-------------|-----|
| Lowercasing | Converts all text to lowercase | "FREE" and "free" are treated the same |
| Remove special characters | Strips punctuation and symbols | Punctuation adds noise without aiding classification |
| Whitespace normalization | Collapses multiple spaces, trims ends | Ensures clean token boundaries |
| Tokenization | Splits text into words | Needed for stopword removal and vectorization |
| Stopword removal | Drops common words ("the", "is", "at") | These appear in both spam and ham and don't help distinguish them |

### Dataset

**Source:** [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — UCI Machine Learning Repository

| Statistic | Value |
|-----------|-------|
| Total messages (original) | 5,572 |
| After preprocessing | 5,565 (7 removed for being empty after cleaning) |
| Spam | 747 |
| Ham (not spam) | 4,818 |
| Train/test split | 80/20 stratified (4,452 train / 1,113 test) |

---

## Model Performance

The following metrics are the evaluation results of the **original SMS text classification model** on the held-out test set. They describe how well the Naive Bayes classifier identifies spam versus ham in SMS messages — **not** the accuracy of the complete scam-risk scoring system, which is a heuristic aggregation of multiple signals.

| Model | Accuracy | Precision | Recall | F1-Score | Confusion Matrix |
|-------|----------|-----------|--------|----------|-----------------|
| **Multinomial Naive Bayes** (selected) | **98.74%** | **100.00%** | **90.60%** | **95.07%** | TN=964, FP=0, FN=14, TP=135 |
| Logistic Regression | 97.04% | 86.71% | 91.95% | 89.25% | TN=943, FP=21, FN=12, TP=137 |

**Why Naive Bayes was selected:** It achieved the highest F1-score (95.07%), which balances precision and recall — both critical for spam detection. Notably, it achieved **perfect precision** (zero false positives), meaning no legitimate message was wrongly flagged as spam.

**Why precision and recall matter for spam detection:**
- **Precision** = Of all messages flagged as spam, how many were actually spam? High precision means few false alarms.
- **Recall** = Of all actual spam messages, how many were caught? High recall means few spam messages slip through.
- **F1-score** = Harmonic mean of precision and recall, balancing both concerns.

> These results are stored in `models/metrics.txt` and were generated by running `python -m src.train`. They are not manually edited.

---

## URL Security Analysis

The URL analyzer (`src/url_analyzer.py`) performs **purely static analysis** on URL strings. It **never visits, downloads, or executes anything from any URL**. All checks are based on parsing the URL string itself.

### Checks Performed

| # | Check | Risk Added | Description |
|---|-------|-----------|-------------|
| 1 | HTTP (not HTTPS) | +10 | URL is not encrypted |
| 2 | IP-based URL | +20 | Uses an IP address instead of a domain name |
| 3 | URL shortening service | +15 | Destination is hidden (bit.ly, tinyurl.com, etc. — 28 services recognized) |
| 4 | Punycode / IDN domain | +20 | Internationalized domain — can be used for homograph attacks |
| 5 | Long URL (>100 chars) | +10 | May be trying to hide the real destination |
| 6 | Very long URL (>200 chars) | +10 | Extremely long — highly suspicious |
| 7 | @ symbol in authority | +15 | Can obscure the real destination |
| 8 | Excessive subdomains (≥3) | +10 | Unusual subdomain depth |
| 9 | Many hyphens in domain (≥3) | +10 | Common in phishing domains |
| 10 | High digit ratio (>30%) | +8 | May be auto-generated |
| 11 | Suspicious TLD | +12 | TLDs frequently associated with spam (.tk, .xyz, .click, etc. — 25 entries) |
| 12 | Phishing keywords in path | +8 | login/verify/secure/account/claim/prize/etc. (16 keywords) |
| 13 | Non-standard port | +10 | URL uses a port other than 80 or 443 |
| 14 | Embedded credentials | +20 | URL contains username:password@ — extremely suspicious |

### Domain Status

The URL's registrable domain is also checked against the trusted-domain configuration:

| Status | Meaning |
|--------|---------|
| **Recognized** | Domain found in `config/trusted_domains.json` (risk slightly reduced) |
| **Unknown** | Domain not found in the configuration |
| **Suspicious** | Domain uses an IP address or Punycode |

---

## Sender Verification

The sender analyzer (`src/sender_analyzer.py`) validates optional sender information provided by the user.

### For Email Addresses

1. **Format validation** — checks the email matches a valid pattern
2. **Domain extraction** — pulls the domain from the email (e.g., `user@google.com` → `google.com`)
3. **Trusted-domain comparison** — checks the domain against `config/trusted_domains.json` (20 example domains)
4. **Free email detection** — identifies known free email providers (Gmail, Yahoo, Outlook, etc. — 10 providers)

### For Phone Numbers

1. **Format validation** — checks for a plausible phone number structure (7–15 digits, allows +, spaces, dashes, parentheses)
2. **Ownership caveat** — the application explicitly states that phone number identity **cannot be independently verified** without an appropriate external telecom/HLR lookup service

### Domain Verification States

| State | Meaning | Risk Contribution |
|-------|---------|-------------------|
| **Recognized** | Domain found in trusted-domain configuration | 0 points |
| **Free email** | Domain is a known free email provider | 3 points |
| **Unknown** | Domain not found in the configuration | 5 points |
| **Phone** | Input is a phone number (ownership unverifiable) | 5 points |
| **Unknown format** | Input doesn't match email or phone patterns | 8 points |
| **Not provided** | No sender information given | 0 points (neutral) |

> **Important limitation:** Recognizing a domain does **not** prove that the sender is who they claim to be. Email addresses can be spoofed. The system checks the domain string only — it does not inspect email headers or SPF/DKIM/DMARC records.

---

## Risk Scoring System

The final risk score (0–100) is a **heuristic, transparent aggregation** of all available evidence. It is **NOT a calibrated probability of being a scam** — it is a weighted sum of independent risk signals designed to help the user make an informed decision.

### Score Formula

```
Total Score = ML Risk (0–60) + URL Risk (0–25) + Sender Risk (0–15) + Reputation Bonus (0–10)
```

| Component | Max Points | How It's Calculated |
|-----------|-----------|---------------------|
| **ML text risk** | 60 | Spam probability from Naive Bayes × 60 (e.g., 80% spam → 48 points) |
| **URL risk** | 25 | Maximum URL risk score (0–100) scaled to 25 |
| **Sender/domain risk** | 15 | Based on domain status: 0 for recognized, 3 for free email, 5 for unknown, 8 for unrecognized format |
| **Reputation bonus** | +10 | Added if external reputation API flags the URL (total capped at 100) |

### Risk Categories

| Score Range | Category | Description |
|------------|----------|-------------|
| 0–33 | **LOW RISK** | Message appears low risk based on the available evidence. |
| 34–66 | **MEDIUM RISK** | Some suspicious indicators were detected. Verify the sender independently before clicking links or providing information. |
| 67–100 | **HIGH RISK** | Multiple suspicious indicators were detected. Avoid clicking links or sharing sensitive information until independently verified. |

> The score weighting (60/25/15/10) is a reasonable heuristic default. It is **not** empirically optimized or calibrated against a ground-truth scam dataset. The full breakdown of each component is displayed in the application UI so the user can see exactly how the score was computed.

---

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Primary programming language |
| Scikit-learn | ML model (Naive Bayes), TF-IDF, Pipeline, evaluation metrics |
| NLTK | English stopword list for NLP preprocessing |
| Pandas / NumPy | Data loading and numerical operations |
| Matplotlib | Confusion matrix visualization |
| Streamlit | Web interface |
| `urllib` (stdlib) | External reputation API HTTP requests |
| `json` (stdlib) | Configuration file parsing |
| `ipaddress` (stdlib) | IP address detection in URL analysis |
| `re` (stdlib) | Regex for URL extraction, email/phone validation |

---

## Project Structure

```
spam-message-detector/
│
├── config/
│   └── trusted_domains.json         # Trusted domain + free email configuration
│
├── data/
│   └── spam.csv                     # SMS Spam Collection dataset (5,572 messages)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                # NLP text preprocessing pipeline
│   ├── train.py                     # Model training & evaluation script
│   ├── predict.py                   # ML prediction utilities (load model, predict)
│   ├── url_analyzer.py              # Static URL security analysis (13+ checks)
│   ├── sender_analyzer.py           # Sender email/phone verification
│   ├── reputation_checker.py        # External reputation API (Google Safe Browsing / VirusTotal)
│   └── risk_scorer.py               # Weighted risk score aggregation (0–100)
│
├── models/
│   ├── spam_classifier.pkl          # Trained ML pipeline (Naive Bayes + TF-IDF)
│   ├── metrics.txt                  # Actual evaluation results
│   ├── confusion_matrix.png         # Best model confusion matrix
│   ├── confusion_matrix_nb.png      # Naive Bayes confusion matrix
│   └── confusion_matrix_lr.png      # Logistic Regression confusion matrix
│
├── app.py                           # Streamlit web interface
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Step 1: Navigate to the project directory

```bash
cd spam-message-detector
```

### Step 2: Create a virtual environment (recommended)

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify the dataset exists

The dataset should be at `data/spam.csv`. If it is missing, download and convert it:

```bash
python -c "
import urllib.request, zipfile, pandas as pd
urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip', 'data/smsspamcollection.zip')
with zipfile.ZipFile('data/smsspamcollection.zip') as z:
    z.extractall('data/')
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label','message'])
df.to_csv('data/spam.csv', index=False)
print('Dataset ready:', df.shape)
"
```

### Step 5: Train the model (if not already trained)

```bash
python -m src.train
```

This loads the dataset, preprocesses text, splits 80/20 stratified, trains both Naive Bayes and Logistic Regression, evaluates both, and saves the best model to `models/spam_classifier.pkl`.

---

## How to Run Locally

### Start the Streamlit application

```bash
streamlit run app.py
```

The app will open in your default browser. Enter a message (required), optionally add a sender email/phone and a URL, then click **Analyze Message**.

### Quick example test messages

| Type | Example Message |
|------|----------------|
| Obvious spam | "Congratulations! You won a cash prize of $50,000. Click this link now!" |
| Normal | "Hey, are we meeting at 5 PM today?" |
| Spam with URL | "URGENT: Your account has been suspended. Verify at http://192.168.1.5/claim" |
| Normal with URL | "Hi, I found a great article: https://github.com/scikit-learn/scikit-learn" |
| Shortened URL | "You have been selected for a free gift! Claim here: http://bit.ly/free-gift" |

### Stopping the application

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## External Reputation API

The project includes a pluggable architecture for real URL/domain reputation checks via external APIs. API keys are read from **environment variables only** — they are never stored in source code.

### Supported Providers

| Provider | API Version | Env Var Value |
|----------|-------------|---------------|
| Google Safe Browsing | v4 | `google_safe_browsing` |
| VirusTotal | v3 | `virustotal` |

### Configuration

**macOS / Linux:**
```bash
export REPUTATION_API_KEY="your-api-key-here"
export REPUTATION_API_PROVIDER="google_safe_browsing"  # or "virustotal"
```

**Windows (PowerShell):**
```powershell
$env:REPUTATION_API_KEY = "your-api-key-here"
$env:REPUTATION_API_PROVIDER = "google_safe_browsing"
```

### Behavior

- **No API key configured:** The UI displays "External reputation verification is not configured." No results are fabricated.
- **API key configured + URL present:** A real HTTP request is made to the provider's API. The actual response is displayed and clearly labeled as external reputation data.
- **API error:** The error is displayed transparently. No fallback to fabricated results.

> Without an API key, the application works fully — only the external reputation section reports "not configured."

---

## Limitations

| Limitation | Details |
|-----------|---------|
| **ML model trained on SMS data** | The model may not generalize perfectly to email, social media, or other message formats |
| **URL analysis is static** | The application inspects URL strings only — it does not visit, download, or execute anything from URLs. It cannot detect real-time malware on a page. |
| **Sender verification is format/domain-based** | Checking a domain against a configuration file does not prove the sender's identity. Email addresses can be spoofed. |
| **Phone number ownership is not verified** | Without an external telecom/HLR lookup service, the application cannot verify who owns a phone number. |
| **Email spoofing not detected** | SPF, DKIM, and DMARC authentication records are not inspected — only the domain string is checked. |
| **External reputation requires configuration** | Without `REPUTATION_API_KEY` set, no external URL reputation data is available. |
| **Risk score is heuristic** | The 0–100 score is a weighted aggregation of signals, not a calibrated probability of fraud. The weighting (60/25/15/10) is a reasonable default, not empirically optimized. |
| **Trusted-domain list is manually maintained** | The configuration file contains a small example list of 20 domains and is not comprehensive. |
| **English only** | The model and preprocessing pipeline are English-only; no multilingual support. |

---

## Future Improvements

1. **Deep learning models** — Train an LSTM or BERT model for better context and semantic understanding
2. **Real-time URL scanning** — Enable the external reputation API with a Google Safe Browsing key (architecture is ready)
3. **Email header analysis** — Parse SPF, DKIM, and DMARC records for email authentication verification
4. **Larger dataset** — Incorporate email spam datasets (Enron, SpamAssassin) for broader coverage
5. **Multilingual support** — Extend detection to languages other than English
6. **Model explainability** — Show which words/features contributed most to the ML prediction
7. **Active learning** — Let users flag incorrect predictions to retrain and improve the model
8. **Batch processing** — Allow uploading a CSV of messages for bulk analysis
9. **Risk score calibration** — Use empirical data to optimize the evidence weighting
10. **Deployment as REST API** — Wrap the analysis pipeline in a FastAPI service for integration into messaging platforms

---

## Testing

The project was tested with the following test cases, all run against the actual trained model and analysis modules (no mocked results):

### Test Results

| Test Case | Message | Sender | Score | Category |
|-----------|---------|--------|-------|----------|
| Normal message | "Hey, are we meeting at 5 PM today?" | — | 0/100 | LOW RISK |
| Spam prize (no URL) | "Congratulations! You won a cash prize..." | — | 59/100 | MEDIUM RISK |
| Spam + HTTP IP URL | "URGENT: Verify at http://192.168.1.5/claim" | — | 70/100 | HIGH RISK |
| Normal + HTTPS trusted URL | "Found a great article: https://github.com/..." | — | 4/100 | LOW RISK |
| Spam + shortened URL | "Claim here: http://bit.ly/free-gift" | — | 67/100 | HIGH RISK |
| Trusted email sender | "Your order has shipped." | support@amazon.com | 28/100 | LOW RISK |
| Unknown domain sender | "You have won a prize!" | admin@suspicious-domain-123.tk | 59/100 | MEDIUM RISK |
| Phone sender spam | "Your mobile won $5000." | +1-555-999-8888 | 64/100 | MEDIUM RISK |

### Edge Cases Tested

| Edge Case | Result |
|-----------|--------|
| Empty message | Correctly rejected with validation error |
| Whitespace-only message | Correctly rejected with validation error |
| Invalid URL format | Analyzed without crashing (produces a risk score) |
| No URLs in message | No URL analysis performed, URL risk = 0 |
| Reputation API not configured | Displays "not configured" — no fabricated results |
| Original ML model | Confirmed preserved and functional (Multinomial Naive Bayes) |

### How to Re-run Tests

```bash
cd spam-message-detector
python -c "
import sys; sys.path.insert(0, '.')
from src.predict import SpamDetector
d = SpamDetector(); d.load()
r = d.predict('Win a free iPhone now!')
print(f'Label: {r[\"label\"]}, Confidence: {r[\"confidence\"]*100:.1f}%')
"
```

---

## Interview Questions

### 1. What is the NLP pipeline in this project?

The pipeline processes raw text through five steps: lowercasing (so "FREE" and "free" match), removing special characters (punctuation adds noise), whitespace normalization, tokenization, and stopword removal (common words like "the" and "is" don't help distinguish spam from ham). The cleaned text is then vectorized using TF-IDF and classified by Multinomial Naive Bayes. The same pipeline is applied during training and inference via a scikit-learn Pipeline object, ensuring consistency and preventing data leakage.

### 2. What is TF-IDF and why use it instead of raw word counts?

TF-IDF (Term Frequency–Inverse Document Frequency) weighs each word by how frequently it appears in a specific message (TF) versus how common it is across all messages (IDF). Raw word counts treat all words equally, so common words like "the" would dominate. TF-IDF down-weights common words and up-weights distinctive ones, making spam-associated terms like "prize" and "claim" stand out. We also use sublinear TF (1 + log(tf)) to prevent a single repeated word from dominating. In this project, TF-IDF uses unigrams and bigrams to capture phrases like "free entry."

### 3. Why is the TF-IDF vectorizer fitted only on training data?

If the vectorizer is fitted on the entire dataset including test data, it learns the vocabulary and IDF weights from the test set — this is data leakage. The model would perform artificially well because it has already "seen" the test data's word distribution. By fitting only on training data inside a Pipeline, the test set simulates truly unseen messages, giving an honest estimate of real-world performance.

### 4. How does the risk scoring system work?

The final 0–100 score combines four evidence sources: ML text risk (up to 60 points, based on the Naive Bayes spam probability), URL risk (up to 25 points, based on static URL analysis scaled from 0–100), sender/domain risk (up to 15 points, based on domain verification status), and an optional reputation bonus (up to 10 points if an external API flags the URL). The score maps to LOW (0–33), MEDIUM (34–66), or HIGH (67–100). Importantly, this is a heuristic aggregation, not a calibrated probability — the full breakdown is shown in the UI so the user can see exactly how the score was computed.

### 5. Why not just use the ML model's prediction?

The ML model only analyzes text patterns. It cannot detect whether a URL uses an IP address, whether a domain is spoofed, or whether the sender's email domain is recognized. A well-crafted phishing message might not trigger the spam model but could have a suspicious URL or unrecognized domain. Combining multiple evidence sources — text, URL, sender, and optional external reputation — provides a more realistic and useful risk assessment than text classification alone.

### 6. What does the URL analyzer check and what does it NOT do?

The URL analyzer performs 13+ static checks: HTTPS vs HTTP, IP-based URLs, shortening services (28 recognized), Punycode/IDN domains, URL length, @ symbols, subdomain count, hyphens, digit ratio, suspicious TLDs (25 entries), phishing keywords in paths (16 keywords), non-standard ports, and embedded credentials. It does NOT visit, download, or execute anything from any URL — it purely inspects the URL string. This is a safety principle: the application never interacts with potentially dangerous URLs.

### 7. How does sender/domain verification work and what are its limitations?

The system extracts the domain from the sender's email address and checks it against a configuration file (`config/trusted_domains.json`). If the domain is in the list, it's "recognized"; if not, it's "unknown." For phone numbers, the system validates the format but explicitly states that ownership cannot be verified. The key limitation is that email addresses can be spoofed — recognizing a domain doesn't prove the sender controls that email account. The system does not inspect SPF, DKIM, or DMARC records. The UI clearly communicates these limitations.

### 8. How is the external reputation API designed?

The `reputation_checker.py` module reads the API key from the `REPUTATION_API_KEY` environment variable — never from source code. It supports Google Safe Browsing (v4) and VirusTotal (v3). If no key is configured, it reports "not configured" and does not fabricate results. If a key is present, it makes a real HTTP request to the provider's API and returns the actual response, clearly labeled as external data. This allows the app to work standalone while remaining ready for production use with a real API key.

### 9. Why does the app never claim "100% safe" or "verified"?

Because no single automated system can guarantee safety. The ML model can be wrong, URLs can be newly created, email addresses can be spoofed, and reputation databases have false negatives. Claiming certainty would be misleading and potentially dangerous. Instead, the app uses probabilistic language ("likely," "appears," "unable to verify") and always recommends independent verification for important messages. The risk score is explicitly described as a heuristic, not a guarantee.

### 10. How would you improve this for production?

I would: (1) enable the external reputation API with a real Google Safe Browsing key, (2) add email header analysis for SPF/DKIM/DMARC authentication, (3) train on a larger, more diverse dataset including email, (4) use a transformer model like BERT for better text understanding, (5) implement active learning from user feedback, (6) calibrate the risk score weights using empirical ground-truth data, (7) deploy as a microservice API with rate limiting, and (8) add multilingual support.
