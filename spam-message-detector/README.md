# Spam Message Detector

## Overview

The **Spam Message Detector** is a machine learning and Natural Language Processing (NLP) project that classifies SMS text messages as either **SPAM** or **NOT SPAM** (ham). It uses the SMS Spam Collection dataset, applies text preprocessing, converts messages into numerical features using TF-IDF, and trains classification models to identify spam patterns. A Streamlit web interface lets users type or paste any message and get an instant prediction with confidence scores.

This project is built as a B.Tech CSE (AI/ML) portfolio piece and demonstrates practical skills in Python, NLP, feature engineering, model training and evaluation, and deployment.

---

## Problem Statement

Spam messages are a persistent problem in SMS communication — they waste time, promote scams, and can compromise user privacy. Manually filtering spam is impractical at scale. The goal of this project is to build an automated classifier that can accurately distinguish spam messages from legitimate ones using machine learning trained on real labeled data.

The key challenge is balancing **precision** (not flagging legitimate messages as spam) with **recall** (catching as many spam messages as possible), since both false positives and false negatives carry real costs.

---

## Features

- Real machine learning model trained on 5,572 labeled SMS messages
- NLP preprocessing pipeline (lowercasing, punctuation removal, stopword filtering)
- TF-IDF feature extraction with unigrams and bigrams
- Two classifiers trained and compared (Multinomial Naive Bayes & Logistic Regression)
- Model selection based on F1-score
- Full evaluation: accuracy, precision, recall, F1-score, confusion matrix
- Interactive Streamlit web interface with confidence scores
- Example messages for quick testing
- Error handling for empty input, missing model, and missing dataset

---

## Dataset

**Source:** [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — UCI Machine Learning Repository

- **Total messages:** 5,572
- **Spam:** 747
- **Ham (not spam):** 4,825
- **Format:** Tab-separated file with two columns: label (`ham`/`spam`) and message text

The dataset is automatically downloaded during project setup (see *How to Run*). If automatic download fails, manually download the zip from the UCI link above, extract `SMSSpamCollection`, and convert it to `data/spam.csv` with columns `label` and `message`.

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
    (convert text to numerical feature vectors)
    ↓
Machine Learning Model
    (Multinomial Naive Bayes / Logistic Regression)
    ↓
Prediction
    (SPAM or NOT SPAM + confidence)
```

### Preprocessing Steps Explained

| Step | What it does | Why |
|------|-------------|-----|
| Lowercasing | Converts all text to lowercase | "FREE" and "free" should be treated as the same word |
| Remove special characters | Strips punctuation, symbols, emojis | Punctuation rarely helps classification and adds noise |
| Whitespace normalization | Collapses multiple spaces, trims ends | Ensures clean token boundaries |
| Tokenization | Splits text into individual words | Needed for stopword removal and vectorization |
| Stopword removal | Drops common words ("the", "is", "at") | These words appear in both spam and ham equally and don't help distinguish them |
| Rejoin tokens | Combines words back into a string | TF-IDF vectorizer takes strings as input |

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Primary programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF, model training, evaluation, pipeline |
| NLTK | Stopword list for preprocessing |
| Matplotlib | Confusion matrix visualization |
| Streamlit | Web interface for interactive prediction |

---

## Machine Learning Models

Two classifiers were trained and compared:

### 1. Multinomial Naive Bayes
- **Type:** Probabilistic classifier based on Bayes' theorem
- **Why suitable:** Designed for text classification with discrete features (word counts/TF-IDF). Fast, simple, and performs well on spam detection. Assumes features are conditionally independent given the class.
- **Parameter:** `alpha=0.1` (Laplace smoothing — small value because the vocabulary is large)

### 2. Logistic Regression
- **Type:** Linear classifier that models the probability of belonging to a class
- **Why suitable:** Works well with TF-IDF features, provides interpretable coefficients, and outputs well-calibrated probabilities. `class_weight="balanced"` handles the class imbalance (747 spam vs 4,825 ham).
- **Parameters:** `max_iter=1000`, `solver="liblinear"`, `class_weight="balanced"`

### Model Selection

Both models are evaluated on the test set using accuracy, precision, recall, and F1-score. The model with the **highest F1-score** is selected and saved, because F1 balances precision and recall — both of which matter for spam detection.

---

## Evaluation

### Why Precision and Recall Matter for Spam Detection

- **Precision** = Of all messages predicted as spam, how many were actually spam? High precision means few legitimate messages are wrongly flagged (few false positives).
- **Recall** = Of all actual spam messages, how many did we catch? High recall means few spam messages slip through (few false negatives).
- **F1-score** = Harmonic mean of precision and recall — balances both.

For spam detection, both matter: flagging a legitimate message as spam (false positive) is annoying, but missing a scam message (false negative) can be harmful. F1-score balances these concerns.

### Actual Results

The evaluation results are generated by running `python -m src.train` and are saved to `models/metrics.txt`. The values below are populated by the training script from the real test set — they are **not** hardcoded.

**Dataset:** 5,565 messages (747 spam, 4,818 ham) — 80/20 stratified split (4,452 train / 1,113 test)

| Model | Accuracy | Precision | Recall | F1-score | Confusion Matrix |
|-------|----------|-----------|--------|----------|-----------------|
| Multinomial Naive Bayes | 0.9874 | 1.0000 | 0.9060 | **0.9507** | TN=964, FP=0, FN=14, TP=135 |
| Logistic Regression | 0.9704 | 0.8671 | 0.9195 | 0.8925 | TN=943, FP=21, FN=12, TP=137 |

**Selected model:** Multinomial Naive Bayes (highest F1-score of 0.9507)

- Naive Bayes achieved **perfect precision** (zero false positives) — no legitimate message was wrongly flagged as spam
- Logistic Regression achieved **higher recall** (caught more spam) at the cost of some false positives
- Confusion matrix plots are saved to `models/confusion_matrix.png`, `models/confusion_matrix_nb.png`, and `models/confusion_matrix_lr.png`

---

## How to Run

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Step 1: Install Dependencies

```bash
cd spam-message-detector
pip install -r requirements.txt
```

### Step 2: Get the Dataset

The dataset is included as `data/spam.csv`. If it's missing, run:

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

### Step 3: Train the Model

```bash
python -m src.train
```

This will:
- Load and preprocess the dataset
- Split into 80% training / 20% testing (stratified)
- Train Multinomial Naive Bayes and Logistic Regression
- Evaluate both and print metrics
- Save the best model to `models/spam_classifier.pkl`
- Save metrics to `models/metrics.txt`
- Save confusion matrix plots to `models/`

### Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser. Type a message and click **Analyze** to get a prediction.

---

## Project Structure

```
spam-message-detector/
│
├── data/
│   └── spam.csv                 # SMS Spam Collection dataset
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Text preprocessing pipeline
│   ├── train.py                 # Model training & evaluation
│   └── predict.py               # Prediction utilities
│
├── models/
│   ├── spam_classifier.pkl      # Saved model pipeline
│   ├── metrics.txt              # Evaluation results
│   ├── confusion_matrix.png     # Best model confusion matrix
│   ├── confusion_matrix_nb.png  # Naive Bayes confusion matrix
│   └── confusion_matrix_lr.png  # Logistic Regression confusion matrix
│
├── app.py                       # Streamlit web interface
├── requirements.txt
├── README.md
└── screenshots/                 # Add screenshots of the app here
```

---

## Screenshots

Add screenshots of the Streamlit app here after running it:

1. `screenshots/app_main.png` — Main interface with message input
2. `screenshots/spam_result.png` — A spam message detected
3. `screenshots/ham_result.png` — A legitimate message detected
4. `screenshots/confidence.png` — Confidence scores displayed

---

## Future Improvements

1. **Deep learning models** — Train an LSTM or transformer (BERT) model for better context understanding
2. **Larger dataset** — Incorporate email spam datasets or the Enron corpus for broader coverage
3. **Multilingual support** — Extend to detect spam in languages other than English
4. **URL and phone number features** — Add feature engineering to detect suspicious links and numbers
5. **Real-time feedback** — Let users flag incorrect predictions to retrain the model
6. **Batch processing** — Allow uploading a CSV of messages for bulk classification
7. **Model explainability** — Show which words contributed most to a spam/ham prediction
8. **Deployment** — Deploy as a REST API using FastAPI or on Streamlit Community Cloud

---

## Interview Questions (Bonus)

### 1. What is NLP and how is it used in this project?
NLP (Natural Language Processing) is a field of AI that helps computers understand and process human language. In this project, NLP is used to preprocess raw text messages (lowercasing, removing stopwords, tokenization) and convert them into numerical features (TF-IDF) that a machine learning model can understand.

### 2. What is TF-IDF and why is it better than raw word counts?
TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word is to a message relative to all messages. Raw word counts (Bag of Words) just count occurrences — common words like "the" would dominate. TF-IDF down-weights words that appear in many messages and up-weights words unique to specific messages, making spam-related words like "prize" or "claim" stand out.

### 3. Why is the TF-IDF vectorizer fitted only on training data?
If we fit the vectorizer on the entire dataset (including test data), the test set would influence the vocabulary and IDF weights — this is called **data leakage**. The model would perform artificially well on the test set because it has already "seen" information from it. By fitting only on training data, we simulate real-world performance on truly unseen messages.

### 4. What is a train/test split and why do we use stratification?
A train/test split divides data into a training set (for learning) and a testing set (for evaluation). We use stratification to ensure the spam/ham ratio is preserved in both sets. Without stratification, the test set might have very few spam messages by chance, giving unreliable evaluation.

### 5. Why compare two models instead of using just one?
Different models have different strengths. Naive Bayes is fast and works well with text but assumes feature independence. Logistic Regression handles correlated features better and outputs calibrated probabilities. Comparing both on the same test set lets us select the one that performs best for our specific problem.

### 6. Why are precision and recall more important than accuracy for spam detection?
Accuracy can be misleading when classes are imbalanced (here, only 13% of messages are spam). A model that always predicts "ham" would be 87% accurate but completely useless. Precision tells us how many flagged messages were actually spam (avoid false alarms), and recall tells us how many spam messages we caught (don't miss scams). F1-score balances both.

### 7. What does the confusion matrix tell us?
The confusion matrix shows four values: True Negatives (ham correctly identified), False Positives (ham wrongly flagged as spam), False Negatives (spam missed), and True Positives (spam correctly caught). It helps us see not just overall accuracy but the specific types of errors the model makes.

### 8. What is the prediction flow when a user enters a message in the app?
1. The raw message is preprocessed (lowercase, remove punctuation, remove stopwords)
2. The cleaned text is transformed using the fitted TF-IDF vectorizer into a numerical vector
3. The trained classifier predicts the class (spam or ham)
4. If the model supports it, probability scores are computed for confidence
5. The result and confidence are displayed to the user

### 9. What is a Pipeline in scikit-learn and why use it?
A Pipeline chains multiple steps (TF-IDF + classifier) into a single object. It ensures that the same preprocessing is applied during training and prediction, prevents data leakage (the vectorizer is fitted only on training data inside the pipeline), and simplifies saving/loading the entire workflow as one pickle file.

### 10. How would you improve this model for production use?
I would: (1) use a larger and more diverse dataset, (2) add features like URL presence and sender reputation, (3) try deep learning models like BERT for better language understanding, (4) implement active learning to retrain on user feedback, (5) monitor precision/recall in production and retrain when performance drifts, and (6) deploy as an API for integration into messaging platforms.
