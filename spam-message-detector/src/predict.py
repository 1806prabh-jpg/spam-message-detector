"""
Prediction utilities for the Spam Message Detector.

This module provides a thin wrapper around the saved pipeline so
that app.py and other callers can load the model once and make
predictions without duplicating logic.
"""

import os
import pickle

from src.preprocess import preprocess_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")


class SpamDetector:
    """Load the trained pipeline and classify messages."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.model_name = None
        self.label_map = None
        self.loaded = False

    def load(self):
        """Load the pickled pipeline from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Run `python -m src.train` first to train and save the model."
            )
        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)
        self.pipeline = bundle["pipeline"]
        self.model_name = bundle.get("model_name", "Unknown")
        self.label_map = bundle.get("label_map", {0: "NOT SPAM", 1: "SPAM"})
        self.loaded = True

    def predict(self, message: str) -> dict:
        """
        Predict whether a message is spam.

        Returns a dict with keys:
            label       - "SPAM" or "NOT SPAM"
            confidence  - float (probability of the predicted class) or None
            probabilities - dict of {label: probability} if available
            cleaned     - the preprocessed message
        """
        if not self.loaded:
            self.load()

        if not message or not message.strip():
            raise ValueError("Message cannot be empty.")

        cleaned = preprocess_text(message)
        if not cleaned.strip():
            raise ValueError(
                "Message is empty after preprocessing. "
                "Please enter a message with actual words."
            )

        # The pipeline expects an iterable
        pred = self.pipeline.predict([cleaned])[0]
        label = self.label_map.get(int(pred), "UNKNOWN")

        result = {
            "label": label,
            "confidence": None,
            "probabilities": None,
            "cleaned": cleaned,
        }

        # Try to get probability estimates
        try:
            proba = self.pipeline.predict_proba([cleaned])[0]
            # proba is [P(ham), P(spam)] for binary with classes [0, 1]
            classes = self.pipeline.classes_
            probabilities = {}
            for cls, p in zip(classes, proba):
                cls_label = self.label_map.get(int(cls), str(cls))
                probabilities[cls_label] = float(p)
            result["probabilities"] = probabilities
            result["confidence"] = float(max(proba))
        except (AttributeError, NotImplementedError):
            pass

        return result


# Convenience function for one-off predictions
def predict_message(message: str, model_path: str = MODEL_PATH) -> dict:
    """Load model, predict, and return the result dict."""
    detector = SpamDetector(model_path)
    return detector.predict(message)
