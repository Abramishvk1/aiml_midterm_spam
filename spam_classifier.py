#!/usr/bin/env python3
import re, pickle, sys
import numpy as np
import pandas as pd
from pathlib import Path

# Feature extraction function
SPAM_WORDS = ["free","win","winner","prize","click","buy","urgent","offer","money","cash","credit","limited","guarantee","click here","subscribe"]

def extract_features_from_text(text):
    # words: number of words
    words = len(re.findall(r"\w+", text))
    # links: count of http or www occurrences
    links = len(re.findall(r"https?://|www\.", text))
    # capital_words: words fully uppercase with length >=2
    capital_words = sum(1 for w in re.findall(r"\b\w+\b", text) if w.isupper() and len(w) >= 2)
    # spam_word_count: occurrences of spam words
    lower = text.lower()
    spam_word_count = sum(lower.count(sw) for sw in SPAM_WORDS)
    return np.array([words, links, capital_words, spam_word_count]).reshape(1,-1)

def load_model(path="spam_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def classify_text(text, model):
    feats = extract_features_from_text(text)
    pred = model.predict(feats)[0]
    prob = model.predict_proba(feats)[0]
    return pred, prob, feats.flatten().tolist()

if __name__ == "__main__":
    model = load_model(Path(__file__).parent / "spam_model.pkl")
    # If email text passed as file path
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            text = path  # treat as raw text
    else:
        print("Please paste your email text. End with an empty line (Ctrl+D to finish):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        text = "\n".join(lines)
    pred, prob, feats = classify_text(text, model)
    label = "SPAM" if pred==1 else "LEGITIMATE"
    print("Prediction:", label)
    print("Probability (not_spam, spam):", prob.tolist())
    print("Extracted features [words, links, capital_words, spam_word_count]:", feats)
