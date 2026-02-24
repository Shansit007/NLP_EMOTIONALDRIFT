# =========================
# IMPORT LIBRARIES
# =========================

import re
from pathlib import Path

import nltk
import spacy
import matplotlib.pyplot as plt
import numpy as np
from nrclex import NRCLex
from textblob import TextBlob

# Load spaCy model (used for sentence segmentation + POS tagging)
nlp = spacy.load("en_core_web_sm")

# Download punkt tokenizer (for sentence splitting)
nltk.download('punkt')


# =========================
# STEP 1: READ TEXT FILE
# =========================

def load_text(file_path):
    """
    Reads input text file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# =========================
# STEP 2: CLEAN TEXT
# (Using Regular Expressions)
# =========================

def clean_text(text):
    """
    Removes special characters and extra spaces.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)
    return text


# =========================
# STEP 3: SENTENCE SEGMENTATION
# (Discourse Segmentation Concept)
# =========================

def segment_sentences(text):
    """
    Breaks text into sentences using spaCy.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


# =========================
# STEP 4: SENTENCE LEVEL SENTIMENT
# =========================

def get_sentiment_scores(sentences):
    """
    Computes sentiment polarity for each sentence.
    Polarity range:
    -1 (negative) to +1 (positive)
    """
    scores = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        scores.append(blob.sentiment.polarity)
    return scores


# =========================
# STEP 4B: SUBJECTIVITY SCORES
# =========================

def get_subjectivity_scores(sentences):
    """
    Computes subjectivity for each sentence.
    Range:
    0 (objective) to 1 (subjective)
    """
    scores = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        scores.append(blob.sentiment.subjectivity)
    return scores


# =========================
# STEP 5: SLIDING WINDOW ANALYSIS
# (Temporal Emotion Tracking)
# =========================

def sliding_window_average(scores, window_size=3):
    """
    Smoothens emotion values over time.
    """
    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - window_size + 1)
        window = scores[start:i+1]
        smoothed.append(np.mean(window))
    return smoothed


# =========================
# STEP 5B: EMOTION CATEGORIES
# (NRC Lexicon via NRCLex)
# =========================

EMOTION_CATEGORIES = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def get_emotion_category_scores(sentences, emotions=None):
    """
    Computes normalized emotion category scores per sentence.
    """
    if emotions is None:
        emotions = EMOTION_CATEGORIES

    scores = {emotion: [] for emotion in emotions}
    for sentence in sentences:
        lex = NRCLex(sentence)
        raw_scores = lex.raw_emotion_scores
        word_count = len(re.findall(r"[A-Za-z']+", sentence))
        denom = max(1, word_count)
        for emotion in emotions:
            scores[emotion].append(raw_scores.get(emotion, 0) / denom)
    return scores


def smooth_emotion_categories(emotion_scores, window_size=3):
    """
    Smooths emotion category scores with a sliding window.
    """
    return {
        emotion: sliding_window_average(values, window_size)
        for emotion, values in emotion_scores.items()
    }


# =========================
# STEP 6: DETECT EMOTIONAL SHIFTS
# =========================

def detect_shifts(scores, threshold=0.5):
    """
    Detects sudden emotional shifts.
    If difference between consecutive sentences > threshold,
    we mark it as emotional jump.
    """
    shifts = []
    for i in range(1, len(scores)):
        if abs(scores[i] - scores[i-1]) > threshold:
            shifts.append(i)
    return shifts


# =========================
# STEP 7: PLOT EMOTION TIMELINE
# =========================

def plot_emotion_timeline(scores, shifts, output_dir="results", filename="emotional_drift.png"):
    """
    Creates emotion timeline graph.
    """
    plt.figure(figsize=(10,5))
    plt.plot(scores, marker='o', label="Emotion Score")
    
    # Mark sudden shifts
    for shift in shifts:
        plt.axvline(x=shift, color='red', linestyle='--')

    plt.title("Emotional Drift Timeline")
    plt.xlabel("Sentence Index (Time)")
    plt.ylabel("Sentiment Polarity")
    plt.legend()
    plt.grid(True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150)
    plt.show()


# =========================
# STEP 8: PLOT EMOTION CATEGORIES
# =========================

def plot_emotion_categories(emotion_scores, output_dir="results", filename="emotion_categories.png"):
    """
    Plots emotion category drift over time.
    """
    plt.figure(figsize=(10,5))
    for emotion, values in emotion_scores.items():
        plt.plot(values, label=emotion.capitalize())

    plt.title("Emotion Category Drift")
    plt.xlabel("Sentence Index (Time)")
    plt.ylabel("Normalized Emotion Score")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150)
    plt.show()


# =========================
# STEP 9: PLOT SUBJECTIVITY DRIFT
# =========================

def plot_subjectivity_timeline(scores, output_dir="results", filename="subjectivity_drift.png"):
    """
    Plots subjectivity drift over time.
    """
    plt.figure(figsize=(10,5))
    plt.plot(scores, marker='o', label="Subjectivity")

    plt.title("Subjectivity Drift Timeline")
    plt.xlabel("Sentence Index (Time)")
    plt.ylabel("Subjectivity (0-1)")
    plt.legend()
    plt.grid(True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=150)
    plt.show()


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    file_path = "data/sample_op_ed.txt"

    # Load
    text = load_text(file_path)

    # Clean
    cleaned = clean_text(text)

    # Segment
    sentences = segment_sentences(cleaned)

    # Sentence-level sentiment
    scores = get_sentiment_scores(sentences)

    # Subjectivity
    subjectivity_scores = get_subjectivity_scores(sentences)

    # Sliding window smoothing
    smoothed_scores = sliding_window_average(scores)
    smoothed_subjectivity_scores = sliding_window_average(subjectivity_scores)

    # Detect sudden shifts
    shifts = detect_shifts(smoothed_scores)

    # Plot
    plot_emotion_timeline(smoothed_scores, shifts)

    # Subjectivity plot
    plot_subjectivity_timeline(smoothed_subjectivity_scores)

    # Emotion categories
    emotion_scores = get_emotion_category_scores(sentences)
    smoothed_emotion_scores = smooth_emotion_categories(emotion_scores)
    plot_emotion_categories(smoothed_emotion_scores)