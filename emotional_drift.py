# =========================
# IMPORT LIBRARIES
# =========================

import re  # Regular Expressions (syllabus: Formal Languages)
import nltk
import spacy
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob

# Load spacy model (used for sentence segmentation + POS tagging)
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

def plot_emotion_timeline(scores, shifts):
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
    plt.show()


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    file_path = "data/sample_script.txt"

    # Load
    text = load_text(file_path)

    # Clean
    cleaned = clean_text(text)

    # Segment
    sentences = segment_sentences(cleaned)

    # Sentence-level sentiment
    scores = get_sentiment_scores(sentences)

    # Sliding window smoothing
    smoothed_scores = sliding_window_average(scores)

    # Detect sudden shifts
    shifts = detect_shifts(smoothed_scores)

    # Plot
    plot_emotion_timeline(smoothed_scores, shifts)