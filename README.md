# PROJECT: Emotional Drift Analyzer in Long Text

This program analyzes how emotions change over time in long text like movie scripts, diaries, or stories. 
For CSA4028: Natural Language Processing.

Core concepts used from NLP Syllabus:
- Sentence Segmentation (Discourse segmentation)
- POS Tagging (Syntax Parsing)
- Sliding Window Analysis (Temporal NLP)
- Sentiment Scoring (Models & Algorithms)
- Regular Expressions (Text Cleaning)

Code is highly commented for demonstration purposes in class.


## Setup

1. create venv
   
   ```
   py -m venv venv
   ```
   
2. activate venv
   
   ```
   venv/scripts/activate
   ```

3. install reqs
   
   ```
   pip install -r requirements.txt
   ```
   if spaCy english model not installed:
   ```
   python -m spacy download en_core_web_sm
   ```

4. run `emotional_drift.py`