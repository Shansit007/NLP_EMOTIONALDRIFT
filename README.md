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