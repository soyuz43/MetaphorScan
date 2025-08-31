#!/bin/bash
# Download spaCy and DistilBERT models

# Activate virtual environment
source envs/metaphorscan_env/Scripts/activate

# Download spaCy model
/c/Python312/python -m spacy download en_core_web_sm

# Download DistilBERT via Python script
/c/Python312/python scripts/download_models.py
