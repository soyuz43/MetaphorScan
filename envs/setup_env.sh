#!/bin/bash
python3 -m venv envs/metaphorscan_env
source envs/metaphorscan_env/Scripts/activate  # Windows-compatible
pip install -r envs/requirements.txt
python -m spacy download en_core_web_sm
python scripts/download_models.py  # Placeholder for downloading DistilBERT