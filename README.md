# TanglishGuard

An end-to-end MLOps pipeline for detecting offensive content in code-mixed Tamil-English (Tanglish) text, with automated drift monitoring and model retraining.

## Problem Statement
Existing content moderation tools are built for clean English and fail to detect offensive Tanglish content used heavily in Indian social media.

## Proposed Solution
TanglishGuard classifies Tanglish text as offensive or non-offensive, monitors model drift as slang evolves, and automatically retrains — no manual intervention needed.

## Tech Stack
- **ML Layer** — IndicBERT, TF-IDF, Logistic Regression, MLflow, DVC
- **Data Analysis** — Dravidian-CodeMix Dataset, Jupyter Notebook, Pandas
- **Backend** — FastAPI, Docker, GitHub Actions, Render
- **Frontend** — Streamlit

## SDG Alignment
- SDG 9 — Industry, Innovation and Infrastructure
- SDG 10 — Reduced Inequalities
- SDG 16 — Peace, Justice and Strong Institutions

## Project Structure
- `data/` — dataset files
- `notebooks/` — EDA and analysis notebooks
- `src/` — training, inference and monitoring scripts
- `models/` — saved model files

## Dataset
Dravidian-CodeMix Dataset — Tamil-English offensive language detection
https://zenodo.org/record/4300921

## Team
Vijay Vishal R & Saicharan A — III Year IT, CIT Chennai