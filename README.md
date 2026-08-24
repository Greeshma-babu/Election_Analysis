# Election Analysis 2026

Kerala Assembly Election Analysis and Prediction using Machine Learning — EDA, voter turnout, victory margin, vote swing, urban/rural demographics, PostgreSQL, and Random Forest.

---

## Table of Contents

- [Overview](#overview)
- [Description](#description)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Setup](#database-setup)
- [Installation & Running the App](#installation--running-the-app)
- [Usage](#usage)

---

## Overview

This project analyzes Kerala Assembly election data across three election cycles — **2011, 2016, and 2021** — to explore trends in voter turnout, victory margins, vote swing, and urban/rural demographics, and to build machine learning models that predict election outcomes at the constituency level.

The cleaned and merged dataset is stored in a **PostgreSQL** database, and results are explored interactively through a **Streamlit** dashboard.

Create a database — `election_db` — in PostgreSQL before running the app.

## Description

The core of this project merges three years of constituency-level election data into a single dataset (one row per constituency, spanning all three elections) and trains machine learning models to predict future election outcomes.

## Models

Three Random Forest models are trained on the merged dataset:

| Model | Description |
|---|---|
|  **Margin model** | Predicts the winning margin |
|  **Retention model** | Predicts whether a seat is strongly retained |
|  **Party model** | Predicts which party wins |

Categorical fields (state, demographic, winning party) are converted into numeric form using **LabelEncoder**, since ML models require numeric input for training and testing.

## Feature Engineering

- **Merging:** `election_2011.csv`, `election_2016.csv`, and `election_2021.csv` are merged on `constituency_id` as the common key, so each row represents one constituency across all three elections.
- **Turnout change:** difference in voter turnout between consecutive elections.
- **Margin change:** difference in victory margin between consecutive elections.
- **Seat flip:** whether the winning party changed between elections (`1` = seat flipped, `0` = seat retained).
- **Strong retention:** a seat is considered *strongly retained* if `margin_2021 > 5`; otherwise it is treated as not (strongly) retained.

## Tech Stack

**Languages & Libraries**

- `pathlib` — handles file and folder paths
- `pandas` — reads and processes CSV data
- `joblib` — saves and loads trained ML models
- `scikit-learn` — LabelEncoder, train/test split, Random Forest models, evaluation metrics
- `streamlit` — interactive dashboard
- `psycopg2` — PostgreSQL connectivity
- `python-dotenv` — environment variable management

**Database**

- PostgreSQL

## Project Structure

```
Election_Analysis/
├── app.py                          # Streamlit entry point
├── database/
│   └── connection.py                # PostgreSQL connection, table creation, insert/update
├── machine_learning/
│   └── election_model.py            # Data merge, feature engineering, model training
├── visualization/
│   └── election_charts.py           # Chart rendering for Streamlit
├── dataset/
│   └── training/
│       ├── election_2011.csv
│       ├── election_2016.csv
│       ├── election_2021.csv
│       └── merged_election_data.csv # generated after training
├── models/                          # generated after training
│   ├── rf_margin_model.pkl
│   ├── rf_retained_model.pkl
│   ├── rf_party_model.pkl
│   ├── state_encoder.json
│   └── party_encoder.json
├── requirements.txt
└── README.md
```

## Database Setup

1. Install and start PostgreSQL locally (or use a hosted instance).
2. Create a database named `election_db`:

   ```sql
   CREATE DATABASE election_db;
   ```

3. Create a `.env` file in the project root with your database credentials:

   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=election_db
   DB_USER=your_username
   DB_PASSWORD=your_password
   ```

4. The `election_constituency` table is created automatically by `database/connection.py` when the app runs — no manual table creation is required.

## Installation & Running the App

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\Activate

# Verify the environment's pip
& "..\.venv\Scripts\python.exe" -m pip --version

# Install dependencies
& "..\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"

# (if not already included in requirements.txt)
pip install streamlit

# Run the Streamlit app
streamlit run app.py
```

## Usage

1. Train the models (optional, if `models/` is not already populated):

   ```bash
   python machine_learning/election_model.py
   ```

2. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

3. The app will:
   - Load and clean `dataset/training/merged_election_data.csv`
   - Create/verify the `election_constituency` table in PostgreSQL
   - Insert or update the election records
   - Render interactive charts — seats won by party, and a turnout scenario simulator