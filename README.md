# Election Analysis 2026

A full-stack machine learning application for analyzing historical Legislative Assembly election data and generating model-based predictions, turnout scenarios, retention analysis, backtesting, interactive visualizations, and constituency-level election insights.

The application combines **Python, Pandas, Scikit-learn, Random Forest, PostgreSQL, FastAPI, Streamlit, Plotly, Folium, and GeoJSON** into an end-to-end machine learning system.

---

## Project Overview

**Election Analysis 2026** analyzes historical election data from:

* 2011
* 2016
* 2021

The application uses historical election information to identify patterns related to:

* Voter turnout
* Turnout changes
* Margin of victory
* Swing factors
* Previous election results
* Party retention
* Demographic classification
* Constituency-level behavior

The processed data is used to train machine learning models that can generate predictions for future election scenarios.


<img width="767" height="342" alt="image" src="https://github.com/user-attachments/assets/e80ed06a-648b-4358-87bf-f9ec7076b0c3" />

<img width="1136" height="411" alt="image" src="https://github.com/user-attachments/assets/74acb53f-895a-4e5d-81e9-8146baef4b86" />

<img width="1905" height="856" alt="image" src="https://github.com/user-attachments/assets/0bde299f-742c-4050-a5a6-37de47dd4524" />


The application follows a full-stack architecture:

```text
Historical Election Data
        |
        v
Data Cleaning & Preparation
        |
        v
Feature Engineering
        |
        v
Machine Learning Training
        |
        v
Saved ML Models
        |
        v
FastAPI Backend
        |
        v
Streamlit Frontend
        |
        v
Charts + Tables + Maps
        |
        v
Prediction / Scenario / Backtesting
```

---

# Key Features

### Historical Election Analysis

Analyze election information from 2011, 2016, and 2021.

### Machine Learning Prediction

Generate constituency-level predictions using trained Random Forest models.

### Margin Prediction

Predict the expected election margin using historical election features.

### Party Prediction

Predict the possible winning party based on historical patterns and engineered features.

### Retention Prediction

Estimate whether the previous winning party is likely to retain a constituency.

### Turnout Scenario Simulation

Simulate hypothetical changes in voter turnout.

Example:

```text
Current Turnout = 75%

Scenario:
Turnout Increase = 5%

Scenario Turnout = 80%
```

The modified data is passed through the prediction API to observe how the model's output changes.

### Historical Backtesting

Evaluate the prediction system against historical election results.

Example:

```text
Training:
2011 + 2016

Prediction:
2021

Comparison:
Predicted Result vs Actual Result
```

### Interactive Dashboard

The Streamlit dashboard presents:

* Seats won by party
* Turnout scenario analysis
* Retention analysis
* Party flips
* Turnout trends
* Constituency details
* Election maps
* ML predictions

### Interactive Election Map

Folium and GeoJSON are used to visualize constituency-level election information geographically.

---

# System Architecture

```text
                    HISTORICAL ELECTION DATA
                              |
                              v
                 +-------------------------+
                 |     Data Preparation    |
                 |                         |
                 | Cleaning                |
                 | Transformation         |
                 | Validation             |
                 +-----------+-------------+
                             |
                             v
                 +-------------------------+
                 |   Feature Engineering   |
                 |                         |
                 | Turnout                |
                 | Margin                 |
                 | Swing                  |
                 | Previous Results       |
                 | Demographics           |
                 +-----------+-------------+
                             |
                             v
                 +-------------------------+
                 |    Machine Learning     |
                 |                         |
                 | Random Forest           |
                 | Regression              |
                 | Classification         |
                 +-----------+-------------+
                             |
                             v
                 +-------------------------+
                 |     Saved ML Models     |
                 |                         |
                 | Margin Model            |
                 | Retention Model         |
                 | Party Model             |
                 +-----------+-------------+
                             |
                             v
                 +-------------------------+
                 |       FastAPI           |
                 |                         |
                 | /health                 |
                 | /predict                |
                 | /predict/bulk           |
                 | /predict/scenario       |
                 | /predict/backtesting    |
                 | /predict/retention      |
                 +-----------+-------------+
                             |
                             | HTTP
                             v
                 +-------------------------+
                 |       Streamlit         |
                 |                         |
                 | Dashboard               |
                 | Predictions             |
                 | Scenarios               |
                 | Backtesting             |
                 | Charts                  |
                 | Tables                  |
                 | Maps                    |
                 +-----------+-------------+
                             |
                             v
                         USER
```

---

# Technology Stack

| Layer                 | Technology         |
| --------------------- | ------------------ |
| Programming Language  | Python             |
| Frontend              | Streamlit          |
| Backend API           | FastAPI            |
| Database              | PostgreSQL         |
| Machine Learning      | Scikit-learn       |
| ML Algorithm          | Random Forest      |
| Data Processing       | Pandas, NumPy      |
| Visualization         | Plotly             |
| Mapping               | Folium             |
| Geospatial Data       | GeoJSON            |
| Model Serialization   | Joblib             |
| Database Connectivity | Psycopg2           |
| Configuration         | python-dotenv      |
| HTTP Communication    | Requests           |
| Dataset Formats       | CSV, Excel         |
| API Documentation     | FastAPI Swagger UI |

---

# Project Structure

```text
Election_Analysis/
│
├── app.py
│
├── api.py
│
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── database/
│   └── connection.py
│
├── dataset/
│   ├── raw/
│   │
│   └── training/
│       ├── election_2011.csv
│       ├── election_2016.csv
│       ├── election_2021.csv
│       ├── legislative_assembly_data.xlsx
│       └── merged_election_data.csv
│
├── machine_learning/
│   ├── election_model.py
│   └── scenario_simulator.py
│
├── models/
│   ├── rf_margin_model.pkl
│   ├── rf_retained_model.pkl
│   ├── rf_party_model.pkl
│   ├── state_encoder.json
│   └── party_encoder.json
│
├── visualization/
│   └── election_charts.py
│
└── maps/
    └── kerala_assembly.geojson
```

> The exact dataset files may vary depending on the version of the project and the training pipeline used.

---

# Application Flow

## 1. Historical Data

The project begins with historical Legislative Assembly election datasets.

```text
2011 Election
     |
     +------+
            |
2016 Election
     |
     +------+
            |
2021 Election
     |
     v
Combined Historical Dataset
```

The datasets contain election-related information such as:

```text
State
Constituency
Party
Candidate
Votes
Vote Percentage
Turnout
Margin
Demographic Category
```

---

# 2. Data Cleaning

Before the data is used for machine learning, it is cleaned and standardized.

Typical operations include:

```text
Remove duplicate records
        |
Handle missing values
        |
Standardize column names
        |
Remove unnecessary spaces
        |
Convert numeric columns
        |
Convert percentage values
        |
Validate records
```

Example:

```text
" Urban "  ->  "Urban"

"78.4%"    ->  78.4

"Party A " ->  "Party A"
```

Cleaned data is then passed to the feature engineering stage.

---

# 3. Feature Engineering

Machine learning models cannot directly work with all raw election information.

Therefore, historical election data is transformed into numerical features.

Important features used by the prediction pipeline include:

```text
state_encoded
demographic_encoded
turnout_2016
turnout_2021
turnout_change_21
margin_2016
swing_2016
result_2016_encoded
```

Historical turnout changes are calculated from election data.

For example:

```text
2016 Turnout = 75%

2021 Turnout = 80%

Turnout Change
= 80 - 75
= 5%
```

This allows the model to learn relationships between turnout changes and election outcomes.

---

# 4. PostgreSQL Database

PostgreSQL is used as the persistent database layer.

The database stores structured election and constituency information.

Examples include:

```text
Constituency
State
District
Demographic Type
Latitude
Longitude
Election Year
Party
Candidate
Votes
Turnout
Margin
```

The database connection is managed through:

```text
database/connection.py
```

The connection configuration is loaded from environment variables.

Example:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=election_db
DB_USER=postgres
DB_PASSWORD=your_password
```

The database keeps application data separate from the frontend and machine learning code.

---

# 5. Machine Learning Pipeline

The machine learning pipeline is implemented in:

```text
machine_learning/election_model.py
```

The general training flow is:

```text
2011 Data
     |
2016 Data
     |
2021 Data
     |
     v
Data Cleaning
     |
     v
Historical Data Merge
     |
     v
Feature Engineering
     |
     v
Feature Encoding
     |
     v
Train/Test Split
     |
     v
Random Forest Models
     |
     v
Model Evaluation
     |
     v
Model Serialization
```

The trained models are stored in the `models/` directory.

---

# 6. Machine Learning Models

The project uses separate models for different prediction tasks.

## Margin Model

File:

```text
models/rf_margin_model.pkl
```

This model is used for margin-related prediction.

Conceptually:

```text
Historical Features
        |
        v
Random Forest Regression
        |
        v
Predicted Margin
```

Example:

```text
Predicted Margin = 6,200 votes
```

---

## Retention Model

File:

```text
models/rf_retained_model.pkl
```

This model performs classification for party/constituency retention.

Conceptually:

```text
Historical Features
        |
        v
Random Forest Classification
        |
        v
Retained / Not Retained
```

Example:

```text
Previous Winner = Party A

Prediction:
Party A likely to retain
```

---

## Party Prediction Model

File:

```text
models/rf_party_model.pkl
```

This model predicts the possible winning party.

Conceptually:

```text
Historical Election Features
        |
        v
Random Forest Classification
        |
        v
Predicted Party
```

---

# 7. Model Encoders

Machine learning models require numerical representations of categorical information.

The project stores the encoding information in:

```text
models/state_encoder.json
models/party_encoder.json
```

For example:

```text
State

Kerala      -> 0
Tamil Nadu  -> 1
Karnataka   -> 2
```

Similarly, party names are transformed into numerical representations.

The same mapping must be used during:

```text
Training
   |
   v
Prediction
```

This prevents inconsistencies between the training and prediction pipelines.

---

# 8. FastAPI Backend

The backend API is implemented in:

```text
api.py
```

FastAPI provides the machine learning prediction service.

The architecture is:

```text
Streamlit
    |
    | HTTP Request
    v
FastAPI
    |
    v
Feature Preparation
    |
    v
ML Models
    |
    v
Prediction
    |
    v
JSON Response
    |
    v
Streamlit
```

The Streamlit application does not directly run the ML prediction models for API-based prediction.

Instead, it communicates with FastAPI using HTTP requests.

This provides separation between:

```text
Frontend
Backend
Machine Learning
```

---

# 9. API Endpoints

## GET `/`

Basic API/root endpoint.

```text
GET /
```

Used to confirm that the API application is available.

---

## GET `/health`

Health-check endpoint.

```text
GET /health
```

Example response:

```json
{
    "status": "ok"
}
```

The Streamlit application can use this endpoint to determine whether the backend is running.

---

## POST `/predict`

Used for prediction of a single constituency or input record.

```text
POST /predict
```

Flow:

```text
Input Features
      |
      v
FastAPI
      |
      v
Feature Preparation
      |
      v
ML Models
      |
      v
Prediction
      |
      v
JSON Response
```

---

## POST `/predict/bulk`

Used for predicting multiple constituencies.

```text
POST /predict/bulk
```

Example:

```text
Multiple Constituencies
          |
          v
    /predict/bulk
          |
          v
      ML Models
          |
          v
Multiple Predictions
```

This endpoint is particularly useful for generating predictions for an entire election dataset.

---

## POST `/predict/scenario`

Used for hypothetical election scenario analysis.

```text
POST /predict/scenario
```

Example:

```text
Current Turnout = 75%

User selects:
Turnout Increase = 5%

Scenario Turnout = 80%

        |
        v

/predict/scenario

        |
        v

ML Prediction
```

The purpose of the scenario is to demonstrate how the model responds to a hypothetical change.

It should not be interpreted as a guaranteed election outcome.

---

## POST `/predict/backtesting`

Used for historical model evaluation.

```text
POST /predict/backtesting
```

Example:

```text
2011 + 2016
     |
     v
Train Model
     |
     v
Predict 2021
     |
     v
Compare With Actual 2021
```

Backtesting provides a way to evaluate model performance against historical data.

---

## POST `/predict/retention`

Used for retention prediction.

```text
POST /predict/retention
```

Example:

```text
Previous Winning Party
        +
Historical Features
        |
        v
Retention Model
        |
        v
Retention Prediction
```

---

# 10. Streamlit Frontend

The main frontend is:

```text
app.py
```

Streamlit provides the interactive user interface.

The frontend is responsible for:

```text
Dashboard UI
User Input
Data Selection
FastAPI Communication
Prediction Display
Chart Display
Table Display
Map Display
Application Status
```

The high-level flow is:

```text
User
 |
 v
Streamlit app.py
 |
 v
Prepare Data
 |
 v
Call FastAPI
 |
 v
Receive JSON Response
 |
 v
Process Results
 |
 v
ElectionCharts
 |
 +------> Charts
 |
 +------> Tables
 |
 +------> Maps
 |
 v
Dashboard
```

---

# 11. Visualization Layer

Visualization functionality is separated into:

```text
visualization/election_charts.py
```

The main visualization class is:

```text
ElectionCharts
```

Separating visualization from `app.py` makes the application easier to maintain.

The visualization layer handles election-related graphical output.

Examples include:

```text
Seats Won by Party
Turnout Scenario
Retention Analysis
Party Flips
Turnout Trend
Election Map
Constituency Details
```

Important processing methods include:

```text
_prepare_api_data()
_ensure_features()
```

These methods help prepare the data in the format expected by the prediction API and visualization functions.

---

# 12. Scenario Simulation

Scenario processing is implemented through:

```text
machine_learning/scenario_simulator.py
```

The scenario simulator allows the user to test hypothetical changes.

For example:

```text
Baseline Turnout
       |
       v
75%
       |
       v
User selects +5%
       |
       v
Scenario Turnout
       |
       v
80%
       |
       v
Prediction API
       |
       v
Scenario Result
```

The dashboard can compare:

```text
Baseline Prediction
        VS
Scenario Prediction
```

This is a **what-if simulation**, not an actual election forecast.

---

# 13. Election Map

The project uses:

```text
maps/kerala_assembly.geojson
```

The GeoJSON file contains geographical constituency boundaries.

The application combines:

```text
Election Data
      +
Prediction Data
      +
GeoJSON Boundaries
      |
      v
Folium
      |
      v
Interactive Election Map
```

The map can display constituency-level information based on election and prediction data.

---

# Complete Prediction Flow

The complete prediction flow is:

```text
                    USER
                     |
                     v
              Streamlit app.py
                     |
                     v
             Prepare Input Data
                     |
                     v
             HTTP POST Request
                     |
                     v
                FastAPI API
                     |
                     v
              Validate Input
                     |
                     v
           Prepare ML Features
                     |
                     v
              Load ML Models
                     |
                     v
            Generate Prediction
                     |
                     v
               JSON Response
                     |
                     v
              Streamlit App
                     |
                     v
              ElectionCharts
                     |
          +----------+----------+
          |          |          |
          v          v          v
        Charts     Tables      Maps
          |          |          |
          +----------+----------+
                     |
                     v
               USER RESULT
```

---

# Example Prediction

Suppose the input contains:

```text
State = Kerala

Demographic = Urban

Turnout 2016 = 75%

Turnout 2021 = 78%

Margin 2016 = 5000

Swing 2016 = 2.5

Previous Result = Party A
```

The data is prepared by the frontend and sent to:

```text
POST /predict
```

FastAPI processes the request:

```text
Input
 |
 v
Feature Transformation
 |
 v
Random Forest Models
 |
 v
Prediction
```

The API returns the result to Streamlit.

The dashboard can then display values such as:

```text
Predicted Party
Predicted Margin
Retention Prediction
```

These values represent the output of the trained machine learning models.

---

# Example Scenario

Assume:

```text
Current Turnout = 75%
```

The user selects:

```text
Turnout Increase = 5%
```

The application creates:

```text
Scenario Turnout = 80%
```

The scenario is sent through the prediction pipeline.

```text
75% Baseline
      |
      v
Baseline Prediction

80% Scenario
      |
      v
Scenario Prediction
```

The dashboard can compare the two results.

---

# Historical Backtesting

Backtesting is used to evaluate the model against historical election outcomes.

Example:

```text
Training Dataset
2011 + 2016
      |
      v
Machine Learning Model
      |
      v
Predict 2021
      |
      v
Actual 2021 Results
      |
      v
Comparison
```

For classification:

```text
Actual Winner:
Party A

Predicted Winner:
Party A

Result:
Correct
```

For regression:

```text
Actual Margin:
5,500

Predicted Margin:
5,100
```

The difference between actual and predicted values can be used to evaluate the regression model.

---

# Dashboard Structure

The Streamlit dashboard is designed around several analytical sections.

```text
+-----------------------------+-----------------------------+
|                             |                             |
|    Seats Won by Party       |   Turnout Scenario         |
|                             |   Simulator                 |
|                             |                             |
+-----------------------------+-----------------------------+

+----------------+----------------+----------------+
|                |                |                |
|   Retention    |  Party Flips   | Turnout Trend  |
|                |                |                |
+----------------+----------------+----------------+

+------------------------------------------------------------+
|                  Constituency Details                      |
+------------------------------------------------------------+

+------------------------------------------------------------+
|                     Election Map                           |
+------------------------------------------------------------+
```

This allows historical analysis and model-based analysis to be viewed in one interface.

# Data and Feature Flow

The overall data transformation is:

```text
Raw Election Data
       |
       v
Cleaning
       |
       v
Standardization
       |
       v
Historical Merge
       |
       v
Feature Engineering
       |
       v
Categorical Encoding
       |
       v
Training Features
       |
       v
Machine Learning
```

---

# Feature Set

The prediction API uses a consistent feature structure.

The core prediction features include:

| Feature               | Description                                      |
| --------------------- | ------------------------------------------------ |
| `state_encoded`       | Numerical representation of state                |
| `demographic_encoded` | Numerical representation of demographic category |
| `turnout_2016`        | Voter turnout in 2016                            |
| `turnout_2021`        | Voter turnout in 2021                            |
| `turnout_change_21`   | Change in turnout between historical elections   |
| `margin_2016`         | Historical margin of victory                     |
| `swing_2016`          | Historical swing factor                          |
| `result_2016_encoded` | Encoded previous election result                 |

Feature names must remain consistent across:

```text
Training
      |
      v
FastAPI
      |
      v
Streamlit
      |
      v
Visualization
```

A mismatch between training and prediction feature names can cause prediction errors.

---

# Model Storage

Trained models are serialized using Joblib.

```text
models/
│
├── rf_margin_model.pkl
├── rf_retained_model.pkl
└── rf_party_model.pkl
```

Encoders:

```text
models/
│
├── state_encoder.json
└── party_encoder.json
```

The models are loaded by the backend/prediction layer when required.

---

# Database Connection

Database functionality is separated into:

```text
database/connection.py
```

The application uses environment variables for database configuration.

Example:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=election_db
DB_USER=postgres
DB_PASSWORD=your_password
```

Do not commit actual database passwords or secrets to GitHub.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Greeshma-babu/Election_Analysis.git
```

Move into the project directory:

```bash
cd Election_Analysis
```

---

# 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project dependencies include packages such as:

```text
psycopg2-binary
python-dotenv
pandas
openpyxl
streamlit
folium
streamlit-folium
plotly
scikit-learn
xgboost
joblib
numpy
fastapi
uvicorn
requests
```

---

# 4. Configure PostgreSQL

Install and start PostgreSQL.

Create the project database:

```text
election_db
```

Configure the `.env` file:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=election_db
DB_USER=postgres
DB_PASSWORD=your_password
```

Use your local PostgreSQL credentials.

---

# 5. Prepare the Dataset

Place the required historical datasets under:

```text
dataset/training/
```

Expected datasets include:

```text
election_2011.csv
election_2016.csv
election_2021.csv
legislative_assembly_data.xlsx
```

The training pipeline can generate the merged training dataset:

```text
merged_election_data.csv
```

---

# 6. Train the Machine Learning Models

Run the machine learning training pipeline from the project environment.

The training process generates the model files under:

```text
models/
```

Expected model artifacts:

```text
rf_margin_model.pkl
rf_retained_model.pkl
rf_party_model.pkl
state_encoder.json
party_encoder.json
```

---

# 7. Start FastAPI

From the project root:

```bash
uvicorn api:app --reload
```

The API normally runs at:

```text
http://127.0.0.1:8000
```

---

# 8. Open FastAPI Swagger Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

You can use Swagger UI to test endpoints such as:

```text
GET  /health

POST /predict

POST /predict/bulk

POST /predict/scenario

POST /predict/backtesting

POST /predict/retention
```

---

# 9. Start Streamlit

Open another terminal.

Activate the virtual environment if necessary:

```bash
.venv\Scripts\activate
```

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# Running the Complete Application

Two services are involved:

```text
              Election Analysis Application

                       USER
                        |
                        v
                Streamlit :8501
                        |
                        | HTTP
                        v
                  FastAPI :8000
                        |
                        v
                  ML Models
                        |
                        v
                   Prediction
                        |
                        v
                  Streamlit
```

Recommended startup order:

```text
1. PostgreSQL
      |
2. FastAPI
      |
3. Streamlit
```

---

# API Testing Flow

A typical API testing flow is:

```text
Start FastAPI
      |
      v
Open /docs
      |
      v
Select endpoint
      |
      v
Provide JSON input
      |
      v
Execute
      |
      v
Check response
```

This makes it possible to test the backend independently from Streamlit.

---

# Separation of Responsibilities

One of the main architectural principles of the project is separation of responsibilities.

```text
+----------------------+---------------------------------------+
| Component            | Responsibility                        |
+----------------------+---------------------------------------+
| app.py               | Streamlit frontend                    |
| api.py               | FastAPI backend / model serving       |
| connection.py        | PostgreSQL connectivity               |
| election_model.py    | ML training and feature engineering   |
| scenario_simulator.py| Scenario processing                   |
| election_charts.py   | Charts, maps and visualization        |
| PostgreSQL           | Persistent structured data            |
| .pkl files           | Trained machine learning models       |
| JSON encoders        | Categorical encoding mappings         |
+----------------------+---------------------------------------+
```

This means:

```text
Frontend
   |
   | displays and collects input
   v
FastAPI
   |
   | validates and serves prediction
   v
Machine Learning
   |
   | generates model output
   v
Frontend
   |
   | visualizes result
   v
User
```

---

# End-to-End Architecture

```text
                 ┌───────────────────────┐
                 │ Historical Election   │
                 │ Data                  │
                 │ 2011 / 2016 / 2021   │
                 └───────────┬───────────┘
                             |
                             v
                 ┌───────────────────────┐
                 │ Data Cleaning         │
                 │ & Preparation         │
                 └───────────┬───────────┘
                             |
                             v
                 ┌───────────────────────┐
                 │ Feature Engineering   │
                 │ Turnout / Margin /    │
                 │ Swing / Demographics  │
                 └───────────┬───────────┘
                             |
                  ┌──────────┴──────────┐
                  |                     |
                  v                     v
        ┌─────────────────┐   ┌──────────────────┐
        │ PostgreSQL      │   │ ML Training      │
        │ Database        │   │ Random Forest    │
        └─────────────────┘   └────────┬─────────┘
                                       |
                                       v
                              ┌─────────────────┐
                              │ Saved Models    │
                              │ .pkl + Encoders │
                              └────────┬────────┘
                                       |
                                       v
                              ┌─────────────────┐
                              │ FastAPI         │
                              │ Prediction API  │
                              └────────┬────────┘
                                       |
                              HTTP / JSON
                                       |
                                       v
                              ┌─────────────────┐
                              │ Streamlit       │
                              │ Dashboard       │
                              └────────┬────────┘
                                       |
                    ┌──────────────────┼──────────────────┐
                    |                  |                  |
                    v                  v                  v
                Plotly              Tables             Folium
                Charts                                  Maps
                    |                  |                  |
                    └──────────────────┼──────────────────┘
                                       |
                                       v
                                  User Analysis
```

---

# Academic Requirements Coverage

| Requirement              | Implementation                                              |
| ------------------------ | ----------------------------------------------------------- |
| Historical Election Data | 2011, 2016, 2021 datasets                                   |
| Data Cleaning            | Data preparation pipeline                                   |
| Feature Engineering      | Turnout, margin, swing, demographics and historical results |
| Voter Turnout            | Historical turnout features                                 |
| Margin of Victory        | Margin features and margin prediction                       |
| Swing Factor             | Historical swing feature                                    |
| Demographic Analysis     | Urban / Semi-Urban / Rural encoding                         |
| Machine Learning         | Scikit-learn Random Forest                                  |
| Regression               | Margin prediction                                           |
| Classification           | Party and retention prediction                              |
| Scenario Simulation      | Turnout scenario API                                        |
| Historical Backtesting   | Backtesting API                                             |
| Backend                  | FastAPI                                                     |
| Frontend                 | Streamlit                                                   |
| Database                 | PostgreSQL                                                  |
| Visualization            | Plotly                                                      |
| Maps                     | Folium + GeoJSON                                            |
| Model Serialization      | Joblib                                                      |
| API Testing              | FastAPI Swagger UI                                          |

---

# Project Lifecycle

The complete machine learning lifecycle is:

```text
1. Data Collection
        |
        v
2. Data Cleaning
        |
        v
3. Data Transformation
        |
        v
4. Feature Engineering
        |
        v
5. Database Storage
        |
        v
6. Model Training
        |
        v
7. Model Evaluation
        |
        v
8. Model Serialization
        |
        v
9. API Model Serving
        |
        v
10. Prediction
        |
        v
11. Scenario Simulation
        |
        v
12. Historical Backtesting
        |
        v
13. Visualization
        |
        v
14. Interactive Dashboard
```

---

# Example Use Cases

## Historical Analysis

A user can analyze:

```text
Which parties performed strongly?
How did turnout change?
What were the historical margins?
Which constituencies changed parties?
```

---

## Constituency Prediction

A user can provide historical constituency features and obtain model-based predictions.

```text
Historical Features
        |
        v
FastAPI
        |
        v
ML Models
        |
        v
Prediction
```

---

## Turnout Scenario

A user can test:

```text
What happens if turnout increases by 5%?
```

The application modifies the turnout feature and generates a scenario prediction.

---

## Retention Analysis

The application can analyze whether historical patterns indicate that a previous winning party may retain a constituency.

---

## Historical Backtesting

The application can simulate prediction on a historical election and compare the predicted output with the actual result.

---

# Important Design Principle

The project follows a layered architecture:

```text
DATA
  |
  v
DATABASE
  |
  v
MACHINE LEARNING
  |
  v
FASTAPI
  |
  v
STREAMLIT
  |
  v
VISUALIZATION
  |
  v
USER
```

Each layer has a separate responsibility.

### Frontend

Handles:

```text
User interaction
Dashboard
Charts
Tables
Maps
```

### Backend

Handles:

```text
API requests
Validation
Prediction serving
JSON responses
```

### Machine Learning

Handles:

```text
Data preparation
Feature engineering
Training
Evaluation
Model generation
```

### Database

Handles:

```text
Persistent structured information
```

### Visualization

Handles:

```text
Charts
Maps
Analytical displays
```

This makes the application modular and easier to maintain.

---

# Project Outcome

This project demonstrates a complete full-stack machine learning workflow.

It combines:

```text
Data Engineering
      |
      v
Feature Engineering
      |
      v
Machine Learning
      |
      v
Model Serialization
      |
      v
FastAPI Model Serving
      |
      v
Streamlit Frontend
      |
      v
Visualization
      |
      v
Scenario Simulation
      |
      v
Historical Backtesting
      |
      v
Interactive Election Analysis
```

The final application demonstrates how historical election data can be transformed into an interactive machine learning application with:

* Historical analysis
* Machine learning prediction
* Party prediction
* Retention analysis
* Turnout scenario simulation
* Historical backtesting
* PostgreSQL integration
* FastAPI model serving
* Streamlit dashboard
* Plotly visualization
* Folium election maps

---

# Conclusion

**Election Analysis 2026** demonstrates the complete lifecycle of a full-stack machine learning application.

The project connects:

```text
Historical Data
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
PostgreSQL
      ↓
Machine Learning
      ↓
Saved Models
      ↓
FastAPI
      ↓
Streamlit
      ↓
Charts / Tables / Maps
      ↓
Scenario Analysis
      ↓
Historical Backtesting
```

The architecture separates data processing, machine learning, API services, database operations, frontend interaction, and visualization into dedicated components.

This makes the project suitable as an academic demonstration of how a machine learning model can be integrated into a complete application rather than being used only as an isolated Python script.

---
