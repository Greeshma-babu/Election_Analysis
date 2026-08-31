# Full Stack ML Application: Election Analysis 2026

## Project Overview

**Election Analysis 2026** is a full-stack machine learning application that analyzes historical Legislative Assembly election data from **2011, 2016, and 2021** to identify voting trends, incumbency patterns, turnout changes, demographic influence, and constituency-level election behavior.

The application uses historical data to train machine learning models and provides **2026 election predictions and scenario-based analysis** through an interactive Streamlit dashboard.

The project demonstrates the complete machine learning lifecycle:

**Data Collection → Data Cleaning → Feature Engineering → Database → Model Training → Model Prediction → FastAPI → Streamlit Dashboard → Visualization → Scenario Simulation → Historical Backtesting**

---

# Project Objectives

The application is designed to:

* Analyze historical Legislative Assembly election results.
* Study voter turnout and turnout changes.
* Calculate margin of victory and swing factors.
* Analyze incumbency and party retention.
* Incorporate demographic information such as Urban, Semi-Urban, and Rural.
* Train machine learning models using historical election data.
* Predict possible 2026 election outcomes.
* Simulate election scenarios such as a 5% increase in voter turnout.
* Perform historical backtesting.
* Store constituency-level information in PostgreSQL.
* Provide an interactive dashboard for analysis and visualization.

---

# System Architecture

```text
                         HISTORICAL ELECTION DATA
                                  |
                                  v
                    +-----------------------------+
                    |       Data Engineering      |
                    |                             |
                    |  Data Cleaning              |
                    |  Data Transformation        |
                    |  Feature Engineering        |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |       PostgreSQL DB          |
                    |                             |
                    | Constituency Metadata       |
                    | Election Information        |
                    | Party Information           |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |      Machine Learning       |
                    |                             |
                    | Feature Preparation         |
                    | Model Training              |
                    | Model Evaluation            |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |       Saved ML Models       |
                    |                             |
                    | Margin Model                |
                    | Retention Model             |
                    | Party Prediction Model      |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |          FastAPI             |
                    |                             |
                    | /health                     |
                    | /predict                    |
                    | /predict/bulk               |
                    | /predict/scenario           |
                    | /predict/backtesting        |
                    | /predict/retention          |
                    +-------------+---------------+
                                  |
                                  v
                    +-----------------------------+
                    |        Streamlit App        |
                    |                             |
                    | Dashboard                   |
                    | Prediction                  |
                    | Scenario Simulation         |
                    | Backtesting                 |
                    | Charts                      |
                    | Maps                        |
                    +-------------+---------------+
                                  |
                                  v
                         USER / FINAL RESULT
```

---

# Technology Stack

| Layer                 | Technology         |
| --------------------- | ------------------ |
| Programming Language  | Python             |
| Frontend              | Streamlit          |
| Backend               | FastAPI            |
| Database              | PostgreSQL         |
| Machine Learning      | Scikit-learn       |
| Models                | Random Forest      |
| Data Processing       | Pandas, NumPy      |
| Visualization         | Plotly             |
| Maps                  | Folium             |
| Model Storage         | Joblib             |
| Configuration         | python-dotenv      |
| API Communication     | Requests           |
| Database Connectivity | Psycopg2           |
| Data Format           | CSV / Excel        |
| API Testing           | FastAPI Swagger UI |

---

# Project Structure

```text
Election_Analysis/
│
├── app.py
│
├── fastapi_app.py
│
├── connection.py
│
├── election_mode.py
│
├── scenario_simulator.py
│
├── requirements.txt
├── .env
│
├── api/
│   └── ...
│
├── dataset/
│   ├── raw/
│   └── training/
│       ├── election_2011.csv
│       ├── election_2016.csv
│       ├── election_2021.csv
│       └── legislative_assembly_data.xlsx
│
├── machine_learning/
│   ├── election_model.py
│   └── ...
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

---

# Application Flow

## Step 1: Historical Data

The application starts with historical election data from:

```text
2011
2016
2021
```

The data contains information such as:

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

Example:

```text
Constituency: Ernakulam
Year: 2021
Party: Party A
Votes: 65000
Turnout: 78.4
Margin: 5200
Demographic: Urban
```

---

# Step 2: Data Cleaning

The data engineering stage cleans the historical datasets.

Typical operations include:

```text
Remove duplicate records
Handle missing values
Standardize column names
Remove unwanted spaces
Convert percentages to numeric values
Convert categorical values
Validate election records
```

Example:

```text
"  Urban " → "Urban"

"78.4%" → 78.4

"Party A " → "Party A"
```

The cleaned data is then used for feature engineering and model training.

---

# Step 3: Feature Engineering

Raw election information is transformed into machine-learning features.

Important features include:

```text
state_encoded
demographic_encoded
turnout_2011
turnout_2016
turnout_2021
turnout_change
margin_2016
margin_2021
swing_factor
result_2016_encoded
incumbency information
```

### Example

Suppose:

```text
2016 turnout = 75%
2021 turnout = 80%
```

Then:

```text
turnout_change = 80 - 75
               = 5%
```

The model can use this information to understand whether increasing turnout is associated with changes in election outcomes.

---

# Step 4: Database Layer

PostgreSQL stores constituency-level information and election metadata.

The database can contain information such as:

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

The database connection is handled through:

```text
connection.py
```

The application obtains a database connection using the PostgreSQL configuration stored in environment variables.

---

# Step 5: Machine Learning Training

Historical election data is passed to the machine learning pipeline.

The training process is:

```text
Historical Data
       |
       v
Feature Engineering
       |
       v
Train/Test Split
       |
       v
Random Forest Model
       |
       v
Model Evaluation
       |
       v
Save Model
```

The project uses multiple prediction models.

### Margin Prediction Model

Predicts the expected margin of victory.

```text
Input:
Turnout
Historical Margin
Swing
Demographic
Previous Election Result
       |
       v
Random Forest
       |
       v
Predicted Margin
```

### Retention Model

Predicts whether the incumbent/previous winning party is likely to retain the constituency.

Example:

```text
Previous winner = Party A
Historical margin = High
Turnout = Stable
Swing = Low

Prediction:
Party A likely to retain
```

### Party Prediction Model

Predicts the possible winning party based on historical features.

---

# Step 6: Model Storage

After training, the models are saved using Joblib.

```text
models/
│
├── rf_margin_model.pkl
├── rf_retained_model.pkl
└── rf_party_model.pkl
```

Encoders are also stored so that the same categorical mappings are used during prediction.

```text
state_encoder.json
party_encoder.json
```

This is important because the prediction system must transform input data in exactly the same way as the training system.

---

# Step 7: FastAPI Backend

FastAPI acts as the backend prediction service.

The main backend file is:

```text
fastapi_app.py
```

The Streamlit frontend does not directly execute the machine learning models.

Instead:

```text
Streamlit
    |
    | HTTP POST
    v
FastAPI
    |
    v
Machine Learning Model
    |
    v
Prediction
    |
    v
FastAPI Response
    |
    v
Streamlit
```

This creates a proper full-stack architecture.

---

# Step 8: API Endpoints

## `/health`

Checks whether FastAPI is running.

Example response:

```json
{
    "status": "ok"
}
```

The Streamlit application uses this endpoint to verify backend availability.

---

## `/predict`

Predicts an election outcome for a single constituency.

Example:

```text
Input
    |
    v
Constituency Features
    |
    v
FastAPI
    |
    v
ML Model
    |
    v
Prediction
```

---

## `/predict/bulk`

Predicts multiple constituencies at once.

This endpoint is useful when the dashboard needs predictions for an entire election dataset.

Example:

```text
1000 Constituencies
       |
       v
/predict/bulk
       |
       v
ML Models
       |
       v
1000 Predictions
```

---

## `/predict/scenario`

Used for scenario simulation.

Example:

```text
Current turnout = 75%

Scenario:
Increase turnout by 5%

New turnout = 80%

       |
       v
ML Model
       |
       v
Scenario Prediction
```

This allows users to understand how election outcomes could change under different assumptions.

---

## `/predict/backtesting`

Used to evaluate the model against historical election outcomes.

For example:

```text
Train using:
2011 + 2016

Predict:
2021

Compare:
Predicted Result vs Actual Result
```

This provides an indication of how well the model performs on unseen historical data.

---

## `/predict/retention`

Predicts whether a party is likely to retain a constituency.

Example:

```text
Previous Winner = Party A

Historical Features
        |
        v
Retention Model
        |
        v
Retained = Yes
```

---

# Step 9: Streamlit Frontend

The main frontend file is:

```text
app.py
```

Streamlit provides the interactive user interface.

The dashboard can contain:

```text
Election Analysis Dashboard

+----------------------+----------------------+
| Seats Won by Party  | Turnout Scenario     |
|                      | Simulator            |
+----------------------+----------------------+

+-------------+-------------+----------------+
| Retention   | Party Flips | Turnout Trend |
+-------------+-------------+----------------+

+--------------------------------------------+
| Constituency Details                       |
+--------------------------------------------+

+--------------------------------------------+
| Election Map                               |
+--------------------------------------------+
```

---

# Step 10: Visualization Layer

The visualization logic is separated into:

```text
visualization/election_charts.py
```

This class is responsible for generating charts and maps.

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

Separating visualization from `app.py` keeps the frontend code cleaner and easier to maintain.

---

# Important Classes and Files

## 1. `app.py`

### Responsibility

`app.py` is the main Streamlit frontend.

It controls:

```text
Dashboard UI
User inputs
FastAPI communication
Prediction results
Charts
Tables
Application status
```

### Flow

```text
User
 |
 v
app.py
 |
 +----> Prepare Input Data
 |
 +----> Call FastAPI
 |
 +----> Receive Prediction
 |
 +----> Display Charts
 |
 +----> Display Map
```

### Example

If the user selects:

```text
Turnout Increase = 5%
```

`app.py` sends the scenario information to FastAPI.

---

# 2. `fastapi_app.py`

### Responsibility

This is the backend API layer.

It:

```text
Loads trained models
Receives requests
Validates input
Prepares model features
Runs predictions
Returns JSON responses
```

Example:

```text
POST /predict/bulk
```

The backend receives constituency data and returns predictions.

---

# 3. `connection.py`

### Responsibility

Handles PostgreSQL database connections.

Instead of opening database connections throughout the application, the project uses a centralized connection function.

Conceptually:

```text
app
 |
 v
connection.py
 |
 v
PostgreSQL
```

Example:

```text
get_connection()
      |
      v
PostgreSQL Connection
```

---

# 4. `election_model.py`

### Responsibility

This file contains the machine learning logic.

It is responsible for:

```text
Loading election datasets
Merging historical elections
Feature engineering
Encoding categorical variables
Training models
Evaluating models
Saving models
```

The general flow is:

```text
2011 Data
     +
2016 Data
     +
2021 Data
     |
     v
Feature Engineering
     |
     v
Training Dataset
     |
     v
Random Forest
     |
     v
Trained Models
```

---

# 5. `election_charts.py`

### Responsibility

Contains the visualization class:

```text
ElectionCharts
```

This class receives election data and generates visual outputs.

Major responsibilities:

```text
_prepare_api_data()
_ensure_features()

plot_seats_won_by_party()
plot_turnout_scenario()
plot_merged_election_map()
```

It also ensures that the input data contains the features required by the prediction system.

---

# 6. `scenario_simulator.py`

### Responsibility

Handles what-if election scenarios.

Example:

```text
Original turnout = 75%

User Scenario:
Turnout +5%

Modified turnout = 80%
```

The modified data is sent to the prediction system.

The purpose is not to claim that the actual election will produce that result.

It demonstrates how the model's prediction changes under a simulated assumption.

---

# 7. `election_mode.py`

### Responsibility

Provides election-related processing/mode logic used by the application.

Depending on the implementation, it can help organize election modes such as:

```text
Historical Analysis
Prediction
Scenario Simulation
Backtesting
```

This keeps election-specific decision logic separate from the main dashboard.

---

# 8. Model Files

## `rf_margin_model.pkl`

Random Forest regression model used to predict margin-related outcomes.

```text
Historical Features
        |
        v
rf_margin_model.pkl
        |
        v
Predicted Margin
```

---

## `rf_retained_model.pkl`

Classification model used to predict retention.

```text
Historical Features
        |
        v
rf_retained_model.pkl
        |
        v
Retained / Not Retained
```

---

## `rf_party_model.pkl`

Classification model used to predict the possible winning party.

```text
Historical Features
        |
        v
rf_party_model.pkl
        |
        v
Predicted Party
```

---

# 9. Encoders

## `state_encoder.json`

Converts state names into numerical values that the ML model can understand.

Example:

```text
Kerala → 0
Tamil Nadu → 1
Karnataka → 2
```

## `party_encoder.json`

Converts party names into numerical representations.

Example:

```text
Party A → 0
Party B → 1
Party C → 2
```

The same encoding must be used during both training and prediction.

---

# 10. GeoJSON Map File

```text
maps/kerala_assembly.geojson
```

This contains the geographical boundaries of Assembly constituencies.

It is used by Folium to create the election map.

The prediction data is joined with the geographical constituency information.

```text
Prediction Data
       +
GeoJSON
       |
       v
Folium Map
       |
       v
Interactive Election Map
```

---

# Complete End-to-End Flow

The complete application works as follows:

```text
                 1. HISTORICAL DATA
                         |
                         v
              2011 / 2016 / 2021 Data
                         |
                         v
                 2. DATA CLEANING
                         |
                         v
              Clean and Standardize
                         |
                         v
                3. FEATURE ENGINEERING
                         |
                         v
       Turnout / Margin / Swing / Demographics
                         |
                         v
                 4. POSTGRESQL
                         |
                         v
              Constituency Metadata
                         |
                         v
                 5. MODEL TRAINING
                         |
                         v
                Random Forest Models
                         |
                         v
                 6. MODEL STORAGE
                         |
                         v
                 .pkl Model Files
                         |
                         v
                  7. FASTAPI
                         |
              +----------+----------+
              |          |          |
              v          v          v
           /predict   /scenario  /backtesting
              |          |          |
              +----------+----------+
                         |
                         v
                 8. ML PREDICTION
                         |
                         v
                 JSON API Response
                         |
                         v
                 9. STREAMLIT
                         |
              +----------+----------+
              |          |          |
              v          v          v
            Charts      Tables      Maps
              |          |          |
              +----------+----------+
                         |
                         v
                 10. USER ANALYSIS
```

---

# Example: Normal Prediction Flow

Suppose the user wants a prediction for a constituency.

### Input

```text
State = Kerala
Demographic = Urban
Turnout 2016 = 75%
Turnout 2021 = 78%
Margin 2016 = 5000
Swing 2016 = 2.5
Previous Result = Party A
```

### Process

```text
Streamlit
    |
    v
Prepare input
    |
    v
POST /predict
    |
    v
FastAPI
    |
    v
Load ML model
    |
    v
Feature transformation
    |
    v
Random Forest
    |
    v
Prediction
    |
    v
JSON response
    |
    v
Streamlit
```

### Output

The dashboard can display:

```text
Predicted Party: Party A
Predicted Margin: 6,200
Retention: Yes
```

---

# Example: Scenario Simulation

Suppose the actual historical turnout is:

```text
75%
```

The user selects:

```text
Turnout Increase = 5%
```

The application creates:

```text
Scenario Turnout = 80%
```

Then:

```text
80% Turnout
     |
     v
FastAPI
     |
     v
ML Model
     |
     v
Scenario Prediction
```

The dashboard compares:

```text
Baseline Prediction
        vs
Scenario Prediction
```

This demonstrates the effect of a hypothetical turnout change.

---

# Example: Historical Backtesting

Backtesting checks whether the model can reproduce a historical election.

Example:

```text
Training Data
2011 + 2016
       |
       v
ML Model
       |
       v
Predict 2021
       |
       v
Compare with Actual 2021
```

For classification:

```text
Actual Winner = Party A
Predicted Winner = Party A

Correct
```

For regression:

```text
Actual Margin = 5,500
Predicted Margin = 5,100
```

The difference can then be used to evaluate model performance.

---

# Why FastAPI Is Used

A major design decision in this project is separating the frontend and machine learning backend.

Instead of:

```text
Streamlit → ML Model
```

the project uses:

```text
Streamlit → FastAPI → ML Model
```

Advantages:

* Clear separation of frontend and backend.
* Machine learning models are centralized.
* APIs can be reused by other applications.
* Easier testing.
* Easier deployment.
* Better representation of a real-world ML system.

---

# Why PostgreSQL Is Used

PostgreSQL is used to maintain structured constituency and election metadata.

Instead of storing everything inside Python variables:

```text
Python
   |
   v
PostgreSQL
```

The database provides persistent storage and allows structured querying.

---

# Why Random Forest Is Used

Random Forest is suitable for this project because election data contains a mixture of numerical and categorical-derived features and may contain nonlinear relationships.

The project uses Random Forest for:

```text
Regression
Classification
```

It also provides a relatively robust baseline for tabular datasets.

---

# Full Application Responsibility

```text
+----------------------+--------------------------------------+
| Component            | Responsibility                       |
+----------------------+--------------------------------------+
| app.py               | Streamlit frontend                   |
| fastapi_app.py       | REST API backend                     |
| connection.py        | PostgreSQL connection                |
| election_model.py    | ML training and feature engineering  |
| election_charts.py   | Charts and maps                      |
| scenario_simulator.py| What-if scenario processing          |
| election_mode.py     | Election processing/modes            |
| PostgreSQL           | Persistent election metadata         |
| .pkl files           | Trained ML models                    |
| JSON encoders        | Categorical encoding                 |
| GeoJSON              | Constituency boundaries              |
+----------------------+--------------------------------------+
```

---

# Academic Requirements Coverage

| Assignment Requirement                    | Implementation                   |
| ----------------------------------------- | -------------------------------- |
| Historical election data 2011, 2016, 2021 | `dataset/training/`              |
| Data cleaning                             | Data engineering pipeline        |
| Margin of victory                         | Feature engineering              |
| Voter turnout                             | Historical and scenario features |
| Swing factors                             | Feature engineering              |
| Rural/Urban demographic data              | `demographic_encoded`            |
| Regression/Classification                 | Random Forest models             |
| Scenario simulation                       | `/predict/scenario`              |
| Historical backtesting                    | `/predict/backtesting`           |
| Backend                                   | FastAPI                          |
| Frontend                                  | Streamlit                        |
| Map visualization                         | Folium + GeoJSON                 |
| Database                                  | PostgreSQL                       |
| Constituency metadata                     | PostgreSQL                       |
| Model serving                             | FastAPI                          |
| Interactive dashboard                     | Streamlit                        |

---

# Important Architecture Principle

The most important concept in this project is the separation of responsibilities.

```text
DATA
 ↓
DATABASE
 ↓
ML
 ↓
API
 ↓
FRONTEND
 ↓
VISUALIZATION
```

Each layer performs a specific job.

The frontend should not contain the complete ML logic.

The backend should not contain dashboard visualization logic.

The database should store data rather than perform prediction.

The ML layer should train and provide models.

This makes the application modular and maintainable.

---

# Final Project Flow

```text
User opens Streamlit Dashboard
              |
              v
        app.py starts
              |
              v
   Check FastAPI /health
              |
              v
       FastAPI available
              |
              v
     User selects analysis
              |
              v
     Prepare election data
              |
              v
       Send HTTP request
              |
              v
          FastAPI
              |
              v
      Validate input data
              |
              v
       Prepare ML features
              |
              v
       Load trained models
              |
              v
       Generate prediction
              |
              v
      Return JSON response
              |
              v
          Streamlit
              |
              v
     Convert response to data
              |
              v
      ElectionCharts class
              |
       +------+------+
       |      |      |
       v      v      v
     Charts  Tables  Maps
       |      |      |
       +------+------+
              |
              v
        Final Dashboard
```

---

# Project Outcome

The project demonstrates an end-to-end full-stack machine learning application capable of:

1. Processing historical election data.
2. Engineering meaningful election features.
3. Storing structured constituency information.
4. Training machine learning models.
5. Serving predictions through FastAPI.
6. Performing hypothetical scenario simulations.
7. Performing historical backtesting.
8. Presenting predictions through an interactive Streamlit dashboard.
9. Visualizing election information through charts, tables, and maps.

The 2026 results generated by the system should be interpreted as **model-based hypothetical predictions**, not as guaranteed election outcomes.

---

# How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure PostgreSQL

Create the required database and configure the environment variables in `.env`.

Example:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=election_db
DB_USER=postgres
DB_PASSWORD=your_password
```

## 3. Train the models

Run the machine learning training pipeline.

This generates the required files inside:

```text
models/
```

## 4. Start FastAPI

```bash
uvicorn fastapi_app:app --reload
```

FastAPI will normally be available at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

## 5. Start Streamlit

```bash
streamlit run app.py
```

The dashboard will open in the browser.

---

# Conclusion

**Election Analysis 2026** demonstrates how a machine learning model can be transformed into a complete production-style application.

The project connects:

```text
Data Engineering
       ↓
Machine Learning
       ↓
Database
       ↓
REST API
       ↓
Interactive Frontend
       ↓
Visualization
       ↓
Scenario Analysis
       ↓
Backtesting
```

This architecture demonstrates the complete lifecycle of a full-stack machine learning application, from historical data processing to model-driven interactive analysis.
