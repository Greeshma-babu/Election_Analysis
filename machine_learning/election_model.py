import pathlib
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    classification_report
)


# =========================================================
# PATH CONFIGURATION
# =========================================================

# This file is inside:
# Election_Analysis/machine_learning/
#
# parent.parent -> Election_Analysis/

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "training"
MODEL_DIR = BASE_DIR / "models"

# Create model directory if it does not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# STEP 1: LOAD THE THREE YEARLY ELECTION FILES
# =========================================================

df_2011 = pd.read_csv(
    DATA_DIR / "election_2011.csv"
)

df_2016 = pd.read_csv(
    DATA_DIR / "election_2016.csv"
)

df_2021 = pd.read_csv(
    DATA_DIR / "election_2021.csv"
)


# =========================================================
# STEP 2: RENAME COLUMNS
# =========================================================

df_2011 = df_2011.rename(columns={
    "result": "result_2011",
    "voter_turnout_pct": "turnout_2011",
    "margin_of_victory_pct": "margin_2011",
    "swing_factor_pct": "swing_2011"
}).drop(
    columns=["election_year"],
    errors="ignore"
)


df_2016 = df_2016.rename(columns={
    "result": "result_2016",
    "voter_turnout_pct": "turnout_2016",
    "margin_of_victory_pct": "margin_2016",
    "swing_factor_pct": "swing_2016"
}).drop(
    columns=["election_year"],
    errors="ignore"
)


df_2021 = df_2021.rename(columns={
    "result": "result_2021",
    "voter_turnout_pct": "turnout_2021",
    "margin_of_victory_pct": "margin_2021",
    "swing_factor_pct": "swing_2021"
}).drop(
    columns=["election_year"],
    errors="ignore"
)


# =========================================================
# STEP 3: MERGE 2011 + 2016 + 2021
# =========================================================

# 2011 contains the common columns:
# constituency_id
# state
# demographic
# result_2011
# turnout_2011
# margin_2011
# swing_2011
#
# For 2016 and 2021, state and demographic are removed
# because they already exist from 2011.

merged = df_2011.merge(
    df_2016.drop(
        columns=["state", "demographic"],
        errors="ignore"
    ),
    on="constituency_id"
).merge(
    df_2021.drop(
        columns=["state", "demographic"],
        errors="ignore"
    ),
    on="constituency_id"
)


print("Merged shape:", merged.shape)


# =========================================================
# STEP 4: DATA CLEANING
# =========================================================

# Remove duplicate constituencies if any
merged.drop_duplicates(
    subset="constituency_id",
    inplace=True
)


# Remove extra spaces from text columns
string_columns = merged.select_dtypes(
    include="object"
).columns

for column in string_columns:
    merged[column] = merged[column].str.strip()


# Convert numeric columns to numeric
numeric_columns = [
    "turnout_2011",
    "turnout_2016",
    "turnout_2021",
    "margin_2011",
    "margin_2016",
    "margin_2021",
    "swing_2011",
    "swing_2016",
    "swing_2021"
]

for column in numeric_columns:

    if column in merged.columns:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce"
        )


# Check missing values
missing_values = merged.isnull().sum()

if missing_values.any():

    print("\nMissing values found:")

    print(
        missing_values[
            missing_values > 0
        ]
    )


# =========================================================
# STEP 5: FEATURE ENGINEERING
# =========================================================

# Turnout change from 2011 -> 2016
merged["turnout_change_16"] = (
    merged["turnout_2016"]
    - merged["turnout_2011"]
)


# Turnout change from 2016 -> 2021
merged["turnout_change_21"] = (
    merged["turnout_2021"]
    - merged["turnout_2016"]
)


# Margin change from 2011 -> 2016
merged["margin_change_16"] = (
    merged["margin_2016"]
    - merged["margin_2011"]
)


# Margin change from 2016 -> 2021
merged["margin_change_21"] = (
    merged["margin_2021"]
    - merged["margin_2016"]
)


# Seat flip from 2011 -> 2016
merged["seat_flip_16"] = (
    merged["result_2016"]
    != merged["result_2011"]
).astype(int)


# Seat flip from 2016 -> 2021
merged["seat_flip_21"] = (
    merged["result_2021"]
    != merged["result_2016"]
).astype(int)


# Retention target
#
# 1 = Retained
# 0 = Lost
#
# Based on the current rule:
# margin > 5 -> retained

merged["retained_2021"] = (
    merged["margin_2021"] > 5
).astype(int)


# =========================================================
# STEP 6: ENCODE CATEGORICAL FEATURES
# =========================================================

# ---------------------------------------------------------
# DEMOGRAPHIC
# ---------------------------------------------------------

demographic_map = {
    "Urban": 0,
    "Semi-Urban": 1,
    "Rural": 2
}

merged["demographic_encoded"] = (
    merged["demographic"]
    .map(demographic_map)
)


# Check for unknown demographic values
unknown_demographic = merged.loc[
    merged["demographic_encoded"].isnull(),
    "demographic"
].unique()


if len(unknown_demographic) > 0:

    raise ValueError(
        "Unknown demographic values found: "
        f"{unknown_demographic.tolist()}"
    )


# ---------------------------------------------------------
# STATE ENCODER
# ---------------------------------------------------------

state_encoder = LabelEncoder()

merged["state_encoded"] = (
    state_encoder.fit_transform(
        merged["state"]
    )
)


# ---------------------------------------------------------
# PARTY ENCODER
# ---------------------------------------------------------

party_encoder = LabelEncoder()

merged["result_2016_encoded"] = (
    party_encoder.fit_transform(
        merged["result_2016"]
    )
)


# =========================================================
# STEP 7: DEFINE FEATURES
# =========================================================

# IMPORTANT:
#
# These columns and their order MUST remain exactly the
# same when making predictions later.
#
# The scenario simulator and election_charts.py use
# this same FEATURE_COLS list.

feature_cols = [
    "state_encoded",
    "demographic_encoded",
    "turnout_2016",
    "turnout_2021",
    "turnout_change_21",
    "margin_2016",
    "swing_2016",
    "result_2016_encoded"
]


# =========================================================
# STEP 8: BUILD X AND Y
# =========================================================

X = merged[feature_cols]


# ---------------------------------------------------------
# Target 1: Margin of Victory
# Regression problem
# ---------------------------------------------------------

y_margin = merged["margin_2021"]


# ---------------------------------------------------------
# Target 2: Incumbent Retention
# Classification problem
# ---------------------------------------------------------

y_retained = merged["retained_2021"]


# ---------------------------------------------------------
# Target 3: Winning Party
# Classification problem
# ---------------------------------------------------------

y_party = merged["result_2021"]


print("\nFeature columns:")
print(feature_cols)

print("\nX shape:")
print(X.shape)

print("\nMargin target shape:")
print(y_margin.shape)

print("\nRetention target shape:")
print(y_retained.shape)

print("\nParty target shape:")
print(y_party.shape)


# =========================================================
# STEP 9: TRAIN / TEST SPLIT
# =========================================================

(
    X_train,
    X_test,
    y_margin_train,
    y_margin_test,
    y_retained_train,
    y_retained_test,
    y_party_train,
    y_party_test
) = train_test_split(
    X,
    y_margin,
    y_retained,
    y_party,
    test_size=0.2,
    random_state=42
)


# =========================================================
# STEP 10: SAVE TEST DATA
# =========================================================

# Save constituency IDs separately so that they can be
# attached back to prediction results later.

id_train = merged.loc[
    X_train.index,
    "constituency_id"
]

id_test = merged.loc[
    X_test.index,
    "constituency_id"
]


id_train.to_csv(
    DATA_DIR / "id_train.csv",
    index=False
)


id_test.to_csv(
    DATA_DIR / "id_test.csv",
    index=False
)


# Save feature datasets
X_train.to_csv(
    DATA_DIR / "X_train.csv",
    index=False
)


X_test.to_csv(
    DATA_DIR / "X_test.csv",
    index=False
)


# Save test targets
y_margin_test.to_csv(
    DATA_DIR / "y_test_margin.csv",
    index=False
)


y_retained_test.to_csv(
    DATA_DIR / "y_test_retained.csv",
    index=False
)


y_party_test.to_csv(
    DATA_DIR / "y_test_party.csv",
    index=False
)


print(
    "\nSaved training/testing datasets to:",
    DATA_DIR
)


# =========================================================
# STEP 11: TRAIN RANDOM FOREST MARGIN MODEL
# =========================================================

print("\n========================================")
print("TRAINING MARGIN MODEL")
print("========================================")


rf_margin = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)


rf_margin.fit(
    X_train,
    y_margin_train
)


margin_predictions = rf_margin.predict(
    X_test
)


margin_r2 = r2_score(
    y_margin_test,
    margin_predictions
)


print(
    "Margin R2:",
    round(margin_r2, 4)
)


# =========================================================
# STEP 12: TRAIN RETENTION MODEL
# =========================================================

print("\n========================================")
print("TRAINING RETENTION MODEL")
print("========================================")


rf_retained = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)


rf_retained.fit(
    X_train,
    y_retained_train
)


retained_predictions = rf_retained.predict(
    X_test
)


retained_accuracy = accuracy_score(
    y_retained_test,
    retained_predictions
)


print(
    "Retained accuracy:",
    round(retained_accuracy, 4)
)


# =========================================================
# STEP 13: TRAIN WINNING PARTY MODEL
# =========================================================

print("\n========================================")
print("TRAINING PARTY MODEL")
print("========================================")


rf_party = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


rf_party.fit(
    X_train,
    y_party_train
)


party_predictions = rf_party.predict(
    X_test
)


party_accuracy = accuracy_score(
    y_party_test,
    party_predictions
)


print(
    "Party prediction accuracy:",
    round(party_accuracy, 4)
)


print("\nParty Classification Report:")

print(
    classification_report(
        y_party_test,
        party_predictions
    )
)


# =========================================================
# STEP 14: SAVE TRAINED MODELS
# =========================================================

joblib.dump(
    rf_margin,
    MODEL_DIR / "rf_margin_model.pkl"
)


joblib.dump(
    rf_retained,
    MODEL_DIR / "rf_retained_model.pkl"
)


joblib.dump(
    rf_party,
    MODEL_DIR / "rf_party_model.pkl"
)


# =========================================================
# STEP 15: SAVE STATE ENCODER AS JSON
# =========================================================

# LabelEncoder internally contains:
#
# state_encoder.classes_
#
# Example:
#
# ["Andhra Pradesh", "Assam", "Bihar", ...]
#
# We convert it into:
#
# {
#     "Andhra Pradesh": 0,
#     "Assam": 1,
#     "Bihar": 2
# }


state_map = {
    label: int(index)
    for index, label
    in enumerate(state_encoder.classes_)
}


with open(
    MODEL_DIR / "state_encoder.json",
    "w"
) as f:

    json.dump(
        state_map,
        f,
        indent=2
    )


# =========================================================
# STEP 16: SAVE PARTY ENCODER AS JSON
# =========================================================

party_map = {
    label: int(index)
    for index, label
    in enumerate(party_encoder.classes_)
}


with open(
    MODEL_DIR / "party_encoder.json",
    "w"
) as f:

    json.dump(
        party_map,
        f,
        indent=2
    )


# =========================================================
# STEP 17: SAVE MERGED DATASET
# =========================================================

# This is useful for inspection and debugging.

merged.to_csv(
    DATA_DIR / "merged_election_data.csv",
    index=False
)


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n========================================")
print("MODEL TRAINING COMPLETED")
print("========================================")

print("\nModels saved:")

print(
    MODEL_DIR / "rf_margin_model.pkl"
)

print(
    MODEL_DIR / "rf_retained_model.pkl"
)

print(
    MODEL_DIR / "rf_party_model.pkl"
)

print("\nEncoders saved:")

print(
    MODEL_DIR / "state_encoder.json"
)

print(
    MODEL_DIR / "party_encoder.json"
)

print("\nMerged dataset saved:")

print(
    DATA_DIR / "merged_election_data.csv"
)

print("\nTraining completed successfully.")