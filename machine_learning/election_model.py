import pathlib
import json
import joblib

import pandas as pd

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

# election_model.py is inside:
#
# Election_Analysis/
#     machine_learning/
#         election_model.py
#
# parent.parent = Election_Analysis/

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "training"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CURRENT MODEL FEATURE COLUMNS
# =========================================================

FEATURE_COLS = [
    "state_encoded",
    "demographic_encoded",
    "turnout_2016",
    "turnout_2021",
    "turnout_change_21",
    "margin_2016",
    "swing_2016",
    "result_2016_encoded",
]


# =========================================================
# HISTORICAL BACKTEST FEATURE COLUMNS
# =========================================================
#
# For historical backtesting:
#
# TRAIN:
#   Historical information available up to 2016
#
# TEST:
#   Predict 2021 using only information available up to 2016
#
# Therefore:
#
# Training features:
#   state
#   demographic
#   turnout_2011
#   turnout_2016
#   turnout_change_16
#   margin_2011
#   swing_2011
#   result_2011_encoded
#
# The same feature structure is used for the 2021
# validation data:
#
#   turnout_2016
#   turnout_2021
#   turnout_change_21
#   margin_2016
#   swing_2016
#   result_2016_encoded
#
# =========================================================

BACKTEST_FEATURE_COLS = [
    "state_encoded",
    "demographic_encoded",
    "turnout_previous",
    "turnout_current",
    "turnout_change",
    "margin_previous",
    "swing_previous",
    "result_previous_encoded",
]


# =========================================================
# DEMOGRAPHIC ENCODING
# =========================================================

DEMOGRAPHIC_MAP = {
    "Urban": 0,
    "Semi-Urban": 1,
    "Rural": 2,
}


# =========================================================
# LOAD ELECTION DATA
# =========================================================

def load_election_data():

    print("\n========================================")
    print("LOADING ELECTION DATA")
    print("========================================")

    file_2011 = DATA_DIR / "election_2011.csv"
    file_2016 = DATA_DIR / "election_2016.csv"
    file_2021 = DATA_DIR / "election_2021.csv"

    if not file_2011.exists():
        raise FileNotFoundError(
            f"Missing file: {file_2011}"
        )

    if not file_2016.exists():
        raise FileNotFoundError(
            f"Missing file: {file_2016}"
        )

    if not file_2021.exists():
        raise FileNotFoundError(
            f"Missing file: {file_2021}"
        )

    df_2011 = pd.read_csv(file_2011)
    df_2016 = pd.read_csv(file_2016)
    df_2021 = pd.read_csv(file_2021)

    print("2011 shape:", df_2011.shape)
    print("2016 shape:", df_2016.shape)
    print("2021 shape:", df_2021.shape)

    return df_2011, df_2016, df_2021


# =========================================================
# RENAME YEARLY COLUMNS
# =========================================================

def rename_year_columns(df, year):

    df = df.copy()

    rename_map = {
        "result": f"result_{year}",
        "voter_turnout_pct": f"turnout_{year}",
        "margin_of_victory_pct": f"margin_{year}",
        "swing_factor_pct": f"swing_{year}",
    }

    df = df.rename(columns=rename_map)

    df = df.drop(
        columns=["election_year"],
        errors="ignore"
    )

    return df


# =========================================================
# CLEAN YEARLY DATA
# =========================================================

def clean_yearly_data(df, year):

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    df = rename_year_columns(
        df,
        year
    )

    text_columns = [
        "constituency_id",
        "state",
        "demographic",
        f"result_{year}",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
            )

    numeric_columns = [
        f"turnout_{year}",
        f"margin_{year}",
        f"swing_{year}",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    if "constituency_id" in df.columns:

        df = df.drop_duplicates(
            subset=["constituency_id"]
        )

    return df


# =========================================================
# MERGE 2011 + 2016 + 2021
# =========================================================

def merge_election_data(
    df_2011,
    df_2016,
    df_2021
):

    print("\n========================================")
    print("PREPARING ELECTION DATA")
    print("========================================")

    df_2011 = clean_yearly_data(
        df_2011,
        2011
    )

    df_2016 = clean_yearly_data(
        df_2016,
        2016
    )

    df_2021 = clean_yearly_data(
        df_2021,
        2021
    )

    merged = df_2011.copy()

    df_2016_merge = df_2016.drop(
        columns=[
            "state",
            "demographic"
        ],
        errors="ignore"
    )

    df_2021_merge = df_2021.drop(
        columns=[
            "state",
            "demographic"
        ],
        errors="ignore"
    )

    merged = merged.merge(
        df_2016_merge,
        on="constituency_id",
        how="inner"
    )

    merged = merged.merge(
        df_2021_merge,
        on="constituency_id",
        how="inner"
    )

    merged = merged.drop_duplicates(
        subset=["constituency_id"]
    )

    for column in merged.select_dtypes(
        include="object"
    ).columns:

        merged[column] = (
            merged[column]
            .astype(str)
            .str.strip()
        )

    print(
        "Merged dataset shape:",
        merged.shape
    )

    return merged


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(merged):

    merged = merged.copy()

    # -----------------------------------------------------
    # Turnout change
    # -----------------------------------------------------

    merged["turnout_change_16"] = (
        merged["turnout_2016"]
        - merged["turnout_2011"]
    )

    merged["turnout_change_21"] = (
        merged["turnout_2021"]
        - merged["turnout_2016"]
    )

    # -----------------------------------------------------
    # Margin change
    # -----------------------------------------------------

    merged["margin_change_16"] = (
        merged["margin_2016"]
        - merged["margin_2011"]
    )

    merged["margin_change_21"] = (
        merged["margin_2021"]
        - merged["margin_2016"]
    )

    # -----------------------------------------------------
    # Seat flip
    # -----------------------------------------------------

    merged["seat_flip_16"] = (
        merged["result_2016"]
        != merged["result_2011"]
    ).astype(int)

    merged["seat_flip_21"] = (
        merged["result_2021"]
        != merged["result_2016"]
    ).astype(int)

    # -----------------------------------------------------
    # Retention target
    # -----------------------------------------------------

    merged["retained_2021"] = (
        merged["margin_2021"] > 5
    ).astype(int)

    return merged


# =========================================================
# ENCODE CATEGORICAL DATA
# =========================================================

def encode_categorical_data(merged):

    merged = merged.copy()

    # -----------------------------------------------------
    # DEMOGRAPHIC
    # -----------------------------------------------------

    if "demographic" not in merged.columns:

        raise ValueError(
            "Missing required column: demographic"
        )

    merged["demographic_encoded"] = (
        merged["demographic"]
        .map(DEMOGRAPHIC_MAP)
    )

    unknown_demographic = (
        merged.loc[
            merged["demographic_encoded"].isna(),
            "demographic"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown_demographic:

        raise ValueError(
            "Unknown demographic value(s): "
            + str(unknown_demographic)
        )

    # -----------------------------------------------------
    # STATE
    # -----------------------------------------------------

    if "state" not in merged.columns:

        raise ValueError(
            "Missing required column: state"
        )

    state_encoder = LabelEncoder()

    merged["state_encoded"] = (
        state_encoder.fit_transform(
            merged["state"]
        )
    )

    # -----------------------------------------------------
    # PARTY
    # -----------------------------------------------------

    if "result_2016" not in merged.columns:

        raise ValueError(
            "Missing required column: result_2016"
        )

    party_encoder = LabelEncoder()

    merged["result_2016_encoded"] = (
        party_encoder.fit_transform(
            merged["result_2016"]
        )
    )

    # -----------------------------------------------------
    # DISPLAY STATE ENCODING
    # -----------------------------------------------------

    print("\n========================================")
    print("STATE ENCODING")
    print("========================================")

    for index, label in enumerate(
        state_encoder.classes_
    ):

        print(
            f"{label} -> {index}"
        )

    # -----------------------------------------------------
    # DISPLAY PARTY ENCODING
    # -----------------------------------------------------

    print("\n========================================")
    print("PARTY ENCODING")
    print("========================================")

    for index, label in enumerate(
        party_encoder.classes_
    ):

        print(
            f"{label} -> {index}"
        )

    return (
        merged,
        state_encoder,
        party_encoder
    )


# =========================================================
# SAVE ENCODERS
# =========================================================

def save_encoders(
    state_encoder,
    party_encoder
):

    print("\n========================================")
    print("SAVING ENCODERS")
    print("========================================")

    # -----------------------------------------------------
    # STATE
    # -----------------------------------------------------

    state_map = {
        str(label): int(index)
        for index, label
        in enumerate(
            state_encoder.classes_
        )
    }

    state_encoder_file = (
        MODEL_DIR / "state_encoder.json"
    )

    with open(
        state_encoder_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state_map,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "State encoder saved:",
        state_encoder_file
    )

    # -----------------------------------------------------
    # PARTY
    # -----------------------------------------------------

    party_map = {
        str(label): int(index)
        for index, label
        in enumerate(
            party_encoder.classes_
        )
    }

    party_encoder_file = (
        MODEL_DIR / "party_encoder.json"
    )

    with open(
        party_encoder_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            party_map,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "Party encoder saved:",
        party_encoder_file
    )

    # -----------------------------------------------------
    # VERIFY
    # -----------------------------------------------------

    if not state_encoder_file.exists():

        raise FileNotFoundError(
            "State encoder JSON was not created: "
            + str(state_encoder_file)
        )

    if not party_encoder_file.exists():

        raise FileNotFoundError(
            "Party encoder JSON was not created: "
            + str(party_encoder_file)
        )

    print(
        "\nEncoder files verified successfully."
    )


# =========================================================
# VALIDATE FEATURES
# =========================================================

def validate_features(merged):

    missing_columns = [
        column
        for column in FEATURE_COLS
        if column not in merged.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing feature columns: "
            + ", ".join(missing_columns)
        )

    for column in FEATURE_COLS:

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce"
        )

    missing_values = (
        merged[
            FEATURE_COLS
        ]
        .isnull()
        .sum()
    )

    missing_features = (
        missing_values[
            missing_values > 0
        ]
    )

    if not missing_features.empty:

        raise ValueError(
            "Missing/invalid values found:\n"
            + str(missing_features)
        )

    return merged


# =========================================================
# HISTORICAL BACKTESTING
# =========================================================
#
# This evaluates the model using historical progression.
#
# TRAIN:
#   2011 -> 2016
#
# VALIDATE:
#   2016 -> 2021
#
# This is different from the normal random train/test split.
#
# It answers:
#
# "If we had trained using information available up to
#  2016, how well could we predict the 2021 election?"
#
# =========================================================

def run_historical_backtesting(merged):

    print("\n========================================")
    print("HISTORICAL BACKTESTING")
    print("========================================")

    backtest_df = merged.copy()

    # -----------------------------------------------------
    # Build training features from 2011 -> 2016
    # -----------------------------------------------------

    train_df = pd.DataFrame(index=backtest_df.index)

    train_df["state_encoded"] = (
        backtest_df["state_encoded"]
    )

    train_df["demographic_encoded"] = (
        backtest_df["demographic_encoded"]
    )

    train_df["turnout_previous"] = (
        backtest_df["turnout_2011"]
    )

    train_df["turnout_current"] = (
        backtest_df["turnout_2016"]
    )

    train_df["turnout_change"] = (
        backtest_df["turnout_change_16"]
    )

    train_df["margin_previous"] = (
        backtest_df["margin_2011"]
    )

    train_df["swing_previous"] = (
        backtest_df["swing_2011"]
    )

    # -----------------------------------------------------
    # Encode 2011 result using the same party mapping
    # -----------------------------------------------------

    party_encoder = LabelEncoder()

    all_parties = pd.concat(
        [
            backtest_df["result_2011"],
            backtest_df["result_2016"],
            backtest_df["result_2021"],
        ],
        ignore_index=True
    ).astype(str)

    party_encoder.fit(all_parties)

    train_df["result_previous_encoded"] = (
        party_encoder.transform(
            backtest_df["result_2011"].astype(str)
        )
    )

    # -----------------------------------------------------
    # Test features:
    #
    # Use only information that would have been known
    # after the 2016 election.
    # -----------------------------------------------------

    test_df = pd.DataFrame(index=backtest_df.index)

    test_df["state_encoded"] = (
        backtest_df["state_encoded"]
    )

    test_df["demographic_encoded"] = (
        backtest_df["demographic_encoded"]
    )

    test_df["turnout_previous"] = (
        backtest_df["turnout_2016"]
    )

    test_df["turnout_current"] = (
        backtest_df["turnout_2021"]
    )

    test_df["turnout_change"] = (
        backtest_df["turnout_change_21"]
    )

    test_df["margin_previous"] = (
        backtest_df["margin_2016"]
    )

    test_df["swing_previous"] = (
        backtest_df["swing_2016"]
    )

    test_df["result_previous_encoded"] = (
        party_encoder.transform(
            backtest_df["result_2016"].astype(str)
        )
    )

    # -----------------------------------------------------
    # Targets
    # -----------------------------------------------------

    y_margin_train = (
        backtest_df["margin_2016"]
    )

    y_margin_test = (
        backtest_df["margin_2021"]
    )

    y_retained_train = (
        backtest_df["margin_2016"] > 5
    ).astype(int)

    y_retained_test = (
        backtest_df["margin_2021"] > 5
    ).astype(int)

    y_party_train = (
        backtest_df["result_2016"]
        .astype(str)
    )

    y_party_test = (
        backtest_df["result_2021"]
        .astype(str)
    )

    # -----------------------------------------------------
    # Remove invalid rows
    # -----------------------------------------------------

    valid_train = (
        train_df[
            BACKTEST_FEATURE_COLS
        ]
        .notna()
        .all(axis=1)
    )

    valid_test = (
        test_df[
            BACKTEST_FEATURE_COLS
        ]
        .notna()
        .all(axis=1)
    )

    train_df = train_df.loc[
        valid_train
    ].copy()

    test_df = test_df.loc[
        valid_test
    ].copy()

    y_margin_train = (
        y_margin_train.loc[
            train_df.index
        ]
    )

    y_margin_test = (
        y_margin_test.loc[
            test_df.index
        ]
    )

    y_retained_train = (
        y_retained_train.loc[
            train_df.index
        ]
    )

    y_retained_test = (
        y_retained_test.loc[
            test_df.index
        ]
    )

    y_party_train = (
        y_party_train.loc[
            train_df.index
        ]
    )

    y_party_test = (
        y_party_test.loc[
            test_df.index
        ]
    )

    # -----------------------------------------------------
    # MARGIN BACKTEST
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("BACKTEST: MARGIN")
    print("----------------------------------------")

    backtest_margin_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    backtest_margin_model.fit(
        train_df[
            BACKTEST_FEATURE_COLS
        ],
        y_margin_train
    )

    margin_predictions = (
        backtest_margin_model.predict(
            test_df[
                BACKTEST_FEATURE_COLS
            ]
        )
    )

    backtest_margin_r2 = r2_score(
        y_margin_test,
        margin_predictions
    )

    print(
        "Historical Margin R2:",
        round(backtest_margin_r2, 4)
    )

    # -----------------------------------------------------
    # RETENTION BACKTEST
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("BACKTEST: RETENTION")
    print("----------------------------------------")

    backtest_retained_model = (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
    )

    backtest_retained_model.fit(
        train_df[
            BACKTEST_FEATURE_COLS
        ],
        y_retained_train
    )

    retained_predictions = (
        backtest_retained_model.predict(
            test_df[
                BACKTEST_FEATURE_COLS
            ]
        )
    )

    backtest_retained_accuracy = (
        accuracy_score(
            y_retained_test,
            retained_predictions
        )
    )

    print(
        "Historical Retention Accuracy:",
        round(
            backtest_retained_accuracy,
            4
        )
    )

    # -----------------------------------------------------
    # PARTY BACKTEST
    # -----------------------------------------------------

    print("\n----------------------------------------")
    print("BACKTEST: PARTY")
    print("----------------------------------------")

    backtest_party_model = (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    )

    backtest_party_model.fit(
        train_df[
            BACKTEST_FEATURE_COLS
        ],
        y_party_train
    )

    party_predictions = (
        backtest_party_model.predict(
            test_df[
                BACKTEST_FEATURE_COLS
            ]
        )
    )

    backtest_party_accuracy = (
        accuracy_score(
            y_party_test,
            party_predictions
        )
    )

    print(
        "Historical Party Accuracy:",
        round(
            backtest_party_accuracy,
            4
        )
    )

    # -----------------------------------------------------
    # PARTY REPORT
    # -----------------------------------------------------

    print("\n========================================")
    print("HISTORICAL PARTY CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_party_test,
            party_predictions,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # CREATE RESULT
    # -----------------------------------------------------

    result = {
        "backtest_period": "2011-2016 -> 2021",
        "training_period": "2011-2016",
        "validation_period": "2021",
        "validation_rows": int(
            len(test_df)
        ),
        "margin_r2": float(
            backtest_margin_r2
        ),
        "retention_accuracy": float(
            backtest_retained_accuracy
        ),
        "party_accuracy": float(
            backtest_party_accuracy
        ),
    }

    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------

    backtest_file = (
        MODEL_DIR
        / "historical_backtesting.json"
    )

    with open(
        backtest_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    print("\nBacktesting results saved:")
    print(backtest_file)

    return result


# =========================================================
# SAVE TRAINING DATA
# =========================================================

def save_training_data(
    merged,
    X_train,
    X_test,
    y_margin_test,
    y_retained_test,
    y_party_test
):

    print("\n========================================")
    print("SAVING TRAINING DATA")
    print("========================================")

    merged.to_csv(
        DATA_DIR / "merged_election_data.csv",
        index=False
    )

    X_train.to_csv(
        DATA_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        DATA_DIR / "X_test.csv",
        index=False
    )

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
        "\nTraining/testing datasets saved to:"
    )

    print(DATA_DIR)


# =========================================================
# TRAIN MODELS
# =========================================================

def train_models():

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    (
        df_2011,
        df_2016,
        df_2021
    ) = load_election_data()

    # -----------------------------------------------------
    # MERGE
    # -----------------------------------------------------

    merged = merge_election_data(
        df_2011,
        df_2016,
        df_2021
    )

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    merged = create_features(
        merged
    )

    # -----------------------------------------------------
    # ENCODE
    # -----------------------------------------------------

    (
        merged,
        state_encoder,
        party_encoder
    ) = encode_categorical_data(
        merged
    )

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    merged = validate_features(
        merged
    )

    # -----------------------------------------------------
    # CREATE X / TARGETS
    # -----------------------------------------------------

    X = merged[
        FEATURE_COLS
    ].copy()

    y_margin = (
        merged["margin_2021"].copy()
    )

    y_retained = (
        merged["retained_2021"].copy()
    )

    y_party = (
        merged["result_2021"].copy()
    )

    # -----------------------------------------------------
    # FEATURE INFORMATION
    # -----------------------------------------------------

    print("\n========================================")
    print("FEATURE INFORMATION")
    print("========================================")

    print("\nFeatures:")

    for feature in FEATURE_COLS:

        print(
            f"  - {feature}"
        )

    print(
        "\nX shape:",
        X.shape
    )

    print(
        "Margin target:",
        y_margin.shape
    )

    print(
        "Retention target:",
        y_retained.shape
    )

    print(
        "Party target:",
        y_party.shape
    )

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_margin_train,
        y_margin_test,
        y_retained_train,
        y_retained_test,
        y_party_train,
        y_party_test,
    ) = train_test_split(
        X,
        y_margin,
        y_retained,
        y_party,
        test_size=0.20,
        random_state=42,
        stratify=y_party
    )

    # -----------------------------------------------------
    # SPLIT INFORMATION
    # -----------------------------------------------------

    print("\n========================================")
    print("TRAIN / TEST SPLIT")
    print("========================================")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "Margin train:",
        y_margin_train.shape
    )

    print(
        "Margin test:",
        y_margin_test.shape
    )

    print(
        "Retention train:",
        y_retained_train.shape
    )

    print(
        "Retention test:",
        y_retained_test.shape
    )

    print(
        "Party train:",
        y_party_train.shape
    )

    print(
        "Party test:",
        y_party_test.shape
    )

    # =====================================================
    # TRAIN MARGIN MODEL
    # =====================================================

    print("\n========================================")
    print("TRAINING MARGIN MODEL")
    print("========================================")

    rf_margin = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    rf_margin.fit(
        X_train,
        y_margin_train
    )

    margin_predictions = (
        rf_margin.predict(
            X_test
        )
    )

    margin_r2 = r2_score(
        y_margin_test,
        margin_predictions
    )

    print(
        "Margin R2:",
        round(
            margin_r2,
            4
        )
    )

    # =====================================================
    # TRAIN RETENTION MODEL
    # =====================================================

    print("\n========================================")
    print("TRAINING RETENTION MODEL")
    print("========================================")

    rf_retained = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    rf_retained.fit(
        X_train,
        y_retained_train
    )

    retained_predictions = (
        rf_retained.predict(
            X_test
        )
    )

    retained_accuracy = (
        accuracy_score(
            y_retained_test,
            retained_predictions
        )
    )

    print(
        "Retention accuracy:",
        round(
            retained_accuracy,
            4
        )
    )

    # =====================================================
    # TRAIN PARTY MODEL
    # =====================================================

    print("\n========================================")
    print("TRAINING PARTY MODEL")
    print("========================================")

    rf_party = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    rf_party.fit(
        X_train,
        y_party_train
    )

    party_predictions = (
        rf_party.predict(
            X_test
        )
    )

    party_accuracy = (
        accuracy_score(
            y_party_test,
            party_predictions
        )
    )

    print(
        "Party prediction accuracy:",
        round(
            party_accuracy,
            4
        )
    )

    # =====================================================
    # PARTY CLASSIFICATION REPORT
    # =====================================================

    print("\n========================================")
    print("PARTY CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_party_test,
            party_predictions,
            zero_division=0
        )
    )

    # =====================================================
    # SAVE MODELS
    # =====================================================

    print("\n========================================")
    print("SAVING MODELS")
    print("========================================")

    margin_model_file = (
        MODEL_DIR / "rf_margin_model.pkl"
    )

    retained_model_file = (
        MODEL_DIR / "rf_retained_model.pkl"
    )

    party_model_file = (
        MODEL_DIR / "rf_party_model.pkl"
    )

    joblib.dump(
        rf_margin,
        margin_model_file
    )

    joblib.dump(
        rf_retained,
        retained_model_file
    )

    joblib.dump(
        rf_party,
        party_model_file
    )

    print(
        "Margin model:",
        margin_model_file
    )

    print(
        "Retention model:",
        retained_model_file
    )

    print(
        "Party model:",
        party_model_file
    )

    # =====================================================
    # SAVE ENCODERS
    # =====================================================

    save_encoders(
        state_encoder,
        party_encoder
    )

    # =====================================================
    # SAVE TRAINING DATA
    # =====================================================

    save_training_data(
        merged,
        X_train,
        X_test,
        y_margin_test,
        y_retained_test,
        y_party_test
    )

    # =====================================================
    # HISTORICAL BACKTESTING
    # =====================================================

    backtest_results = (
        run_historical_backtesting(
            merged
        )
    )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n========================================")
    print("MODEL TRAINING COMPLETED")
    print("========================================")

    print(
        f"Margin R2       : {margin_r2:.4f}"
    )

    print(
        f"Retention Acc.  : "
        f"{retained_accuracy:.4f}"
    )

    print(
        f"Party Accuracy  : "
        f"{party_accuracy:.4f}"
    )

    print("\n----------------------------------------")
    print("HISTORICAL BACKTEST RESULTS")
    print("----------------------------------------")

    print(
        f"Backtest Margin R2       : "
        f"{backtest_results['margin_r2']:.4f}"
    )

    print(
        f"Backtest Retention Acc.  : "
        f"{backtest_results['retention_accuracy']:.4f}"
    )

    print(
        f"Backtest Party Accuracy  : "
        f"{backtest_results['party_accuracy']:.4f}"
    )

    print("\nGenerated model files:")

    print(
        f"  {margin_model_file}"
    )

    print(
        f"  {retained_model_file}"
    )

    print(
        f"  {party_model_file}"
    )

    print("\nGenerated encoder files:")

    print(
        f"  {MODEL_DIR / 'state_encoder.json'}"
    )

    print(
        f"  {MODEL_DIR / 'party_encoder.json'}"
    )

    print("\nGenerated backtesting file:")

    print(
        f"  {MODEL_DIR / 'historical_backtesting.json'}"
    )

    print("\nGenerated training files:")

    print(
        f"  {DATA_DIR / 'merged_election_data.csv'}"
    )

    print(
        f"  {DATA_DIR / 'X_train.csv'}"
    )

    print(
        f"  {DATA_DIR / 'X_test.csv'}"
    )

    print(
        f"  {DATA_DIR / 'y_test_margin.csv'}"
    )

    print(
        f"  {DATA_DIR / 'y_test_retained.csv'}"
    )

    print(
        f"  {DATA_DIR / 'y_test_party.csv'}"
    )

    return {
        "margin_r2": margin_r2,
        "retention_accuracy": retained_accuracy,
        "party_accuracy": party_accuracy,
        "historical_backtesting": backtest_results
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_models()