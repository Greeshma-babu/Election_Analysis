import os
import psycopg2
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from visualization.election_charts import ElectionCharts


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    layout="wide",
    page_title="Election Analysis",
    page_icon="🗳️"
)


# =========================================================
# FILE PATH
# =========================================================

file_path = "dataset/training/merged_election_data.csv"


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Unable to read election data: {e}")
    st.stop()


# =========================================================
# NORMALIZE COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


# =========================================================
# REMOVE DUPLICATES
# =========================================================

if "constituency_id" not in df.columns:
    st.error(
        "The CSV does not contain the required column: constituency_id"
    )
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df.drop_duplicates(
    subset="constituency_id",
    inplace=True
)


# =========================================================
# REMOVE EXTRA SPACES FROM STRING COLUMNS
# =========================================================

string_columns = df.select_dtypes(
    include="object"
).columns

for column in string_columns:
    df[column] = df[column].astype(str).str.strip()


# =========================================================
# STANDARDIZE POSSIBLE COLUMN NAMES
#
# This allows the application to work if the CSV contains
# either the old names or the standardized names.
# =========================================================

column_mapping = {

    # Turnout
    "voter_turnout_2011_pct": "turnout_2011",
    "voter_turnout_2016_pct": "turnout_2016",
    "voter_turnout_2021_pct": "turnout_2021",

    # Turnout changes
    "turnout_change_2016_pct": "turnout_change_16",
    "turnout_change_2021_pct": "turnout_change_21",

    # Margins
    "margin_of_victory_2011_pct": "margin_2011",
    "margin_of_victory_2016_pct": "margin_2016",
    "margin_of_victory_2021_pct": "margin_2021",

    # Swing
    "swing_factor_2011_pct": "swing_2011",
    "swing_factor_2016_pct": "swing_2016",
    "swing_factor_2021_pct": "swing_2021",
}

df.rename(
    columns=column_mapping,
    inplace=True
)


# =========================================================
# REQUIRED BASE COLUMNS
# =========================================================

required_base_columns = [

    "constituency_id",
    "state",
    "demographic",

    "result_2011",
    "result_2016",
    "result_2021",

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


missing_base_columns = [
    column
    for column in required_base_columns
    if column not in df.columns
]


if missing_base_columns:

    st.error(
        "The election CSV is missing required columns:"
    )

    for column in missing_base_columns:
        st.write(f"- {column}")

    st.write("")

    st.write(
        "Columns found in CSV:"
    )

    st.write(df.columns.tolist())

    st.stop()


# =========================================================
# CONVERT NUMERIC COLUMNS
#
# IMPORTANT:
# result_2011/result_2016/result_2021 are NOT here
# because they contain party names.
# =========================================================

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

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# =========================================================
# CALCULATE TURNOUT CHANGE
# =========================================================

if "turnout_change_16" not in df.columns:

    df["turnout_change_16"] = (
        df["turnout_2016"]
        - df["turnout_2011"]
    )


if "turnout_change_21" not in df.columns:

    df["turnout_change_21"] = (
        df["turnout_2021"]
        - df["turnout_2016"]
    )


# =========================================================
# HANDLE NaN VALUES BEFORE POSTGRESQL INSERT
# =========================================================

df = df.where(
    pd.notnull(df),
    None
)


print(
    f"Data cleaning completed successfully. "
    f"Rows: {len(df)}"
)


# =========================================================
# POSTGRESQL CONNECTION
# =========================================================

load_dotenv()


try:

    conn = psycopg2.connect(

        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    print(
        "Database connection established successfully."
    )

except Exception as e:

    st.error(
        f"Database connection failed: {e}"
    )

    st.stop()


# =========================================================
# CREATE TABLE
#
# IMPORTANT:
# We DO NOT DROP the table every Streamlit rerun.
# =========================================================

try:

    with conn.cursor() as cursor:

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS election_constituency (

                constituency_id VARCHAR(50) PRIMARY KEY,

                state VARCHAR(255),

                demographic TEXT,

                result_2011 VARCHAR(255),
                result_2016 VARCHAR(255),
                result_2021 VARCHAR(255),

                turnout_2011 DECIMAL(10, 2),
                turnout_2016 DECIMAL(10, 2),
                turnout_2021 DECIMAL(10, 2),

                turnout_change_16 DECIMAL(10, 2),
                turnout_change_21 DECIMAL(10, 2),

                margin_2011 DECIMAL(10, 2),
                margin_2016 DECIMAL(10, 2),
                margin_2021 DECIMAL(10, 2),

                swing_2011 DECIMAL(10, 2),
                swing_2016 DECIMAL(10, 2),
                swing_2021 DECIMAL(10, 2)

            )
        """)

        conn.commit()


except Exception as e:

    conn.rollback()

    st.error(
        f"Unable to create database table: {e}"
    )

    conn.close()

    st.stop()


# =========================================================
# INSERT / UPDATE DATA
#
# ON CONFLICT prevents duplicate constituency errors.
# =========================================================

try:

    with conn.cursor() as cursor:

        for _, row in df.iterrows():

            data = (

                row["constituency_id"],
                row["state"],
                row["demographic"],

                row["result_2011"],
                row["result_2016"],
                row["result_2021"],

                row["turnout_2011"],
                row["turnout_2016"],
                row["turnout_2021"],

                row["turnout_change_16"],
                row["turnout_change_21"],

                row["margin_2011"],
                row["margin_2016"],
                row["margin_2021"],

                row["swing_2011"],
                row["swing_2016"],
                row["swing_2021"]
            )


            cursor.execute("""

                INSERT INTO election_constituency (

                    constituency_id,
                    state,
                    demographic,

                    result_2011,
                    result_2016,
                    result_2021,

                    turnout_2011,
                    turnout_2016,
                    turnout_2021,

                    turnout_change_16,
                    turnout_change_21,

                    margin_2011,
                    margin_2016,
                    margin_2021,

                    swing_2011,
                    swing_2016,
                    swing_2021

                )

                VALUES (

                    %s, %s, %s,

                    %s, %s, %s,

                    %s, %s, %s,

                    %s, %s,

                    %s, %s, %s,

                    %s, %s, %s

                )

                ON CONFLICT (constituency_id)

                DO UPDATE SET

                    state = EXCLUDED.state,

                    demographic = EXCLUDED.demographic,

                    result_2011 =
                        EXCLUDED.result_2011,

                    result_2016 =
                        EXCLUDED.result_2016,

                    result_2021 =
                        EXCLUDED.result_2021,

                    turnout_2011 =
                        EXCLUDED.turnout_2011,

                    turnout_2016 =
                        EXCLUDED.turnout_2016,

                    turnout_2021 =
                        EXCLUDED.turnout_2021,

                    turnout_change_16 =
                        EXCLUDED.turnout_change_16,

                    turnout_change_21 =
                        EXCLUDED.turnout_change_21,

                    margin_2011 =
                        EXCLUDED.margin_2011,

                    margin_2016 =
                        EXCLUDED.margin_2016,

                    margin_2021 =
                        EXCLUDED.margin_2021,

                    swing_2011 =
                        EXCLUDED.swing_2011,

                    swing_2016 =
                        EXCLUDED.swing_2016,

                    swing_2021 =
                        EXCLUDED.swing_2021

            """, data)


        conn.commit()


    print(
        f"{len(df)} cleaned records inserted/updated successfully."
    )


except Exception as e:

    conn.rollback()

    st.error(
        f"Database insertion failed: {e}"
    )

    conn.close()

    st.stop()


# =========================================================
# CLOSE DATABASE CONNECTION
# =========================================================

conn.close()


# =========================================================
# DATAFRAME FOR STREAMLIT CHARTS
# =========================================================
#
# At this point the DataFrame already uses the names expected
# by ElectionCharts.
#
# Therefore we DO NOT rename voter_turnout_2016_pct etc.
# again.
# =========================================================

df_for_charts = df.copy()


# =========================================================
# VERIFY CHART COLUMNS
# =========================================================

chart_required_columns = [

    "state",
    "demographic",

    "turnout_2016",
    "turnout_2021",

    "turnout_change_21",

    "margin_2016",

    "swing_2016",

    "result_2016"
]


missing_chart_columns = [

    column
    for column in chart_required_columns
    if column not in df_for_charts.columns

]


if missing_chart_columns:

    st.error(
        "Data is missing required columns for charts:"
    )

    for column in missing_chart_columns:
        st.write(f"- {column}")

    st.write("Available columns:")

    st.write(
        df_for_charts.columns.tolist()
    )

    st.stop()


# =========================================================
# CREATE ELECTION CHARTS OBJECT
# =========================================================

charts = ElectionCharts(
    df_for_charts
)


# =========================================================
# TOP SECTION
#
# Seats Won by Party | Turnout Scenario Simulator
# =========================================================

left_col, right_col = st.columns(
    [1, 2]
)


# =========================================================
# LEFT COLUMN
# SEATS WON BY PARTY
# =========================================================

with left_col:

    charts.plot_seats_won_by_party()


# =========================================================
# RIGHT COLUMN
# TURNOUT SCENARIO SIMULATOR
# =========================================================

with right_col:

    charts.plot_turnout_scenario()