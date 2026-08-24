import pandas as pd
import streamlit as st

from visualization.election_charts import ElectionCharts
from database.connection import create_election_table, insert_election_data


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
df.columns = df.columns.str.strip().str.lower()

# =========================================================
# REMOVE DUPLICATES
# =========================================================
if "constituency_id" not in df.columns:
    st.error("The CSV does not contain the required column: constituency_id")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df.drop_duplicates(subset="constituency_id", inplace=True)

# =========================================================
# REMOVE EXTRA SPACES FROM STRING COLUMNS
# =========================================================
string_columns = df.select_dtypes(include="object").columns

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

    # Turnout Changes
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

df.rename(columns=column_mapping, inplace=True)

# =========================================================
# REQUIRED BASE COLUMNS
# =========================================================
required_base_columns = [
    "constituency_id",
    "state",
    "demographic",
    "result_2011", "result_2016", "result_2021",
    "turnout_2011", "turnout_2016", "turnout_2021",
    "margin_2011", "margin_2016", "margin_2021",
    "swing_2011", "swing_2016", "swing_2021",
]

missing_base_columns = [
    column for column in required_base_columns if column not in df.columns
]

if missing_base_columns:
    st.error("The election CSV is missing required columns:")
    for column in missing_base_columns:
        st.write(f"- {column}")
    st.write("")
    st.write("Columns found in CSV:")
    st.write(df.columns.tolist())
    st.stop()

# =========================================================
# CONVERT NUMERIC COLUMNS
#
# IMPORTANT:
# result_2011/result_2016/result_2021 are NOT included
# because they contain party names.
# =========================================================
numeric_columns = [
    "turnout_2011", "turnout_2016", "turnout_2021",
    "margin_2011", "margin_2016", "margin_2021",
    "swing_2011", "swing_2016", "swing_2021",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# =========================================================
# CALCULATE TURNOUT CHANGE
# =========================================================
if "turnout_change_16" not in df.columns:
    df["turnout_change_16"] = df["turnout_2016"] - df["turnout_2011"]

if "turnout_change_21" not in df.columns:
    df["turnout_change_21"] = df["turnout_2021"] - df["turnout_2016"]

# =========================================================
# HANDLE NaN VALUES
#
# Convert pandas NaN values to None before sending the
# DataFrame to PostgreSQL.
# =========================================================
df = df.where(pd.notnull(df), None)

# =========================================================
# DATA CLEANING COMPLETE
# =========================================================
print(f"Data cleaning completed successfully. Rows: {len(df)}")

# =========================================================
# POSTGRESQL DATABASE
#
# IMPORTANT:
# app.py does NOT contain:
#
# - load_dotenv()
# - psycopg2.connect()
# - SQL CREATE TABLE
# - SQL INSERT
# - conn.close()
#
# All database operations are handled by
# database/connection.py.
# =========================================================

# Create table
table_created = create_election_table()

if not table_created:
    st.error("Unable to create or verify the election_constituency table.")
    st.stop()

# Insert / Update cleaned data
data_inserted = insert_election_data(df)

if not data_inserted:
    st.error("Unable to insert/update election data in PostgreSQL.")
    st.stop()

# =========================================================
# DATAFRAME FOR STREAMLIT CHARTS
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
    "result_2016",
]

missing_chart_columns = [
    column for column in chart_required_columns if column not in df_for_charts.columns
]

if missing_chart_columns:
    st.error("Data is missing required columns for charts:")
    for column in missing_chart_columns:
        st.write(f"- {column}")
    st.write("Available columns:")
    st.write(df_for_charts.columns.tolist())
    st.stop()

# =========================================================
# CREATE ELECTION CHARTS OBJECT
# =========================================================
charts = ElectionCharts(df_for_charts)

# =========================================================
# TOP SECTION
#
# Seats Won by Party | Turnout Scenario Simulator
# =========================================================
left_col, right_col = st.columns([1, 2])

# LEFT COLUMN - SEATS WON BY PARTY
with left_col:
    charts.plot_seats_won_by_party()

# RIGHT COLUMN - TURNOUT SCENARIO SIMULATOR
with right_col:
    charts.plot_turnout_scenario()