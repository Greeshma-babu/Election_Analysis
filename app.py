import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import plotly.express as px


from visualization.election_charts import ElectionCharts
#from database.create_table import create_table

# --------------------------------------------------------
# Read the legislative assembly data from the Excel file,
#  clean it, and prepare it for database insertion.
# --------------------------------------------------------

file_path = "dataset/training/legislative_assembly_data.xlsx"
df = pd.read_excel(file_path)

# Drop duplicate rows based on the 'constituency_id' column
df.drop_duplicates(subset='constituency_id', inplace=True)

# Find any missing values in the DataFrame
missing_values = df.isnull().sum()
if(missing_values.any()):
    print("Missing values found in the following columns:")
    print(missing_values[missing_values > 0])

# Remove extra spaces from string columns
string_columns = df.select_dtypes(include="object").columns
for column in string_columns:
    df[column] = df[column].str.strip()

# Convert percentage columns to numeric
percentage_columns = [
    "voter_turnout_2011_pct",
    "voter_turnout_2016_pct",
    "voter_turnout_2021_pct",
    "turnout_change_2016_pct",
    "turnout_change_2021_pct",
    "margin_of_victory_2011_pct",
    "margin_of_victory_2016_pct",
    "margin_of_victory_2021_pct",
    "swing_factor_2016_pct",
    "swing_factor_2021_pct"
]

for column in percentage_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Data cleaning completed successfully.")

# -------------------------------------------------------
# Write the cleaned DataFrame to the PostgreSQL database.
# -------------------------------------------------------

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
if conn:
    print("Database connection established successfully.")

    # Clean up the database by dropping the table if it exists 
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS election_constituency;")
        conn.commit() 

    # Create the table in the database
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS election_constituency (
                constituency_id VARCHAR(20) PRIMARY KEY,
                state VARCHAR(255),
                demographic TEXT,
                result_2011 VARCHAR(255),
                result_2016 VARCHAR(255),
                result_2021 VARCHAR(255),
                voter_turnout_2011_pct DECIMAL(10, 2),
                voter_turnout_2016_pct DECIMAL(10, 2),
                voter_turnout_2021_pct DECIMAL(10, 2),
                turnout_change_2016_pct DECIMAL(10, 2),
                turnout_change_2021_pct DECIMAL(10, 2),
                margin_of_victory_2011_pct DECIMAL(10, 2),
                margin_of_victory_2016_pct DECIMAL(10, 2),
                margin_of_victory_2021_pct DECIMAL(10, 2),
                swing_factor_2016_pct DECIMAL(10, 2),
                swing_factor_2021_pct DECIMAL(10, 2),
                incumbent_retained_2016 INTEGER,
                incumbent_retained_2021 INTEGER
            )
        """)
        conn.commit()

    # Insert the cleaned data into the database
    with conn.cursor() as cursor:
        for index, row in df.iterrows():
            data = (
                row["constituency_id"],
                row["state"],
                row["demographic"],
                row["result_2011"],
                row["result_2016"],
                row["result_2021"],
                row["voter_turnout_2011_pct"],
                row["voter_turnout_2016_pct"],
                row["voter_turnout_2021_pct"],
                row["turnout_change_2016_pct"],
                row["turnout_change_2021_pct"],
                row["margin_of_victory_2011_pct"],
                row["margin_of_victory_2016_pct"],
                row["margin_of_victory_2021_pct"],
                row["swing_factor_2016_pct"],
                row["swing_factor_2021_pct"],
                row["incumbent_retained_2016"],
                row["incumbent_retained_2021"]
            )

            cursor.execute("""
                INSERT INTO election_constituency (
                    constituency_id,
                    state,
                    demographic,
                    result_2011,
                    result_2016,
                    result_2021,
                    voter_turnout_2011_pct,
                    voter_turnout_2016_pct,
                    voter_turnout_2021_pct,
                    turnout_change_2016_pct,
                    turnout_change_2021_pct,
                    margin_of_victory_2011_pct,
                    margin_of_victory_2016_pct,
                    margin_of_victory_2021_pct,
                    swing_factor_2016_pct,
                    swing_factor_2021_pct,
                    incumbent_retained_2016,
                    incumbent_retained_2021
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, data)
        conn.commit()
    print(f"{len(df)} cleaned records inserted successfully.")
conn.close()

# -------------------------------------------------------
# Interactive Streamlit Application for Election Analysis
# -------------------------------------------------------

charts = ElectionCharts(df)
charts.plot_seats_won_by_party()