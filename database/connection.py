import os
import psycopg2
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():

    try:

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        print("Database connection established successfully.")

        return conn

    except psycopg2.Error as e:

        print(f"Database connection failed: {e}")

        return None


# =========================================================
# CREATE TABLE
# =========================================================

def create_election_table():

    conn = get_connection()

    if conn is None:
        return False

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

        print("Election table created/verified successfully.")

        return True

    except psycopg2.Error as e:

        conn.rollback()

        print(f"Unable to create election table: {e}")

        return False

    finally:

        conn.close()


# =========================================================
# INSERT / UPDATE ELECTION DATA
# =========================================================

def insert_election_data(df):

    conn = get_connection()

    if conn is None:
        return False

    try:

        with conn.cursor() as cursor:

            insert_sql = """

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

                    result_2011 = EXCLUDED.result_2011,

                    result_2016 = EXCLUDED.result_2016,

                    result_2021 = EXCLUDED.result_2021,

                    turnout_2011 = EXCLUDED.turnout_2011,

                    turnout_2016 = EXCLUDED.turnout_2016,

                    turnout_2021 = EXCLUDED.turnout_2021,

                    turnout_change_16 = EXCLUDED.turnout_change_16,

                    turnout_change_21 = EXCLUDED.turnout_change_21,

                    margin_2011 = EXCLUDED.margin_2011,

                    margin_2016 = EXCLUDED.margin_2016,

                    margin_2021 = EXCLUDED.margin_2021,

                    swing_2011 = EXCLUDED.swing_2011,

                    swing_2016 = EXCLUDED.swing_2016,

                    swing_2021 = EXCLUDED.swing_2021

            """

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

                cursor.execute(
                    insert_sql,
                    data
                )

        conn.commit()

        print(
            f"{len(df)} election records "
            f"inserted/updated successfully."
        )

        return True

    except psycopg2.Error as e:

        conn.rollback()

        print(f"Database insertion failed: {e}")

        return False

    finally:

        conn.close()