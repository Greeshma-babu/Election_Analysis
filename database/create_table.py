import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

load_dotenv()


def create_table():

    connection = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "election_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD")
    )

    cursor = connection.cursor()

    create_query = """
    CREATE TABLE IF NOT EXISTS legislative_election_data (
        id SERIAL PRIMARY KEY,

        election_year INTEGER,
        constituency VARCHAR(150),
        district VARCHAR(100),
        party VARCHAR(100),

        votes INTEGER,
        vote_share NUMERIC(10, 4),

        margin_of_victory INTEGER,
        turnout_percentage NUMERIC(10, 4),

        swing_factor NUMERIC(10, 4),

        urban_rural VARCHAR(20),

        incumbent_retained_2016 INTEGER,
        incumbent_retained_2021 INTEGER,

        previous_vote_share NUMERIC(10, 4),

        target_winner INTEGER
    );
    """

    cursor.execute(create_query)

    connection.commit()

    cursor.close()
    connection.close()

    print("Table created successfully.")


if __name__ == "__main__":
    create_table()