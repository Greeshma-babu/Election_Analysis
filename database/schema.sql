CREATE TABLE election_data (
    id SERIAL PRIMARY KEY,
    date_time TIMESTAMP,
    car_count INTEGER,
    scooter_count INTEGER,
    overspeed_count INTEGER
);