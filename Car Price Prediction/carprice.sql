CREATE DATABASE car_prediction;

USE car_prediction;


DROP DATABASE car_prediction;


CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    make_year INT,
    mileage_kmpl FLOAT,
    engine_cc FLOAT,
    fuel_type VARCHAR(50),
    brand VARCHAR(50),
    transmission INT,
    service_history INT,
    accidents_reported INT,
    insurance_valid INT,
    buyer_type INT,
    predicted_price_usd FLOAT
);

select * from predictions;



