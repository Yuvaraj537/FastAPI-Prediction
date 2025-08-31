CREATE DATABASE bitcoin_forecast ;
USE bitcoin_forecast;

CREATE TABLE  forecast_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ds DATE NOT NULL,                -- Date of observation / forecast
    y DOUBLE,                        -- Actual value (if available)
    yhat DOUBLE NOT NULL,            -- Forecasted value
    yhat_lower DOUBLE NOT NULL,      -- Lower bound of forecast
    yhat_upper DOUBLE NOT NULL    -- Upper bound of forecast
);

drop table forecast_data;

select * from forecast_data;