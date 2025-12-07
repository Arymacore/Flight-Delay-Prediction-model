# Flight-Delay-Prediction-model
This project predicts whether flights at a given airport will be delayed by more than 15 minutes by combining airline delay statistics with local weather data, and presents the results in an interactive Streamlit dashboard.

Overview
Pipeline:
Collect and merge:
BTS airline delay statistics (monthly, by airport and carrier)
NOAA Local Climatological Data (hourly weather at airport stations)
Map NOAA weather stations to airports using geodesic distance.
Clean and explore the merged dataset (EDA).
Engineer a binary target delay_15 (delay ≥ 15 minutes).
Train a leakage‑free Random Forest classifier.
Visualize and interact with results in a Streamlit app.
The final model uses only pre‑outcome features (weather + airport/carrier), avoiding data leakage from actual delay values.

Project Structure
flight-delay-prediction/
├─ data/
│  ├─ Airline_Delay_Cause.csv          # BTS delay stats
│  ├─ <NOAA LCD CSV files>             # hourly weather
│  ├─ airports.csv                     # airport reference (IATA + coords)
│  ├─ all_weather.csv                  # combined weather (generated)
│  ├─ station_airport_map.csv          # station→airport mapping (generated)
│  ├─ all_weather_with_airport.csv     # weather + airport (generated)
│  ├─ flights_with_date.csv            # BTS with synthetic date (generated)
├─ outputs/
│  ├─ merged_flights_weather.csv       # merged BTS + weather
│  ├─ model_input.csv                  # cleaned features + target
├─ src/
│  ├─ combine_weather.py               # combine NOAA LCD files
│  ├─ make_station_airport_map.py      # station→airport mapping
│  ├─ add_airport_to_weather.py        # add airport/date to weather
│  ├─ flights_with_date.py             # add date to BTS data
│  ├─ merge_flight_weather.py          # merge BTS + weather
│  ├─ flight_delay_eda.py              # EDA + feature prep
│  ├─ train_delay_model.py             # Random Forest training/eval
│  ├─ app.py                           # Streamlit dashboard

Setup
git clone <your-repo-url>
cd flight-delay-prediction

# (optional) create a virtual env
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt  # or manually:
pip install pandas numpy scikit-learn matplotlib seaborn streamlit geopy
Data Pipeline
Run these from the project root to rebuild all derived files:

bash
python src/combine_weather.py
python src/make_station_airport_map.py
python src/add_airport_to_weather.py
python src/flights_with_date.py
python src/merge_flight_weather.py
python src/flight_delay_eda.py
This produces:

outputs/merged_flights_weather.csv

outputs/model_input.csv

Modeling
Train and evaluate the Random Forest model:

bash
python src/train_delay_model.py
The script:

Uses delay_15 as the target.

Uses weather + one‑hot encoded airport and carrier as features.

Does not use arr_delay as a feature (to avoid data leakage).

Prints classification report, confusion matrix, ROC‑AUC.

Plots top feature importances.

Dashboard
Launch the Streamlit app:

bash
streamlit run src/app.py
The dashboard provides:

Data preview.

Class balance of delay_15.

Filters for airport/carrier.

Live model metrics (accuracy, ROC‑AUC, classification report, confusion matrix).

Feature importance bar chart.

Scatter plots of delay vs temperature, wind speed, and precipitation.

Key Findings
Data is highly imbalanced: many more delayed than non‑delayed records.

With only pre‑flight features, Random Forest achieves ~0.89 ROC‑AUC.

Temperature and wind speed are the most important weather predictors.

Removing arr_delay from features was crucial to eliminate data leakage.

Limitations & Future Work
Current labels are based on monthly aggregates, not per‑flight data.

Dataset focuses on a single main airport; generalization is limited.

Future improvements:

Use flight‑level BTS on‑time data.

Add schedule features (time‑of‑day, day‑of‑week, holidays).

Address imbalance with resampling or class‑weighted models.

Compare with other models (logistic regression, gradient boosting, XGBoost).
