# Flight Delay Prediction

## Overview
This project leverages historical flight and weather data to **predict flight delays**. By integrating flight schedules with weather conditions, the model provides actionable insights for airlines and passengers, helping improve operational efficiency and passenger experience.

---

## Key Features
- **Data Integration:** Combines flight and weather datasets for enriched predictive insights.  
- **Data Processing & Cleaning:** Efficient handling of large datasets with robust preprocessing.  
- **Exploratory Data Analysis (EDA):** Identifies trends, patterns, and factors contributing to flight delays.  
- **Predictive Modeling:** Implements machine learning models to accurately forecast flight delays.

---

## Project Structure
flight-delay-prediction/
│
├── data/ # Raw and processed datasets (large CSVs excluded)
├── src/ # Python scripts for data processing, EDA, and modeling
│ ├── add_airport_to_weather.py
│ ├── app.py
│ ├── combine_weather.py
│ ├── create_merge_keys.py
│ ├── flight_delay_eda.py
│ ├── flights_with_date.py
│ ├── make_station_airport_map.py
│ ├── merge_flight_weather.py
│ ├── preview_data.py
│ └── train_delay_model.py
├── .gitignore # Excludes large files and unnecessary artifacts
└── README.md # Project overview and instructions


---

## Installation & Setup
1. **Clone the repository**
```bash
git clone https://github.com/Arymacore/Flight-Delay-Prediction-model.git
cd Flight-Delay-Prediction-model

##Install dependencies

pip install -r requirements.txt


(Ensure requirements.txt lists all packages used, e.g., pandas, numpy, scikit-learn.)

Usage

Run scripts in the src/ directory to process data, perform EDA, or train models.

Example:

python src/train_delay_model.py

Notes

Large datasets (all_weather.csv, all_weather_with_airport.csv) are excluded due to size constraints. Download separately if needed.

Ensure all dependencies are installed for smooth execution.

Author
Aryma Rawat (Arymacore)
