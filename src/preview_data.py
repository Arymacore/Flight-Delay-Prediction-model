import pandas as pd
df_weather=pd.read_csv('data/all_weather.csv')
print(df_weather.head())
print(df_weather.columns)
df_flight=pd.read_csv('data/Airline_Delay_Cause.csv')
print(df_flight.head())
print(df_flight.columns)
