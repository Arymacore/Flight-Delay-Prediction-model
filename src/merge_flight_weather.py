import pandas as pd
weather=pd.read_csv('data/all_weather_with_airport.csv',low_memory=False)
flights=pd.read_csv('data/flights_with_date.csv')
merged=pd.merge(flights,weather,on=['airport','date'],how='inner')
merged.to_csv('outputs/merged_flights_weather.csv',index=False)
print('Merged flight + weather data saved to outputs/merged_flights_weather.csv')