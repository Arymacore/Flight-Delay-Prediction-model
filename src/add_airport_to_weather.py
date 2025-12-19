import pandas as pd
weather=pd.read_csv('data/all_weather.csv',low_memory=False)
mapping=pd.read_csv('data/station_airport_map.csv')
weather=weather.merge(mapping,left_on='STATION',right_on='station',how='left')
weather['date']=weather['DATE'].str.slice(0,10)
weather.to_csv('data/all_weather_with_airport.csv',index=False)
print('Added airport codes to weather data.')