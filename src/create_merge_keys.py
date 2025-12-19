import pandas as pd
#Weather data: extract date only and rename station column for clarity
weather=pd.read_csv('data/all_weather.csv',low_memory=False)
weather['date']=weather['DATE'].str.slice(0,10)  #"YYYY-MM-DD"
weather['station']=weather['STATION']
print(weather[['station','date']].head())
#Flight data: create date for the first of each month and ensure airport code is a string
flight=pd.read_csv('data/Airline_Delay_Cause.csv')
flight['month']=flight['month'].astype(str).str.zfill(2)
flight['date']=flight['year'].astype(str)+'-'+flight['month']+'-01'
flight['airport']=flight['airport'].astype(str)
print(flight[['airport','date']].head())