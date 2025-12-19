import pandas as pd
flight=pd.read_csv('data/Airline_Delay_Cause.csv')
flight['month']=flight['month'].astype(str).str.zfill(2)
flight['date']=flight['year'].astype(str)+'-'+flight['month']+'-01'
flight.to_csv('data/flights_with_date.csv',index=False)
print('Date column added to flight data.')
