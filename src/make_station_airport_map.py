import pandas as pd
from geopy.distance import geodesic
#Read weather stations (from all_weather.csv) and airports data
weather=pd.read_csv('data/all_weather.csv',low_memory=False)
airports=pd.read_csv('data/airports.csv',header=None, 
    names=['AirportID','Name','City','Country','IATA','ICAO','Latitude','Longitude','Altitude','Timezone','DST','TZ','Type','Source'])
#Only keep airports with valid IATA (explicit .copy() for safety)
airport_map=airports[airports['IATA'].notnull()&(airports['IATA']!='\\N')].copy()
#For each unique weather station, find closest airport (no in-place assignment!)
mapping=[]
unique_stations=weather[['STATION','LATITUDE','LONGITUDE']].drop_duplicates()
for _,row in unique_stations.iterrows():
    station_coord=(row['LATITUDE'],row['LONGITUDE'])
    #Calculate distances as a Series
    dists=airport_map.apply(
        lambda x:geodesic(station_coord,(x['Latitude'],x['Longitude'])).km,axis=1)
    closest=airport_map.iloc[dists.idxmin()]
    mapping.append({'station':row['STATION'],'airport':closest['IATA']})
pd.DataFrame(mapping).to_csv('data/station_airport_map.csv',index=False)
print('Saved station to airport mapping!')
