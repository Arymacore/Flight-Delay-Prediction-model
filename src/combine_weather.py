import pandas as pd
import glob
import os
csv_files=glob.glob(os.path.join('data','72509014739*.csv'))
dfs=[pd.read_csv(f) for f in csv_files]
all_weather=pd.concat(dfs,ignore_index=True)
all_weather.to_csv('data/all_weather.csv',index=False)
print(f"Combined {len(csv_files)} files into data/all_weather.csv")