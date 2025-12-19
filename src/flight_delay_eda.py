import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Step 1: Load data
df=pd.read_csv('outputs/merged_flights_weather.csv',low_memory=False)
#Step 2: Quick inspection
print('=== Head of Data ===')
print(df.head())
print('=== Info ===')
print(df.info())
print('=== Description ===')
print(df.describe(include='all'))
print('Weather column hints:')
for col in df.columns:
    if 'temp' in col.lower() or 'wind' in col.lower() or 'precip' in col.lower():
        print(col)
#Step 3: Missing values
print('=== Missing values per column ===')
print(df.isnull().sum())
#Step 4: Clean numeric weather columns (replace ''T'' and non-numerics with 0/NaN)
weather_cols=[
    'HourlyDryBulbTemperature',
    'HourlyPrecipitation',
    'HourlyWindSpeed'
]
for col in weather_cols:
    if col in df.columns:
        df[col]=pd.to_numeric(df[col].replace('T',0),errors='coerce')
#Step 5: Visualize delays by airport and airline (use errorbar=None instead of ci=None)
plt.figure(figsize=(12,4))
sns.barplot(x='airport',y='arr_delay',data=df,errorbar=None)
plt.title('Average Arrival Delay by Airport')
plt.ylabel('Avg Arrival Delay (min)')
plt.xlabel('Airport')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,4))
sns.barplot(x='carrier',y='arr_delay',data=df,errorbar=None)
plt.title('Average Arrival Delay by Airline')
plt.ylabel('Avg Arrival Delay (min)')
plt.xlabel('Airline')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
#Step 6: Correlation with weather features
num_cols=['arr_delay']+[c for c in weather_cols if c in df.columns]
if len(num_cols)>1:
    corr=df[num_cols].corr()
    sns.heatmap(corr,annot=True,cmap='coolwarm')
    plt.title('Correlation between Delay and Weather')
    plt.show()
else:
    print('No numeric weather columns found for correlation plot.')
#Step 7: Delay distributions
plt.figure(figsize=(7,4))
df['arr_delay'].hist(bins=50)
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (min)')
plt.ylabel('Count')
plt.show()
#Step 8: Create and inspect delay flag (for ML classification)
if 'arr_delay' in df.columns:
    df['delay_15']=(df['arr_delay']>15).astype(int)
    print(df['delay_15'].value_counts())
else:
    print('arr_delay column not found for delay flag creation.')
#Step 9: Save cleaned data for modeling (only columns that exist!)
clean_columns=[
    'airport','carrier','year','month','arr_delay','delay_15',
    'HourlyDryBulbTemperature', 'HourlyWindSpeed','HourlyPrecipitation'
]
existing_columns=[c for c in clean_columns if c in df.columns]
df[existing_columns].to_csv('outputs/model_input.csv',index=False)
print('Cleaned data saved for modeling!')