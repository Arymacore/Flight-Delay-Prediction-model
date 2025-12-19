import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
st.set_page_config(page_title='Flight Delay Prediction',layout='wide')
st.title('Flight Delay Prediction Dashboard')
st.markdown('''
**About this Demo**

This app predicts whether a flight will be delayed (>15 minutes) using airline and weather data. Explore class balance, feature importances, model results, and weather correlations below!
\n**Note:** Most records in this data are labeled as delayed, so class results reflect that skew.
''')
#Load the prepped modeling data
try:
    df=pd.read_csv('outputs/model_input.csv')
except FileNotFoundError:
    st.error("Missing file: outputs/model_input.csv. Please make sure to build your dataset first!")
    st.stop()
#Data sample
st.write('## Data Sample')
st.dataframe(df.head(20))
#Class balance visualization
st.write('### Class Balance: Delayed vs On-time')
class_counts=df['delay_15'].value_counts().sort_index()
st.bar_chart(class_counts.rename({0:'On-Time',1:'Delayed'}))
# Allow filtering by airport or carrier
airport_opts=list(df['airport'].unique()) if 'airport' in df.columns else []
carrier_opts=list(df['carrier'].unique()) if 'carrier' in df.columns else []
df_filtered=df.copy()
col1,col2=st.columns(2)
if airport_opts:
    with col1:
        airport=st.selectbox('Filter by Airport',['All']+airport_opts)
        if airport!='All':
            df_filtered=df_filtered[df_filtered['airport']==airport]
if carrier_opts:
    with col2:
        carrier=st.selectbox('Filter by Carrier',['All']+carrier_opts)
        if carrier!='All':
            df_filtered=df_filtered[df_filtered['carrier']==carrier]
#Weather columns
weather_cols=[
    'HourlyDryBulbTemperature',
    'HourlyWindSpeed',
    'HourlyPrecipitation'
]
df_clean=df_filtered.dropna()
#Model training and prediction (on filtered data)
if len(df_clean)>15:
    X=df_clean[weather_cols]
    y=df_clean['delay_15']
    # Encode
    if 'airport' in df_clean.columns and df_clean['airport'].nunique()>1:
        X=pd.concat([X,pd.get_dummies(df_clean['airport'], prefix='airport',drop_first=True)],axis=1)
    if 'carrier' in df_clean.columns and df_clean['carrier'].nunique()>1:
        X=pd.concat([X,pd.get_dummies(df_clean['carrier'],prefix='carrier',drop_first=True)],axis=1)
    #Remove missing
    mask=~(X.isnull().any(axis=1)|y.isnull())
    X,y=X[mask],y[mask]
    if len(y)>0:
        rf=RandomForestClassifier(n_estimators=100,random_state=42)
        rf.fit(X,y)
        y_pred=rf.predict(X)
        #Metrics
        accuracy=(y_pred==y).mean()
        auc=roc_auc_score(y,rf.predict_proba(X)[:,1]) if y.nunique()>1 else float('nan')
        st.write('### Model Performance on Selected Data')
        col1,col2=st.columns(2)
        col1.metric('Accuracy',f'{accuracy:.2%}')
        if y.nunique()>1:
            col2.metric('ROC-AUC',f'{auc:.2f}')
        with st.expander("Show Classification Report"):
            st.text(classification_report(y,y_pred,digits=3))
        st.write('#### Confusion Matrix')
        cm=confusion_matrix(y, y_pred)
        fig,ax=plt.subplots()
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        #Feature importances
        st.write('#### Feature Importances')
        importances=rf.feature_importances_
        feat_names=X.columns
        imp_df=pd.DataFrame({"feature":feat_names,"importance":importances}).sort_values("importance",ascending=False)[:10]
        st.bar_chart(imp_df.set_index('feature'))
    else:
        st.info('No valid samples after filtering and cleaning.')
else:
    st.info('Insufficient data after filtering for modeling. Choose other filters or clean your data.')
#Visualizations: Delay vs Weather
st.write('# Delay vs Weather Visualizations')
for col in weather_cols:
    if col in df_clean.columns:
        fig,ax=plt.subplots()
        sns.scatterplot(x=col,y='arr_delay',data=df_clean,ax=ax,alpha=0.3)
        ax.set_title(f'Delay vs {col}')
        st.pyplot(fig)
st.write('---')
st.markdown('''
**How to interpret this dashboard:**
- Use the dropdowns to focus on specific airports or carriers.
- "Class Balance" shows how many flights were delayed vs on-time.
- "Model Performance" is for predicting delays just from airline and weather info—no data leakage (it does NOT use hidden answers).
- "Feature Importances" shows which variables matter most in prediction.
- "Delay vs Weather" plots help you visually explore patterns.

**Limitations:** This demo only predicts delay risk for the current dataset, which is skewed. In real projects, use multiple airports and balance the classes for better insight.
''')