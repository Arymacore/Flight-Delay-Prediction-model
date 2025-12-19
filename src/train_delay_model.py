import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
#Step 1:Load cleaned data
df=pd.read_csv('outputs/model_input.csv')
#Step 2:Handle missing values
X=df[['HourlyDryBulbTemperature','HourlyWindSpeed','HourlyPrecipitation']].copy()
y=df['delay_15']
#Add one-hot encoding for categorical variables if present
if 'airport' in df.columns:
    X=pd.concat([X,pd.get_dummies(df['airport'],prefix='airport',drop_first=True)],axis=1)
if 'carrier' in df.columns:
    X=pd.concat([X,pd.get_dummies(df['carrier'],prefix='carrier',drop_first=True)],axis=1)
#Remove any rows with missing values in X/y
mask=~(X.isnull().any(axis=1)|y.isnull())
X,y=X[mask],y[mask]
#Step 3:Train/Test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
#Step 4:Fit Random Forest Classifier
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
#Step 5: Predict and Evaluate
y_pred=rf.predict(X_test)
y_prob=rf.predict_proba(X_test)[:,1] if hasattr(rf,'predict_proba') else rf.decision_function(X_test)
print('--- Classification Report ---')
print(classification_report(y_test,y_pred,digits=3))
print('--- Confusion Matrix ---')
print(confusion_matrix(y_test,y_pred))
print('--- ROC-AUC Score ---')
print(roc_auc_score(y_test,y_prob))
#Step 6:Feature Importance Plot
importances=rf.feature_importances_
feat_names=X.columns
indices=importances.argsort()[::-1]
plt.figure(figsize=(8,4))
sns.barplot(x=importances[indices][:10],y=feat_names[indices][:10])
plt.title('Top 10 Feature Importances (No Leakage)')
plt.show()