"""Reproducible SpaceX Falcon 9 capstone analysis.
Run: python analysis.py
Reads data/dataset_part_2.csv and writes evidence/ charts, SQL tables, map and model metrics.
"""
from pathlib import Path
import sqlite3, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ROOT=Path(__file__).parent
DATA=ROOT/'data/dataset_part_2.csv'
OUT=ROOT/'evidence'
OUT.mkdir(exist_ok=True)
df=pd.read_csv(DATA)
df['Date']=pd.to_datetime(df['Date'],errors='coerce')
df['Year']=df.Date.dt.year
df['Class']=pd.to_numeric(df['Class'],errors='coerce').astype(int)
sns.set_theme(style='whitegrid')
def savefig(name):
    plt.tight_layout(); plt.savefig(OUT/name,dpi=180,bbox_inches='tight'); plt.close()

plt.figure(figsize=(9,5)); sns.countplot(data=df,x='LaunchSite',hue='Class'); plt.title('Landing outcome by launch site'); plt.xticks(rotation=15); savefig('eda_site_outcome.png')
plt.figure(figsize=(9,5)); sns.scatterplot(data=df,x='PayloadMass',y='Class',hue='LaunchSite',s=65); plt.title('Payload mass vs landing outcome'); savefig('eda_payload_outcome.png')
plt.figure(figsize=(9,5)); sns.countplot(data=df,x='Year',hue='Class'); plt.title('Launch outcomes by year'); plt.xticks(rotation=45); savefig('eda_year_trend.png')
plt.figure(figsize=(9,5)); sns.countplot(data=df,x='Orbit',hue='Class'); plt.title('Landing outcome by orbit'); plt.xticks(rotation=30); savefig('eda_orbit_outcome.png')

# SQL analysis using SQLite
con=sqlite3.connect(':memory:')
sql_df=df.copy(); sql_df['Date']=sql_df['Date'].dt.strftime('%Y-%m-%d')
sql_df.to_sql('spacex',con,index=False,if_exists='replace')
queries={
'launches_by_site':"SELECT LaunchSite, COUNT(*) launches FROM spacex GROUP BY LaunchSite ORDER BY launches DESC",
'launches_by_orbit':"SELECT Orbit, COUNT(*) launches FROM spacex GROUP BY Orbit ORDER BY launches DESC",
'success_by_site':"SELECT LaunchSite, COUNT(*) total_launches, SUM(Class) successes, ROUND(100.0*SUM(Class)/COUNT(*),1) success_rate_pct FROM spacex GROUP BY LaunchSite ORDER BY success_rate_pct DESC",
'payload_by_orbit':"SELECT Orbit, ROUND(AVG(PayloadMass),1) avg_payload_kg FROM spacex GROUP BY Orbit ORDER BY avg_payload_kg DESC",
'yearly_trend':"SELECT substr(Date,1,4) year, COUNT(*) launches, SUM(Class) successes FROM spacex GROUP BY year ORDER BY year"}
for name,q in queries.items(): pd.read_sql_query(q,con).to_csv(OUT/f'sql_{name}.csv',index=False)

# Folium site map: launch-site coordinates from the dataset; approximate landing-zone reference pins are illustrative only.
m=folium.Map(location=[28.5,-80.6],zoom_start=4,tiles='CartoDB positron')
sites=df.groupby('LaunchSite')[['Latitude','Longitude']].first().reset_index()
for _,r in sites.iterrows():
    folium.Marker([r.Latitude,r.Longitude],tooltip=r.LaunchSite,popup=f"{r.LaunchSite}: {int((df.LaunchSite==r.LaunchSite).sum())} launches",icon=folium.Icon(color='blue',icon='rocket',prefix='fa')).add_to(m)
# proximity buffer around each site (10 km illustrative); not a claim about actual recovery routes
for _,r in sites.iterrows():
    folium.Circle([r.Latitude,r.Longitude],radius=10000,color='orange',fill=False,tooltip='10 km illustrative proximity buffer').add_to(m)
m.save(str(OUT/'folium_launch_sites.html'))

# Predict landing outcome. Exclude Outcome and Class to avoid target leakage.
features=['PayloadMass','Flights','GridFins','Reused','Legs','Block','ReusedCount','Orbit','LaunchSite','BoosterVersion','Year']
X=df[features].copy(); y=df.Class
cat=X.select_dtypes(include=['object','bool']).columns.tolist()
num=[c for c in X.columns if c not in cat]
pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
                       ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
models={'Logistic Regression':LogisticRegression(max_iter=2000),'Decision Tree':DecisionTreeClassifier(max_depth=4,random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=300,max_depth=5,random_state=42,class_weight='balanced'),
        'Gradient Boosting':GradientBoostingClassifier(random_state=42)}
rows=[]; fitted={}
for name,model in models.items():
    pipe=Pipeline([('prep',pre),('model',model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); fitted[name]=pipe
    rows.append({'model':name,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0)})
metrics=pd.DataFrame(rows).sort_values('f1',ascending=False)
metrics.to_csv(OUT/'model_metrics.csv',index=False)
best=metrics.iloc[0]['model']; pred=fitted[best].predict(Xte)
cm=confusion_matrix(yte,pred,labels=[0,1])
disp=ConfusionMatrixDisplay(cm,display_labels=['No landing','Landing success'])
disp.plot(cmap='Blues',values_format='d'); plt.title(f'Confusion matrix — {best}'); savefig('confusion_matrix.png')
plt.figure(figsize=(8,5)); metrics.set_index('model')[['accuracy','precision','recall','f1']].plot(kind='bar'); plt.title('Model comparison'); plt.ylabel('Score'); plt.ylim(0,1.05); savefig('model_comparison.png')
summary={'rows':len(df),'columns':len(df.columns),'landing_success_rate_pct':round(100*y.mean(),1),'best_model_by_f1':best,'metrics':metrics.to_dict(orient='records'),'confusion_matrix_labels':['No landing','Landing success'],'confusion_matrix':cm.tolist()}
(OUT/'results_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
