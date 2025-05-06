# Databricks notebook source
import requests
import json

def get_data_from_api(api_url, params=None, headers=None):
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

api_url = "https://data.transportation.gov/api/views/keg4-3bc2/rows.json?accessType=DOWNLOAD"  
data = get_data_from_api(api_url)

# COMMAND ----------

import pandas as pd
data_rows = data.get('data', [])
c=data.get('meta').get('view').get('columns')
col=[col['name'] for col in c]
df = pd.DataFrame(data_rows, columns=col)
df=df.iloc[:,9:]
df.info()

# COMMAND ----------

!pip install pymongo
import pymongo
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import AutoReconnect

# COMMAND ----------

uri = "mongodb+srv://lokesh:loki1234@cluster0.hqc9i.mongodb.net/?retryWrites=true&w=majority&connectTimeoutMS=300000"
client = MongoClient(uri, server_api=ServerApi('1'))
data = df.to_dict(orient='records')
db = client['Lokesh']
collection = db['Border']
try:
    batch_size = 1500
    for i in range(0, 150000, batch_size):
        batch = data[i : i + batch_size]
        collection.insert_many(batch, ordered=False)
except AutoReconnect as e:
    print(f"AutoReconnect error: {e}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting Data From Mongo DB

# COMMAND ----------

border_data = pd.DataFrame(list(collection.find()))
border_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Profiling/Analysis/Transform

# COMMAND ----------

border_data.columns

# COMMAND ----------

# Drop unnecessary columns
columns_to_drop = ['_id', 'Point', 'US Counties Shapefile', 'US Counties', 'US States and Territories']
border_data_clean = border_data.drop(columns=columns_to_drop)

# Basic data cleaning
border_data_clean = border_data_clean.dropna() 

# Convert date to datetime 
if 'Date' in border_data_clean.columns:
    border_data_clean['Date'] = pd.to_datetime(border_data_clean['Date'])
    
# Ensure numeric columns are proper types
numeric_cols = ['Value', 'Latitude', 'Longitude']
for col in numeric_cols:
    if col in border_data_clean.columns:
        border_data_clean[col] = pd.to_numeric(border_data_clean[col], errors='coerce')

border_data_clean = border_data_clean.drop_duplicates()

# Validate categorical columns
categorical_cols = ['State', 'Port Code', 'Border', 'Measure']
for col in categorical_cols:
    if col in border_data_clean.columns:
        border_data_clean[col] = border_data_clean[col].astype('category')

# COMMAND ----------

border_data_clean.head()

# COMMAND ----------

border_data_clean.columns

# COMMAND ----------

# Resample by month and calculate % change
time_agg = (
    border_data_clean.set_index('Date')
    .groupby(['Border', 'Measure'])['Value']
    .resample('M').sum()
    .unstack(level=[0, 1])
    .pct_change()  # Month-over-month growth
    .mul(100)     # Convert to percentage
)

# COMMAND ----------

geo_agg = (
    border_data_clean.groupby(['Port Code', 'Latitude', 'Longitude'])['Value']
    .sum()
    .pipe(lambda x: x[x > x.quantile(0.95)])  # Top 5% threshold
    .reset_index()
    .sort_values('Value', ascending=False)
)

# COMMAND ----------

geo_agg

# COMMAND ----------

measure_contribution = (
    border_data_clean.groupby(['Border', 'Measure'])['Value']
    .sum()
    .groupby(level=0).apply(lambda x: 100 * x / x.sum())
    .unstack()
    .style.background_gradient(cmap='Blues'))

# COMMAND ----------

measure_contribution

# COMMAND ----------

from scipy.stats import zscore
import numpy as np
anomalies = (
    border_data_clean.groupby(['Port Code', 'Measure'])['Value']
    .transform(lambda x: np.abs(zscore(x)) > 3)
    .value_counts()
)

# COMMAND ----------

anomalies

# COMMAND ----------

cross_border_agg = (
    border_data_clean.groupby(['State', 'Border', 'Measure'])['Value']
    .agg(['sum', 'mean', lambda x: x.ewm(span=3).mean().iloc[-1]])
    .rename(columns={'<lambda>': 'EWMA'})
    .nlargest(10, 'sum')
)
cross_border_agg

# COMMAND ----------

predictive_agg = (
    border_data_clean.groupby('Measure')['Value']
    .agg([np.mean, lambda x: np.percentile(x, 95)])
    .rename(columns={'<lambda>': 'P95_Threshold'})
)
predictive_agg

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization 

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Create month/weekday features
polar_data = border_data_clean.copy()
polar_data['Month'] = polar_data['Date'].dt.month_name()
polar_data['Weekday'] = polar_data['Date'].dt.day_name()

plt.figure(figsize=(10, 6))
ax = plt.subplot(111, polar=True)
sns.lineplot(
    data=polar_data.groupby(['Month', 'Measure'])['Value'].sum().reset_index(),
    x='Month',
    y='Value',
    hue='Measure',
    palette='viridis',
    sort=False,  # Preserve month order
    ax=ax
)
ax.set_theta_offset(np.pi/2)
ax.set_title("Seasonal Patterns by Measure", pad=20)
plt.tight_layout()

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.boxplot(
    data=border_data_clean,
    x='Measure',
    y='Value',
    palette='Set2'
)
plt.title('Distribution of Crossing Measures')
plt.ylabel('Daily Volume')
plt.xticks(rotation=90)
plt.show()


# COMMAND ----------

border_data_clean['Month'] = border_data_clean['Date'].dt.month_name()

plt.figure(figsize=(12,6))
sns.lineplot(
    data=border_data_clean.groupby(['Month','Border'])['Value'].sum().reset_index(),
    x='Month',
    y='Value',
    hue='Border',
    marker='o',
    sort=False  # Preserve calendar order
)
plt.title('Monthly Border Crossing Trends')
plt.ylabel('Total Volume')
plt.show()

# COMMAND ----------

plt.figure(figsize=(12,6))
sns.scatterplot(
    data=border_data_clean,
    x='Date',
    y='Value',
    hue='Measure',
    alpha=0.6
)
plt.title('Daily Crossing Volumes Over Time')
plt.ylabel('Volume')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Calculate total crossings per border
border_totals = border_data_clean.groupby('Border')['Value'].sum()

plt.figure(figsize=(6,6))
plt.pie(
    border_totals,
    labels=border_totals.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#66b3ff','#ff9999']
)
plt.title('Border Traffic Distribution')
plt.show()

# COMMAND ----------

top_ports = border_data_clean.groupby('Port Code')['Value'].sum().nlargest(5)

plt.figure(figsize=(10,5))
top_ports.plot(kind='bar', color='skyblue')
plt.title('Top 5 Busiest Border Ports')
plt.ylabel('Total Crossings')
plt.xlabel('Port Code')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

measure_totals = border_data_clean.groupby('Measure')['Value'].sum()

plt.figure(figsize=(8,8))
plt.pie(
    measure_totals,
    labels=measure_totals.index,
    autopct='%1.1f%%',
    pctdistance=0.85,
    wedgeprops={'width':0.4}  # Makes a donut chart
)
plt.title('Crossing Types Distribution')
plt.show()

# COMMAND ----------

border_data_clean['Month'] = border_data_clean['Date'].dt.month_name()
monthly_traffic = border_data_clean.groupby('Month')['Value'].sum()

plt.figure(figsize=(12,5))
monthly_traffic.plot(kind='bar', color='#2ca02c')
plt.title('Monthly Border Crossings')
plt.ylabel('Total Volume')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

from sklearn.ensemble import IsolationForest

# Prepare data
anomaly_data = border_data_clean.groupby(['Port Code', 'Date'])['Value'].sum().reset_index()
features = anomaly_data[['Value']]  # Can add more features like dayofweek

# Train model
iso_forest = IsolationForest(contamination=0.05)  # Expect 5% anomalies
anomaly_data['IsAnomaly'] = iso_forest.fit_predict(features)

# Visualize anomalies
plt.scatter(
    anomaly_data['Date'], 
    anomaly_data['Value'], 
    c=anomaly_data['IsAnomaly'], 
    cmap='cool'
)
plt.title('Anomaly Detection: Port Traffic')
plt.ylabel('Daily Crossings')

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare data
cluster_data = border_data_clean.groupby('Port Code').agg({
    'Value': ['mean', 'std'],  # Avg traffic and variability
    'Latitude': 'first',
    'Longitude': 'first'
}).dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)

# Cluster ports
kmeans = KMeans(n_clusters=3)  # Try 3-5 clusters
cluster_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot on map
plt.scatter(
    cluster_data['Longitude']['first'], 
    cluster_data['Latitude']['first'], 
    c=cluster_data['Cluster'], 
    cmap='viridis'
)
plt.title('Port Clusters by Traffic Patterns')

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

border_data_clean['Year'] = border_data_clean['Date'].dt.year
border_data_clean['Month'] = border_data_clean['Date'].dt.month
border_data_clean['DayOfWeek'] = border_data_clean['Date'].dt.dayofweek  # Monday=0, Sunday=6

cat_cols = ['State', 'Border', 'Measure', 'Port Code']
for col in cat_cols:
    border_data_clean[col] = border_data_clean[col].astype('category').cat.codes  # Convert to numeric codes

X = border_data_clean[['Year', 'Month', 'DayOfWeek', 'State', 'Border', 'Measure', 'Port Code']]
y = border_data_clean['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")  # Closer to 1 is better