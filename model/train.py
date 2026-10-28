import os
import json
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

LONGEVITY_MAP = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
FEATURES = ['brand', 'type', 'category', 'target_audience']

def train_from_csv(csv_path, output_pkl):
    df = pd.read_csv(csv_path)
    df = df[df['brand'] != 'Brand'].dropna(axis=0)
    df['y'] = df['longevity'].map(LONGEVITY_MAP).fillna(6)
    X = pd.get_dummies(df[FEATURES], drop_first=True)
    y = df['y']
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)
    model = DecisionTreeRegressor(max_leaf_nodes=150, random_state=1)
    model.fit(train_X, train_y)
    mae = mean_absolute_error(val_y, model.predict(val_X))
    joblib.dump(model, output_pkl)
    return round(float(mae), 4), len(df)

os.makedirs('model', exist_ok=True)
primary_csv = 'data/Perfumes_dataset.csv' if os.path.exists('data/Perfumes_dataset.csv') else 'Perfumes_dataset.csv'
mae_main, rows_main = train_from_csv(primary_csv, 'model.pkl')
joblib.copy_func = None
joblib.dump(joblib.load('model.pkl'), 'model/perfume-v1.joblib')

metrics = {'active': {'mae': mae_main, 'rows': rows_main}}

for pct in ['100', '50', '25']:
    path = f'data/perfumes_{pct}.csv'
    if os.path.exists(path):
        m_mae, m_rows = train_from_csv(path, f'model/model_{pct}.pkl')
        metrics[pct] = {'mae': m_mae, 'rows': m_rows}

with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Entrenamiento completado:", json.dumps(metrics))
