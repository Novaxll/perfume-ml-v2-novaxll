import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

longevity_map = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
df['y'] = df['longevity'].map(longevity_map).fillna(6)

X = df[['brand', 'type', 'category', 'target_audience']]
y = df['y']
mean_y = y.mean()

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
prep = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['brand', 'type', 'category', 'target_audience'])])

models = {
    'DecisionTree': DecisionTreeRegressor(max_leaf_nodes=150, random_state=1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=1),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=1)
}

res = []
for name, model in models.items():
    p = Pipeline([('prep', prep), ('model', model)])
    p.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_va, p.predict(X_va))
    res.append({
        'Modelo': name,
        'MAE (hrs)': round(mae, 4),
        'Margen': f"{int(mae*60)} min",
        'Error (%)': f"{(mae/mean_y)*100:.2f}%"
    })

print("\n--- Comparativa de Pipelines ---")
print(pd.DataFrame(res).to_string(index=False) + "\n")
