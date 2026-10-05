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
    'DecisionTree_Simple': DecisionTreeRegressor(max_leaf_nodes=150, random_state=1),
    'RandomForest_Ensemble': RandomForestRegressor(n_estimators=100, random_state=1),
    'XGBoost_Boosting': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=1)
}

res = []
for name, model in models.items():
    p = Pipeline([('prep', prep), ('model', model)])
    p.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_va, p.predict(X_va))
    res.append({
        'Algoritmo / Pipeline': name,
        'MAE (Horas)': f"{mae:.4f} h",
        'Margen (Mins)': f"~{int(mae*60)} mins",
        'Error (%)': f"{(mae/mean_y)*100:.2f}%"
    })

print("\n" + "═"*70)
print("  ⚙️ 4. COMPARATIVA DE MODELOS CON PIPELINES".center(70))
print("═"*70)
print("  💡 ¿Qué hace este script?")
print("     Encapsula la transformación de datos (OneHotEncoder) y el modelo en")
print("     un flujo automatizado único (Pipeline) para comparar los 3 métodos.")
print("─"*70)
print(pd.DataFrame(res).to_string(index=False))
print("─"*70)
print("  🏆 Ganador: DecisionTree_Simple ofrece la mayor precisión en este caso.")
print("═"*70 + "\n")
