import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
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

pipe = Pipeline([('prep', prep), ('model', DecisionTreeRegressor(max_leaf_nodes=150, random_state=1))])
pipe.fit(X_tr, y_tr)
mae = mean_absolute_error(y_va, pipe.predict(X_va))
error_pct = (mae / mean_y) * 100
mins = int(mae * 60)

print("\n" + "═"*70)
print("  📌 1. ÁRBOLES DE DECISIÓN SIMPLES (DecisionTreeRegressor)".center(70))
print("═"*70)
print("  💡 ¿Qué hace este script?")
print("     Entrena un árbol de decisión dividiendo los datos en ramas según las")
print("     características del perfume (Marca, Tipo, Categoría, Audiencia).")
print("─"*70)
print(f"  📊 Resultado de Evaluación (Out-of-Sample 80/20):")
print(f"     • Error Absoluto Medio (MAE) : {mae:.4f} horas (~{mins} minutos)")
print(f"     • Porcentaje de Error        : {error_pct:.2f}%")
print("─"*70)
print("  ✅ Conclusión:")
print(f"     El modelo simple predice la durabilidad del perfume con un margen")
print(f"     de error promedio de solo {mins} minutos frente a la realidad.")
print("═"*70 + "\n")
