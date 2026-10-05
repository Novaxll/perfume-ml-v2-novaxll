import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

longevity_map = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
df['y'] = df['longevity'].map(longevity_map).fillna(6)

X = pd.get_dummies(df[['brand', 'type', 'category', 'target_audience']], drop_first=True)
y = df['y']
mean_y = y.mean()

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)

evaluaciones = []
m_in = DecisionTreeRegressor(random_state=1).fit(X, y)
mae_in = mean_absolute_error(y, m_in.predict(X))
evaluaciones.append({
    'Etapa de Evaluación': '1. In-Sample (Entrenamiento)',
    'MAE (Horas)': f"{mae_in:.4f} h",
    'Margen (Mins)': f"~{int(mae_in*60)} mins",
    'Error (%)': f"{(mae_in/mean_y)*100:.2f}%",
    'Descripción': 'Memoria directa de datos conocidos'
})

m_split = DecisionTreeRegressor(random_state=1).fit(X_tr, y_tr)
mae_split = mean_absolute_error(y_va, m_split.predict(X_va))
evaluaciones.append({
    'Etapa de Evaluación': '2. Out-Of-Sample (Base 80/20)',
    'MAE (Horas)': f"{mae_split:.4f} h",
    'Margen (Mins)': f"~{int(mae_split*60)} mins",
    'Error (%)': f"{(mae_split/mean_y)*100:.2f}%",
    'Descripción': 'Prueba en datos reales no vistos'
})

m_opt = DecisionTreeRegressor(max_leaf_nodes=150, random_state=1).fit(X_tr, y_tr)
mae_opt = mean_absolute_error(y_va, m_opt.predict(X_va))
evaluaciones.append({
    'Etapa de Evaluación': '3. Out-Of-Sample (Optimizado)',
    'MAE (Horas)': f"{mae_opt:.4f} h",
    'Margen (Mins)': f"~{int(mae_opt*60)} mins",
    'Error (%)': f"{(mae_opt/mean_y)*100:.2f}%",
    'Descripción': 'Árbol podado con max_leaf_nodes=150'
})

print("\n" + "═"*75)
print("  📉 5. EVALUACIÓN Y EVOLUCIÓN DEL ERROR ABSOLUTO MEDIO (MAE)".center(75))
print("═"*75)
print("  💡 ¿Qué hace este script?")
print("     Mide el MAE (Error Absoluto Medio en horas) en tres etapas distintas")
print("     para demostrar cómo cambia el rendimiento al validar correctamente.")
print("─"*75)
print(pd.DataFrame(evaluaciones).to_string(index=False))
print("─"*75)
print("  💡 Interpretación:")
print("     • El MAE de entrenamiento (~18 mins) es optimista por memorización.")
print("     • El MAE out-of-sample (~46 mins) representa el margen de error real.")
print("═"*75 + "\n")
