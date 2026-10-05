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
evaluaciones.append({'Etapa': 'In-Sample (Memorizado)', 'MAE_Horas': round(mae_in, 4), 'Error_Pct': f"{(mae_in/mean_y)*100:.2f}%"})

m_split = DecisionTreeRegressor(random_state=1).fit(X_tr, y_tr)
mae_split = mean_absolute_error(y_va, m_split.predict(X_va))
evaluaciones.append({'Etapa': 'Out-Of-Sample (Base)', 'MAE_Horas': round(mae_split, 4), 'Error_Pct': f"{(mae_split/mean_y)*100:.2f}%"})

m_opt = DecisionTreeRegressor(max_leaf_nodes=150, random_state=1).fit(X_tr, y_tr)
mae_opt = mean_absolute_error(y_va, m_opt.predict(X_va))
evaluaciones.append({'Etapa': 'Out-Of-Sample (Optimizado)', 'MAE_Horas': round(mae_opt, 4), 'Error_Pct': f"{(mae_opt/mean_y)*100:.2f}%"})

print(pd.DataFrame(evaluaciones).to_string(index=False))
