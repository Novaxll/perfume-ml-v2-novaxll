import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
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

pipe = Pipeline([('prep', prep), ('model', RandomForestRegressor(n_estimators=100, random_state=1))])
pipe.fit(X_tr, y_tr)
mae = mean_absolute_error(y_va, pipe.predict(X_va))

print(pd.DataFrame([{
    'Modelo': 'RandomForest',
    'MAE_Horas': round(mae, 4),
    'Error_Pct': f"{(mae/mean_y)*100:.2f}%"
}]).to_string(index=False))
