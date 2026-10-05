import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

pipe = Pipeline([('prep', prep), ('model', XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=1))])
pipe.fit(X_tr, y_tr)
mae = mean_absolute_error(y_va, pipe.predict(X_va))
error_pct = (mae / mean_y) * 100

print("\n--- XGBoost ---")
print(f"MAE: {mae:.4f} hrs ({int(mae*60)} min)")
print(f"Error relativo: {error_pct:.2f}%\n")
