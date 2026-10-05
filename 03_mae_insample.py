import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

longevity_map = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
df['y'] = df['longevity'].map(longevity_map).fillna(6)

X = pd.get_dummies(df[['brand', 'type', 'category', 'target_audience']], drop_first=True)
y = df['y']

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

mae = mean_absolute_error(y, model.predict(X))
mean_y = y.mean()
err_pct = (mae / mean_y) * 100

print(f"MAE_InSample_Hours: {mae:.4f}")
print(f"Error_Percentage: {err_pct:.2f}%")
