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

train_X1, val_X1, train_y1, val_y1 = train_test_split(X, y, test_size=0.25, random_state=0)
m1 = DecisionTreeRegressor(random_state=1).fit(train_X1, train_y1)
mae1 = mean_absolute_error(val_y1, m1.predict(val_X1))

train_X2, val_X2, train_y2, val_y2 = train_test_split(X, y, test_size=0.20, random_state=0)
m2 = DecisionTreeRegressor(random_state=1).fit(train_X2, train_y2)
mae2 = mean_absolute_error(val_y2, m2.predict(val_X2))

res = pd.DataFrame({
    'Split_Ratio': ['75/25', '80/20'],
    'MAE_Hours': [round(mae1, 4), round(mae2, 4)],
    'Error_Pct': [f"{(mae1/mean_y)*100:.2f}%", f"{(mae2/mean_y)*100:.2f}%"]
})
print(res.to_string(index=False))
