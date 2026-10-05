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

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

data = []
for n in [5, 15, 30, 75, 150]:
    m = DecisionTreeRegressor(max_leaf_nodes=n, random_state=1)
    m.fit(train_X, train_y)
    mae = mean_absolute_error(val_y, m.predict(val_X))
    data.append({
        'Max_Leaf_Nodes': n,
        'MAE_Hours': round(mae, 4),
        'Error_Pct': round((mae / mean_y) * 100, 2)
    })

print(pd.DataFrame(data).to_string(index=False))
