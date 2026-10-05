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

progression = []
nodes = [5, 15, 30, 75, 150]
base_mae = None

for step, n in enumerate(nodes, start=1):
    m = DecisionTreeRegressor(max_leaf_nodes=n, random_state=1)
    m.fit(train_X, train_y)
    mae = mean_absolute_error(val_y, m.predict(val_X))
    if base_mae is None:
        base_mae = mae
    reduction = ((base_mae - mae) / base_mae) * 100
    err_pct = (mae / mean_y) * 100
    progression.append({
        'Step': step,
        'Config': f'max_leaf_nodes={n}',
        'MAE_Hours': round(mae, 4),
        'Error_Pct': f"{err_pct:.2f}%",
        'Error_Reduction': f"-{reduction:.2f}%"
    })

print(pd.DataFrame(progression).to_string(index=False))
