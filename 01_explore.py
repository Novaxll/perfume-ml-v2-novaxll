import warnings
warnings.filterwarnings('ignore')
import pandas as pd

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

print(df[['brand', 'type', 'category', 'target_audience', 'longevity']].describe(include='all').fillna(''))
