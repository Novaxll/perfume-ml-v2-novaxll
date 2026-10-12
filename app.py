from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    
    # Si el JSON contiene columnas categóricas sin codificar, aplicar One-Hot Encoding
    if any(col in ['brand', 'type', 'category', 'target_audience'] for col in df.columns):
        df = pd.get_dummies(df)

    # Reordenar y rellenar las columnas faltantes con 0 para coincidir con las de fit()
    df_aligned = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(df_aligned)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
