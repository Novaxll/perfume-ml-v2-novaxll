from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'online',
        'message': 'API de Predicción de Perfumes lista',
        'endpoint': '/predict [POST]'
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    
    if any(col in ['brand', 'type', 'category', 'target_audience'] for col in df.columns):
        df = pd.get_dummies(df)

    df_aligned = df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(df_aligned)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
