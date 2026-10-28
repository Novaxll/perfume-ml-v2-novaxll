import os
import json
import urllib.request
import urllib.error
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODELS = {}
if os.path.exists('model/model_100.pkl'):
    MODELS['100'] = joblib.load('model/model_100.pkl')
if os.path.exists('model/model_50.pkl'):
    MODELS['50'] = joblib.load('model/model_50.pkl')
if os.path.exists('model/model_25.pkl'):
    MODELS['25'] = joblib.load('model/model_25.pkl')
if os.path.exists('model.pkl'):
    MODELS['active'] = joblib.load('model.pkl')

METRICS = {}
if os.path.exists('model/metrics.json'):
    with open('model/metrics.json', 'r') as f:
        METRICS = json.load(f)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'online',
        'models_loaded': list(MODELS.keys()),
        'metrics': METRICS
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
@app.route('/score', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json(force=True)
    ds_choice = str(data.get('dataset', '100'))
    selected_model = MODELS.get(ds_choice) or MODELS.get('100') or MODELS.get('active')

    clean_input = {k: v for k, v in data.items() if k not in ['dataset', 'model_version']}
    df = pd.DataFrame([clean_input])
    
    if any(col in ['brand', 'type', 'category', 'target_audience'] for col in df.columns):
        df = pd.get_dummies(df)
        
    df_aligned = df.reindex(columns=selected_model.feature_names_in_, fill_value=0)
    prediction = selected_model.predict(df_aligned)
    val = float(prediction[0])
    meta = METRICS.get(ds_choice, {})

    return jsonify({
        'prediction': [val],
        'score': val,
        'dataset_used': ds_choice,
        'mae': meta.get('mae'),
        'rows': meta.get('rows')
    })

@app.route('/retrain', methods=['POST', 'OPTIONS'])
def retrain():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    body = request.get_json(force=True)
    repo = body.get('repo', 'Novaxll/perfume-ml-v2-novaxll')
    token = body.get('token') or os.environ.get('GITHUB_TOKEN', '')
    dataseturl = body.get('dataseturl')
    sha = body.get('sha', 'latest')

    if not token:
        return jsonify({'error': 'Falta el GitHub Token'}), 400

    url = f'https://api.github.com/repos/{repo}/dispatches'
    payload = json.dumps({
        'event_type': 'ml_ci_cd',
        'client_payload': {
            'dataseturl': dataseturl,
            'sha': sha,
            'MODEL_NAME': 'perfumeml'
        }
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, method='POST')
    req.add_header('Authorization', f'Bearer {token}')
    req.add_header('Accept', 'application/vnd.github.v3+json')
    req.add_header('Content-Type', 'application/json')
    req.add_header('User-Agent', 'Perfume-MLOps-App')

    try:
        with urllib.request.urlopen(req) as resp:
            return jsonify({'status': resp.status, 'message': 'Workflow disparado'}), 200
    except urllib.error.HTTPError as e:
        return jsonify({'error': f'GitHub API {e.code}: {e.read().decode("utf-8", errors="ignore")}'}), e.code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
