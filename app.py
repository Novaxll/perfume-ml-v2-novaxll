import os
import json
import urllib.request
import urllib.error
from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

def load_models():
    models = {}
    if os.path.exists('model.pkl'):
        models['active'] = joblib.load('model.pkl')
    for pct in ['100', '50', '25']:
        p = f'model/model_{pct}.pkl'
        if os.path.exists(p):
            models[pct] = joblib.load(p)
    return models

MODELS = load_models()
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

HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelOps - Predicción y Re-entrenamiento de Perfumes</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f1f5f9; color: #1e293b; margin: 0; padding: 2rem 1rem; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 640px; display: flex; flex-direction: column; gap: 1.5rem; }
        .card { background: #ffffff; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; }
        h1 { font-size: 1.4rem; margin: 0 0 0.4rem 0; text-align: center; color: #0f172a; }
        p.sub { text-align: center; color: #64748b; font-size: 0.9rem; margin: 0 0 1.5rem 0; }
        label { display: block; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.35rem; color: #334155; }
        input, select { width: 100%; padding: 0.65rem 0.75rem; margin-bottom: 1rem; border-radius: 8px; border: 1px solid #cbd5e1; background: #f8fafc; color: #0f172a; font-size: 0.92rem; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem; }
        button { width: 100%; padding: 0.8rem; border-radius: 8px; background: #2563eb; color: white; font-weight: 600; font-size: 0.95rem; border: none; cursor: pointer; transition: background 0.15s; }
        button:hover { background: #1d4ed8; }
        button.retrain-btn { background: #0f766e; }
        button.retrain-btn:hover { background: #115e59; }
        .result-box { margin-top: 1.2rem; padding: 1rem; border-radius: 8px; background: #dcfce7; border: 1px solid #86efac; color: #166534; text-align: center; display: none; }
        .result-box .val { font-size: 1.45rem; font-weight: 700; margin-top: 0.25rem; }
        .status-box { margin-top: 1rem; padding: 0.85rem; border-radius: 8px; font-size: 0.88rem; display: none; word-break: break-all; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🧴 Predicción de Longevidad de Perfumes</h1>
            <p class="sub">Selecciona el modelo entrenado y calcula la duración estimada</p>
            <form id="predictForm">
                <label>Dataset / Modelo a consultar:</label>
                <select id="dataset">
                    <option value="100">Modelo Entrenado al 100% Dataset</option>
                    <option value="50">Modelo Entrenado al 50% Dataset</option>
                    <option value="25">Modelo Entrenado al 25% Dataset</option>
                    <option value="active">Modelo Activo del Contenedor (Último CI/CD)</option>
                </select>

                <div class="grid-2">
                    <div>
                        <label>Marca (Brand)</label>
                        <input type="text" id="brand" value="Dior" required>
                    </div>
                    <div>
                        <label>Tipo de Concentración</label>
                        <select id="type">
                            <option value="EDP">EDP (Eau de Parfum)</option>
                            <option value="EDT">EDT (Eau de Toilette)</option>
                            <option value="Parfum">Parfum</option>
                            <option value="Extrait de Parfum">Extrait de Parfum</option>
                            <option value="Cologne">Cologne</option>
                        </select>
                    </div>
                </div>

                <div class="grid-2">
                    <div>
                        <label>Categoría Olfativa</label>
                        <input type="text" id="category" value="Floral" required>
                    </div>
                    <div>
                        <label>Público Objetivo</label>
                        <select id="target_audience">
                            <option value="Women">Women</option>
                            <option value="Men">Men</option>
                            <option value="Unisex">Unisex</option>
                        </select>
                    </div>
                </div>

                <button type="submit">Calcular Predicción</button>
            </form>

            <div id="predResult" class="result-box">
                <div id="predLabel">Duración Estimada (Modelo 100%):</div>
                <div class="val" id="predValue">0.00 horas</div>
                <div id="predMeta" style="font-size:0.8rem; margin-top:0.3rem;"></div>
            </div>
        </div>

        <div class="card">
            <h1>⚙️ Re-entrenar Modelo con GitHub Actions</h1>
            <p class="sub">Dispara el workflow ml_ci_cd con cualquiera de los 3 datasets en Cloudinary</p>
            <form id="retrainForm">
                <label>Seleccionar Versión de Dataset para Re-entrenar:</label>
                <select id="retrainPreset" onchange="updatePreset()">
                    <option value="100">Dataset al 100% (perfumes_100.csv)</option>
                    <option value="50">Dataset al 50% (perfumes_50.csv)</option>
                    <option value="25">Dataset al 25% (perfumes_25.csv)</option>
                </select>

                <label>URL del Dataset (Cloudinary Raw URL):</label>
                <input type="url" id="dataseturl" placeholder="https://res.cloudinary.com/p9kjggho/raw/upload/.../perfumes_100.csv" required>

                <div class="grid-2">
                    <div>
                        <label>Tag / SHA de Imagen Docker:</label>
                        <input type="text" id="sha" value="perfume-100" required>
                    </div>
                    <div>
                        <label>Repositorio GitHub:</label>
                        <input type="text" id="repo" value="Novaxll/perfume-ml-v2-novaxll" required>
                    </div>
                </div>

                <label>GitHub Classic Token (scope: workflow):</label>
                <input type="password" id="ghtoken" placeholder="ghp_xxxxxxxxxxxxxxxxxxxx">

                <button type="submit" class="retrain-btn">Disparar Re-entrenamiento (GitHub Actions)</button>
            </form>
            <div id="retrainStatus" class="status-box"></div>
        </div>
    </div>

    <script>
        const PRESET_URLS = {
            "100": { url: "https://res.cloudinary.com/p9kjggho/raw/upload/perfumes_100.csv", sha: "perfume-100" },
            "50":  { url: "https://res.cloudinary.com/p9kjggho/raw/upload/perfumes_50.csv",  sha: "perfume-50" },
            "25":  { url: "https://res.cloudinary.com/p9kjggho/raw/upload/perfumes_25.csv",  sha: "perfume-25" }
        };

        function updatePreset() {
            const val = document.getElementById('retrainPreset').value;
            document.getElementById('dataseturl').value = PRESET_URLS[val].url;
            document.getElementById('sha').value = PRESET_URLS[val].sha;
        }
        updatePreset();

        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const ds = document.getElementById('dataset').value;
            const payload = {
                dataset: ds,
                brand: document.getElementById('brand').value,
                type: document.getElementById('type').value,
                category: document.getElementById('category').value,
                target_audience: document.getElementById('target_audience').value
            };
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            const box = document.getElementById('predResult');
            if (data.prediction) {
                box.style.display = 'block';
                document.getElementById('predLabel').innerText = `Duración Estimada (Modelo ${data.dataset_used}%):`;
                document.getElementById('predValue').innerText = `${data.prediction[0].toFixed(2)} horas`;
                document.getElementById('predMeta').innerText = data.mae ? `MAE del modelo: ${data.mae} hrs | Filas: ${data.rows}` : '';
            }
        });

        document.getElementById('retrainForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const statusBox = document.getElementById('retrainStatus');
            statusBox.style.display = 'block';
            statusBox.style.background = '#e0f2fe';
            statusBox.style.color = '#0369a1';
            statusBox.innerText = 'Enviando repository_dispatch a GitHub Actions...';

            const res = await fetch('/retrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    repo: document.getElementById('repo').value,
                    token: document.getElementById('ghtoken').value,
                    dataseturl: document.getElementById('dataseturl').value,
                    sha: document.getElementById('sha').value
                })
            });
            const data = await res.json();
            if (res.ok) {
                statusBox.style.background = '#dcfce7';
                statusBox.style.color = '#166534';
                statusBox.innerText = `✅ Workflow disparado con éxito (HTTP ${data.status}). Revisa la pestaña Actions en tu GitHub.`;
            } else {
                statusBox.style.background = '#fee2e2';
                statusBox.style.color = '#991b1b';
                statusBox.innerText = `❌ Error: ${data.error || 'No se pudo disparar el workflow'}`;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
@app.route('/score', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    ds_choice = str(data.get('dataset', 'active'))
    selected_model = MODELS.get(ds_choice) or MODELS.get('active')
    
    clean_input = {k: v for k, v in data.items() if k != 'dataset'}
    df = pd.DataFrame([clean_input])
    if any(col in ['brand', 'type', 'category', 'target_audience'] for col in df.columns):
        df = pd.get_dummies(df)
    df_aligned = df.reindex(columns=selected_model.feature_names_in_, fill_value=0)
    prediction = selected_model.predict(df_aligned)
    val = float(prediction[0])
    meta = METRICS.get(ds_choice, METRICS.get('active', {}))
    return jsonify({
        'prediction': [val],
        'score': val,
        'dataset_used': ds_choice,
        'mae': meta.get('mae'),
        'rows': meta.get('rows')
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    body = request.get_json(force=True)
    repo = body.get('repo', 'Novaxll/perfume-ml-v2-novaxll')
    token = body.get('token') or os.environ.get('GITHUB_TOKEN', '')
    dataseturl = body.get('dataseturl')
    sha = body.get('sha', 'latest')

    if not token:
        return jsonify({'error': 'Falta el GitHub Token (ingrésalo o configúralo en GITHUB_TOKEN)'}), 400

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
            return jsonify({'status': resp.status, 'message': 'Workflow dispatched'}), 200
    except urllib.error.HTTPError as e:
        return jsonify({'error': f'GitHub API HTTP {e.code}: {e.read().decode("utf-8", errors="ignore")}'}), e.code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
