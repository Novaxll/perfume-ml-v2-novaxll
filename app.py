from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Duración de Perfumes</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 2rem; border-radius: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); width: 100%; max-width: 420px; border: 1px solid #334155; }
        h1 { font-size: 1.5rem; margin-bottom: 1.5rem; text-align: center; color: #38bdf8; }
        label { display: block; font-size: 0.875rem; margin-bottom: 0.25rem; color: #94a3b8; }
        input, select, button { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border-radius: 0.5rem; border: 1px solid #475569; background: #0f172a; color: white; box-sizing: border-box; }
        button { background: #0284c7; font-weight: bold; cursor: pointer; border: none; margin-top: 0.5rem; }
        button:hover { background: #0369a1; }
        .result { margin-top: 1rem; padding: 1rem; background: #0f172a; border-radius: 0.5rem; text-align: center; font-weight: bold; color: #4ade80; display: none; border: 1px solid #22c55e; }
    </style>
</head>
<body>
    <div class="card">
        <h1>✨ Longevidad de Perfumes</h1>
        <form id="predictForm">
            <label>Marca (Brand)</label>
            <input type="text" id="brand" placeholder="Ej. Dior, Chanel, Creed" required>
            
            <label>Tipo (Type)</label>
            <select id="type">
                <option value="EDP">EDP (Eau de Parfum)</option>
                <option value="EDT">EDT (Eau de Toilette)</option>
                <option value="Parfum">Parfum</option>
                <option value="Cologne">Cologne</option>
            </select>
            
            <label>Categoría</label>
            <input type="text" id="category" placeholder="Ej. Floral, Woody, Oriental" required>
            
            <label>Público Objetivo</label>
            <select id="target_audience">
                <option value="Women">Women</option>
                <option value="Men">Men</option>
                <option value="Unisex">Unisex</option>
            </select>
            
            <button type="submit">Calcular Duración</button>
        </form>
        <div id="result" class="result"></div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {
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
            const resDiv = document.getElementById('result');
            if (data.prediction) {
                resDiv.style.display = 'block';
                resDiv.innerText = `Duración estimada: ${data.prediction[0].toFixed(2)} horas`;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_LAYOUT)

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
