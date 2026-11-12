import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

app = Flask(__name__)

AZURE_ENDPOINT = "https://reclamos-92b647.canadacentral.inference.ml.azure.com/score"
AZURE_KEY = os.getenv("AZURE_ML_KEY")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    texto = data.get('texto', '')
    
    if not texto:
        return jsonify({"error": "El texto está vacío"}), 400

    # Configurar la petición a Azure
    headers = {
        "Authorization": f"Bearer {AZURE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"textos": [texto]}

    try:
        response = requests.post(AZURE_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status() # Lanza error si Azure falla
        
        # Tu compañero dijo que devuelve una lista con los resultados
        resultado = response.json()[0] 
        return jsonify(resultado)
        
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error al conectar con el modelo: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)