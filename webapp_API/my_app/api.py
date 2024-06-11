from flask import Flask, request, jsonify, redirect,render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

api = Flask(__name__)

UPLOAD_FOLDER = "src/upload"
MODEL_VERSION = "1.0.0"  
API_VERSION = "1.0.0"

model_path = 'src/model/cat_classifier.h5'
model = load_model(model_path)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Charger le modèle
@api.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

        # Prétraiter l'image
        image = load_img(filepath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

         # Faire la prédiction
        prediction = model.predict(image)
        prediction_value = prediction[0][0]
        is_cat = prediction_value <= 0.5

        # Afficher les valeurs pour déboguer
        print(f"Prediction value: {prediction_value}")
        print(f"Is cat: {is_cat}")

        # Supprimer l'image après la prédiction
        os.remove(filepath)
        
        # Obtenir la base URL depuis les variables d'environnement
        base_url = os.getenv('APP_URL')

        # Retourner la prédiction dans une page HTML
        return render_template('result.html', is_cat=is_cat, base_url=base_url)

    except Exception as e:
        return str(e), 500
    
    
 # version

@api.route('/api/model_version', methods=['GET'])
def model_version():
    return jsonify({'model_version': MODEL_VERSION})

@api.route('/api/version', methods=['GET'])
def get_api_version():
    return jsonify({'api_version': API_VERSION})


if __name__ == "__main__":
    api.run(port=5001, debug=True)
