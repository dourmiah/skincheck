from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os


api = Flask(__name__)

CORS(api, origins="http://127.0.0.1:5000")  # Autorise les requêtes depuis ce domaine
api_base_url = os.getenv('API_BASE_URL')

CORS(api, origins={api_base_url})  # Autorise les requêtes depuis ce domaine

UPLOAD_FOLDER = "src/upluad"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


model_path = 'src/model/cat_classifier.h5'
model = load_model(model_path)

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
        is_cat = prediction[0][0] > 0.5

        # Supprimer l'image après la prédiction
        os.remove(filepath)

        # Retourner la prédiction dans une page HTML
        return render_template('result.html', is_cat=is_cat)

    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    api.run(port=5001, debug=True)
