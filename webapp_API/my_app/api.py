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


@api.route("/api/greet", methods=["GET"])
def greet():
    return jsonify(message="coucou")


@api.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify(message="Aucun fichier trouvé"), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify(message="Aucun fichier sélectionné"), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)
    app_base_url = os.getenv("APP_BASE_URL")
    return redirect(f"{app_base_url}/")

model_path = 'src/model/cat_classifier.h5'
model = load_model(model_path)

# Charger le modèle
@api.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Vérifier si le fichier est dans la requête
        if 'file' not in request.files:
            return 'No file provided', 400

        file = request.files['file']

        # Prétraiter l'image
        image = load_img(file, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # Faire la prédiction
        prediction = model.predict(image)
        is_cat = prediction[0][0] > 0.5

        # Retourner la prédiction dans une page HTML
        return render_template('result.html', is_cat=is_cat)

    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    api.run(port=5001, debug=True)
