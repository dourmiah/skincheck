import os
import numpy as np
import mlflow
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

api = Flask(__name__)

# Configuration de MLflow
MLFLOW_TRACKING_URI = "https://mlflow-jedha-app-ac2b4eb7451e.herokuapp.com/"
MODEL_RUN_ID = "495bc520d5ff42039590cc8038977981"

# Définition des classes
classes = ['chat', 'pas un chat']

# Assurez-vous que le dossier d'images existe
UPLOAD_FOLDER = 'src/upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fonction pour télécharger et charger le modèle depuis une exécution MLflow
def get_model_from_mlflow(run_id):
 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    artifact_uri = f"runs:/{run_id}/model"  # Spécifiez le chemin relatif du modèle
    model = mlflow.keras.load_model(artifact_uri)
    return model

# Chargement du modèle
model = get_model_from_mlflow(MODEL_RUN_ID)

# Fonction de prédiction
def predict(image_path):
 
    # Prétraitement de l'image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0

    # Prediction
    prediction = model.predict(img)
    probability = prediction[0][0]  # Get probability for positive class
    is_cat = probability > 0.5  # Apply threshold

    predicted_class = classes[1] if not is_cat else classes[0]  # If not cat, set to 'pas un chat'

    print(f"Probability: {probability}")

    return predicted_class, float(probability), bool(is_cat)

# Fonction pour traiter une requête d'envoi d'image
def new_predict(request):
    try:
        # Récupérer l'image envoyée
        if 'file' not in request.files:
            return jsonify({'error': 'Aucune image envoyée'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier invalide'}), 400

        # Enregistrer l'image temporairement
        image_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(image_path)

        # Prédire la classe de l'image
        predicted_class, probability, is_cat = predict(image_path)

        # Supprimer l'image temporaire
        os.remove(image_path)
        
        print(f"Image supprimée : {image_path}")

        # Retourner la classe prédite et la probabilité associée
        return {'classe': predicted_class, 'probabilité': probability, 'est_un_chat': is_cat}

    except Exception as e:
        # Gérer les erreurs
        print(f"Erreur lors de la prédiction : {e}")
        return jsonify({'error': 'Une erreur est survenue'}), 500

# Route pour l'envoi d'une image
@api.route('/api/predict', methods=['POST'])
def predict_image():
    try:
        predicted_result = new_predict(request)
        if 'error' in predicted_result:
            return jsonify(predicted_result), 500
        return redirect(url_for('show_result', classe=predicted_result['classe'], probability=predicted_result['probabilité'], is_cat=predicted_result['est_un_chat']))
    except Exception as e:
        return jsonify({'error': f'Une erreur est survenue : {e}'}), 500

# Route pour afficher le résultat
@api.route('/result')
def show_result():
    predicted_class = request.args.get('classe')
    probability = float(request.args.get('probability'))
    is_cat = request.args.get('is_cat') == 'True'
    return render_template('result.html', is_cat=is_cat, probability=probability)


# Fonction pour obtenir la version du modèle
def get_model_version(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.info.version if hasattr(run.info, 'version') else "unknown"

@api.route('/api/model_version', methods=['GET'])
def get_model_version_route():
    try:
        version = get_model_version(MODEL_RUN_ID)
        return jsonify({'model_version': version})
    except Exception as e:
        return jsonify({'error': f'Une erreur est survenue : {e}'}), 500

if __name__ == "__main__":
    api.run(port=5001, debug=True)
