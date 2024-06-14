import os
import numpy as np
import mlflow
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import json

api = Flask(__name__)

# Configuration de MLflow
MLFLOW_TRACKING_URI = "https://mlflow-jedha-app-ac2b4eb7451e.herokuapp.com/"
MODEL_RUN_ID = "a2b4d06991ab4c75bbf7523c0dc61dea"

UPLOAD_FOLDER = 'src/upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fonction pour télécharger et charger le modèle et les classes depuis une exécution MLflow
def get_model_and_classes_from_mlflow(run_id):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    artifact_uri = f"runs:/{run_id}/SkinCheck"
    print(f"Artifact URI: {artifact_uri}")  # Debugging line
    model = mlflow.keras.load_model(artifact_uri)

    # Télécharger les artefacts
    artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)
    print(f"Artifacts path: {artifacts_path}")  # Debugging line

    classes_file_path = os.path.join(artifacts_path, "classes.json")

    # Charger les classes depuis le fichier JSON
    with open(classes_file_path, 'r') as f:
        classes = json.load(f)
        print(f"Classes: {classes}")       

    return model, classes

# Chargement du modèle et des classes
model, classes = get_model_and_classes_from_mlflow(MODEL_RUN_ID)


# Fonction de prédiction
def predict(image_path):
    # Prétraitement de l'image
    img = image.load_img(image_path, target_size=(512, 512))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0

    # Prediction
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction, axis=1)[0]
    print("MMMMMMMMMMMMMMMMMMMMMM")
    print(f"text: {image_path}")
    print('prediction :', prediction)
    print('predicted_index :', predicted_index)
    print('prediction argmax sans 0 :', np.argmax(prediction, axis=1))
    print(classes)
    predicted_class = classes[predicted_index]
    probability = prediction[0][predicted_index]

    return predicted_class, float(probability)

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
        predicted_class, probability = predict(image_path)

        # Supprimer l'image temporaire
        os.remove(image_path)
        
        print(f"Image supprimée : {image_path}")

        # Retourner la classe prédite et la probabilité associée
        return {'classe': predicted_class, 'probabilite': probability}

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
        return redirect(url_for('show_result', classe=predicted_result['classe'], probabilite=predicted_result['probabilite']))
    except Exception as e:
        return jsonify({'error': f'Une erreur est survenue : {e}'}), 500

# Route pour afficher le résultat
@api.route('/result')
def show_result():
    predicted_class = request.args.get('classe')
    print(predicted_class)
    probability = float(request.args.get('probabilite'))
    return render_template('result.html', classe=predicted_class, probability=probability)

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
