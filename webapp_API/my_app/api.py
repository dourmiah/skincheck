import os
import numpy as np
import mlflow
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import json

api = Flask(__name__)

# Configuring MLflow
MLFLOW_TRACKING_URI = "https://mlflow-jedha-app-ac2b4eb7451e.herokuapp.com/"
MODEL_RUN_ID = "a92f4cdbbf1c42468531275f4a8556d3"

UPLOAD_FOLDER = "src/upload"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to download and load the model and classes from an MLflow execution
def get_model_and_classes_from_mlflow(run_id):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    artifact_uri = f"runs:/{run_id}/SkinCheck"
    model = mlflow.keras.load_model(artifact_uri)

    # Download artifacts
    artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

    classes_file_path = os.path.join(artifacts_path, "classes.json")

    # Load classes from JSON file
    with open(classes_file_path, "r") as f:
        classes = json.load(f)

    return model, classes


# Loading the model and classes
model, classes = get_model_and_classes_from_mlflow(MODEL_RUN_ID)


# Prediction function
def predict(image_path):
    # Image preprocessing
    img = image.load_img(image_path, target_size=(512, 512))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype("float32") / 255.0

    # Prediction
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class = classes[predicted_index]
    probability = prediction[0][predicted_index]

    return predicted_class, float(probability)


# Function to process an image send request
def new_predict(request):
    try:
        # Retrieve the sent image
        if "file" not in request.files:
            return jsonify({"error": "Aucune image envoyée"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Nom de fichier invalide"}), 400

        # Save image temporarily
        image_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(image_path)

        # Predict image class
        predicted_class, probability = predict(image_path)

        # Delete temporary image
        os.remove(image_path)

        print(f"Image supprimée : {image_path}")

        # Return the predicted class and associated probability
        return {"classe": predicted_class, "probabilite": probability}

    except Exception as e:
       
        print(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": "Une erreur est survenue"}), 500


# Route for sending an image
@api.route("/api/predict", methods=["POST"])
def predict_image():
    try:
        predicted_result = new_predict(request)
        if "error" in predicted_result:
            return jsonify(predicted_result), 500
        return redirect(
            url_for(
                "show_result",
                classe=predicted_result["classe"],
                probabilite=predicted_result["probabilite"],
            )
        )
    except Exception as e:
        return jsonify({"error": f"Une erreur est survenue : {e}"}), 500


# Route to display result
@api.route("/result")
def show_result():
    predicted_class = request.args.get("classe")
    print(predicted_class)
    probability = float(request.args.get("probabilite"))
    return render_template(
        "result.html", classe=predicted_class, probability=probability
    )


# Function to get model version
def get_model_version(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.info.version if hasattr(run.info, "version") else "unknown"


@api.route("/api/model_version", methods=["GET"])
def get_model_version_route():
    try:
        version = get_model_version(MODEL_RUN_ID)
        return jsonify({"model_version": version})
    except Exception as e:
        return jsonify({"error": f"Une erreur est survenue : {e}"}), 500


if __name__ == "__main__":
    api.run(port=5001, debug=True)
