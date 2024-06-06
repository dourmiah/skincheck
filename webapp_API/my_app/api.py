from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os


api = Flask(__name__)

CORS(api, origins="http://127.0.0.1:5000")  # Autorise les requêtes depuis ce domaine
api_base_url = os.getenv('API_BASE_URL')

CORS(api, origins={api_base_url})  # Autorise les requêtes depuis ce domaine

UPLOAD_FOLDER = "images"
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


if __name__ == "__main__":
    api.run(port=5001, debug=True)
