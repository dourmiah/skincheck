from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        image = Image.open(file_path)
        width, height = image.size
        return render_template(
            "index.html", filename=file.filename, width=width, height=height
        )


@app.route("/crop", methods=["POST"])
def crop_image():
    filename = request.form["filename"]
    x = int(float(request.form["x"]))
    y = int(float(request.form["y"]))

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image = Image.open(file_path)
    cropped_image = image.crop((x, y, x + 512, y + 512))
    cropped_image = cropped_image.resize((512, 512))
    output_path = os.path.join(app.config["RESULT_FOLDER"], "cropped_" + filename)
    cropped_image.save(output_path, "JPEG")

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
