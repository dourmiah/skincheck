from flask import Flask, render_template
from dotenv import load_dotenv
import os
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)


@app.route("/")
def home():
    base_url = os.getenv("APP_URL")
    api_url = os.getenv("API_URL")
    return render_template("index.html", base_url=base_url, api_url=api_url)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
