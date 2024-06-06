from flask import Flask, render_template
import os

app = Flask(__name__)


@app.route("/")
def home():
    app_base_url = os.getenv('APP_BASE_URL')
    return render_template('index.html', app_base_url=app_base_url)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
