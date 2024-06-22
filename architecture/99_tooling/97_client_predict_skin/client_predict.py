# A faire tourner dans un virtual env qui supporte tensorflow and mlflow

import mlflow
import boto3
import os
from PIL import Image

#
# import tensorflow as tf
from botocore.exceptions import NoCredentialsError


# -----------------------------------------------------------------------------
def load_and_preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Resize the image to 299x299
    image = image.resize((299, 299))

    # Convert the image to a numpy array and normalize it
    image_array = np.array(image).astype("float32") / 255.0

    # Expand dimensions to match the input shape of the model [-1, 299, 299, 3]
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    try:
        boto3.setup_default_session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
    except NoCredentialsError:
        print("Please make sure to run ./secrets.ps1 before to run this script.")

    mlflow.set_tracking_uri(
        "https://skincheck-tracking-server-6e98556bcc6b.herokuapp.com/"
    )

    logged_model = "runs:/63351df2b59a4d7bb23f62e3fd1232c1/skin_check"

    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    loaded_model = mlflow.keras.load_model(logged_model)

    input = load_and_preprocess_image("./data/07Acne081101.jpg")
    predictions = loaded_model.predict(input)
    print("Prédictions : ", predictions)
