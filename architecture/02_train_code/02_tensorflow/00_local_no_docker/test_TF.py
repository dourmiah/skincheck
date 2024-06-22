# F5 dans VSCode
# Choisir l'option qui permet de passer des paramètres
# Passer au pif : --epochs 5 --batch_size 1000

import sys
import argparse
import pandas as pd
import time

# import mlflow
import tensorflow as tf

# from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    print()
    print("<START>")
    print()

    print("OS                   : ", sys.platform)
    print("Python version       : ", sys.version.split("|")[0])
    print("TensorFlow version   : ", tf.__version__)
    print()

    print("Training started     :")

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_csv(
        "https://skincheck-bucket.s3.eu-west-3.amazonaws.com/skincheck-dataset/california_housing_market.csv"
    )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Data preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    epochs = args.epochs
    batch_size = args.batch_size

    # Define the model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                64, activation="relu", input_shape=(X_train_scaled.shape[1],)
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
    )
    loss = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {loss}")

    print(f"Training finished    :")
    print(f"Training time        : {(time.time()-start_time):.3f} sec.")

    print()
    print("<STOP>")
    print()
