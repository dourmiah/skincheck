import sys
import argparse
import pandas as pd
import time

import mlflow
import mlflow.keras

import tensorflow as tf

from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Artifacts will be stored under    : skincheck-artifacts/2/118b36ff7c8f440db1a1c2bdb98d1008/artifacts/<k_RelativePath>/
# skincheck-artifacts               : Compartiment S3
# 2                                 : Experiment ID (defined by mlflow)
# 118b36ff7c8f440db1a1c2bdb98d1008  : Run ID (defined by mlflow)
# artifacts/<k_RelativePath>/       : relative path see artifact_path param below
k_RelativePath = "modeling_housing_market"


class ModelTrainer:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def load_data(self):
        start_time = time.time()
        data = pd.read_csv(
            "https://skincheck-bucket.s3.eu-west-3.amazonaws.com/skincheck-dataset/california_housing_market.csv"
        )
        mlflow.log_metric("load_data_time", time.time() - start_time)
        return data

    def preprocess_data(self, df):
        start_time = time.time()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlflow.log_metric("preprocess_data_time", time.time() - start_time)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self, input_shape):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    64, activation="relu", input_shape=(input_shape,)
                ),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def train_model(self, model, X_train, y_train):
        start_time = time.time()
        history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
        )
        mlflow.log_metric("train_model_time", time.time() - start_time)
        return model, history

    def evaluate_model(self, model, X_test, y_test):
        start_time = time.time()
        loss = model.evaluate(X_test, y_test)
        mlflow.log_metric("evaluate_model_time", time.time() - start_time)
        mlflow.log_metric("test_loss", loss)
        return loss

    def log_parameters(self):
        mlflow.log_param("epochs", self.epochs)
        mlflow.log_param("batch_size", self.batch_size)

    def log_model(self, model, X_train):
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.keras.log_model(
            model=model,
            artifact_path=k_RelativePath,
            registered_model_name="keras_sequential",
            signature=signature,
        )

    def run(self):
        with mlflow.start_run():
            total_start_time = time.time()
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            self.log_parameters()
            model = self.build_model(X_train.shape[1])
            model, _ = self.train_model(model, X_train, y_train)
            self.evaluate_model(model, X_test, y_test)
            self.log_model(model, X_train)
            mlflow.log_metric("total_run_time", time.time() - total_start_time)


if __name__ == "__main__":
    print("\n<START>\n")
    print("OS                   : ", sys.platform)
    print("Python version       : ", sys.version.split("|")[0])
    print("mlflow version       : ", mlflow.__version__)
    print("TensorFlow version   : ", tf.__version__)
    print("\nTraining started     :")

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    trainer = ModelTrainer(args.epochs, args.batch_size)
    trainer.run()

    print(f"Training finished    :")
    print(f"Training time        : {(time.time()-start_time):.3f} sec.")
    print("\n<STOP>\n")


# On peut aussi sauver des images dans les artifacts
# J'ai pas fait le test mais je pense qu'il faut que sklearn, maplotlib, seaborn, numpy soient dans l'image utilisée pour faire tourner train.py

# Partie I où on génère une matrice de confusion
#   from sklearn.metrics import confusion_matrix
#   import matplotlib.pyplot as plt
#   import seaborn as sns
#   import numpy as np
#   import mlflow

#   y_true = [0, 1, 0, 1]
#   y_pred = [0, 0, 0, 1]

#   cm = confusion_matrix(y_true, y_pred)
#   plt.figure(figsize=(10,7))
#   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#   plt.xlabel('Predicted')
#   plt.ylabel('True')
#   plt.title('Confusion Matrix')
#   plt.savefig("confusion_matrix.png")

# Partie II où on sauve l'artifact
#   with mlflow.start_run():
#       mlflow.log_artifact("confusion_matrix.png")


# Un autre exemple où tout se fait en mémoire, pas d'échange sur le disque
#   import io
#   from sklearn.metrics import confusion_matrix
#   import matplotlib.pyplot as plt
#   import seaborn as sns
#   import numpy as np
#   import mlflow

#   y_true = [0, 1, 0, 1]
#   y_pred = [0, 0, 0, 1]

#   cm = confusion_matrix(y_true, y_pred)
#   plt.figure(figsize=(10,7))
#   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#   plt.xlabel('Predicted')
#   plt.ylabel('True')
#   plt.title('Confusion Matrix')

# sauvegarde en mémoire
#   buf = io.BytesIO()
#   plt.savefig(buf, format='png')
#   buf.seek(0)

# Partie II où on sauve l'artifact
#   with mlflow.start_run():
#     # Le 2em para = le nom sous lequel l'image sera enregistrée dans MLflow
#     mlflow.log_image(buf, "Zoubida.png")
