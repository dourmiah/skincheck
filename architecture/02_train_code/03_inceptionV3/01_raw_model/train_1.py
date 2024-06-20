import os
import sys
import argparse
import pandas as pd
import time
import mlflow
import mlflow.keras
import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Keep in mind :
# mlflow.log_param  to log a parameter
# mlflow.log_metric to log a result

# Prétraitement des images avec ImageDataGenerator
k_Batch_Size = 32
k_Img_Height = 224
k_Img_Width = 224

k_data_dir = "../data/data_4/train"
k_nb_classes = 4
k_epochs = 25  # 8 pour aller vite
k_steps_per_epoch = 20


# Artifacts will be stored under    : skincheck-artifacts/2/118b36ff7c8f440db1a1c2bdb98d1008/artifacts/<k_RelativePath>/
# skincheck-artifacts               : Compartiment S3
# 2                                 : Experiment ID (defined by mlflow)
# 118b36ff7c8f440db1a1c2bdb98d1008  : Run ID (defined by mlflow)
# artifacts/<k_RelativePath>/       : relative path see artifact_path param below
k_RelativePath = "skin_check"


class ModelTrainer:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def load_data(self):
        start_time = time.time()
        # data = pd.read_csv(
        #     "https://skincheck-bucket.s3.eu-west-3.amazonaws.com/skincheck-dataset/california_housing_market.csv"
        # )
        mlflow.log_metric("load_data_time", time.time() - start_time)
        return

    # def preprocess_data(self, df):
    def preprocess_data(self):
        start_time = time.time()
        # X = df.iloc[:, :-1]
        # y = df.iloc[:, -1]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        # C'est là qu'on fait le split entre train et validation.
        # Voir le 0.3. Pas de data augmentation
        img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255, validation_split=0.3
        )

        # On crée 2 flux d'images : entrainement et validation
        # train_data_dir = "../data/train"
        # TODO : passer en paramètre
        train_data_dir = k_data_dir

        img_generator_flow_train = img_generator.flow_from_directory(
            directory=train_data_dir,
            target_size=(k_Img_Height, k_Img_Width),
            batch_size=k_Batch_Size,
            shuffle=True,
            subset="training",
        )

        img_generator_flow_valid = img_generator.flow_from_directory(
            directory=train_data_dir,
            target_size=(k_Img_Height, k_Img_Width),
            batch_size=k_Batch_Size,
            shuffle=True,
            subset="validation",
        )

        mlflow.log_metric("preprocess_data_time", time.time() - start_time)
        # return X_train_scaled, X_test_scaled, y_train, y_test
        return img_generator_flow_train, img_generator_flow_valid

    # def build_model(self, input_shape):
    def build_model(self):
        # model = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Dense(
        #             64, activation="relu", input_shape=(input_shape,)
        #         ),
        #         tf.keras.layers.Dense(1),
        #     ]
        # )
        # model.compile(optimizer="adam", loss="mse")

        base_model = tf.keras.applications.InceptionV3(
            input_shape=(k_Img_Height, k_Img_Width, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    # k_nb_classes classes à prédire
                    k_nb_classes,
                    activation="softmax",
                ),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        return model

    # def train_model(self, model, X_train, y_train):
    def train_model(self, model, img_generator_flow_train, img_generator_flow_valid):
        start_time = time.time()
        # history = model.fit(
        #     X_train,
        #     y_train,
        #     epochs=self.epochs,
        #     batch_size=self.batch_size,
        #     validation_split=0.2,
        # )
        model.fit(
            img_generator_flow_train,
            validation_data=img_generator_flow_valid,
            steps_per_epoch=k_steps_per_epoch,
            epochs=k_epochs,
        )

        mlflow.log_metric("train_model_time", time.time() - start_time)
        return model

    # def evaluate_model(self, model, X_test, y_test):
    def evaluate_model(self, model, img_generator_flow_valid):
        start_time = time.time()
        # loss = model.evaluate(X_test, y_test)
        # mlflow.log_metric("test_loss", loss)
        # Évaluation du modèle sur les données de validation

        history_dict = model.history.history

        plt.figure()
        plt.plot(history_dict["categorical_accuracy"], c="r", label="Train Accuracy")
        plt.plot(
            history_dict["val_categorical_accuracy"], c="b", label="Validation Accuracy"
        )
        plt.legend()
        plt.title("Accuracy vs epochs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title = f"./img/{timestamp}_accuracy.png"
        plt.savefig(title)
        # with mlflow.start_run():
        mlflow.log_artifact(title)

        plt.figure()
        plt.plot(history_dict["loss"], c="r", label="Train Loss")
        plt.plot(history_dict["val_loss"], c="b", label="Validation Loss")
        plt.legend()
        plt.title("Loss vs epochs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title = f"./img/{timestamp}_loss.png"
        plt.savefig(title)
        # with mlflow.start_run():
        mlflow.log_artifact(title)

        # Pour img_generator_flow_valid voir la ligne
        # img_generator_flow_valid = img_generator.flow_from_directory()
        y_true = img_generator_flow_valid.classes
        y_pred = np.argmax(model.predict(img_generator_flow_valid), axis=-1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        # print(f"Accuracy: {accuracy}")
        mlflow.log_metric("Accuracy", accuracy)

        # print(f"Precision: {precision}")
        mlflow.log_metric("Precision", precision)

        # print(f"Recall: {recall}")
        mlflow.log_metric("Recall/Sensitivity", recall)

        # print(f"F1 Score: {f1}")
        mlflow.log_metric("F1 Score", f1)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title = f"./img/{timestamp}_confusion_matrix.png"
        plt.savefig(title)
        # with mlflow.start_run():
        mlflow.log_artifact(title)

        # Voir plus tard
        # average=None => pour chaque classe
        # recall = recall_score(y_true, y_pred, average=None)
        # specificity = []
        # for i in range(num_classes):
        #     tn = np.sum(cm) - (np.sum(cm[:, i]) + np.sum(cm[i, :]) - cm[i, i])
        #     fp = np.sum(cm[:, i]) - cm[i, i]
        #     specificity.append(tn / (tn + fp))
        #
        # for i in range(num_classes):
        #     print(f"Class {i}: Sensitivity (Recall) = {recall[i]}, Specificity = {specificity[i]}")

        mlflow.log_metric("evaluate_model_time", time.time() - start_time)
        # return loss
        return

    def log_parameters(self):
        mlflow.log_param("epochs", self.epochs)
        mlflow.log_param("batch_size", self.batch_size)

    def log_model(self, model, img_generator_flow_train):
        example_input, _ = next(img_generator_flow_train)
        example_output = model.predict(example_input)
        signature = infer_signature(example_input, example_output)
        # signature = infer_signature(img_generator_flow_train, model.predict(img_generator_flow_train))
        mlflow.keras.log_model(
            model=model,
            artifact_path=k_RelativePath,
            registered_model_name="keras_sequential",
            signature=signature,
        )

    def run(self):
        with mlflow.start_run():
            total_start_time = time.time()
            # df = self.load_data()
            # X_train, X_test, y_train, y_test = self.preprocess_data(df)
            img_generator_flow_train, img_generator_flow_valid = self.preprocess_data()
            self.log_parameters()
            # model = self.build_model(X_train.shape[1])
            model = self.build_model()
            # model, _ = self.train_model(model, X_train, y_train)
            model = self.train_model(
                model, img_generator_flow_train, img_generator_flow_valid
            )
            # self.evaluate_model(model, X_test, y_test)
            self.evaluate_model(model, img_generator_flow_valid)
            # self.log_model(model, X_train)
            self.log_model(model, img_generator_flow_train)
            mlflow.log_metric("total_run_time", time.time() - total_start_time)


if __name__ == "__main__":
    # print("\n<START>\n")
    # print("OS                   : ", sys.platform)
    # print("Python version       : ", sys.version.split("|")[0])
    # print("mlflow version       : ", mlflow.__version__)
    # print("TensorFlow version   : ", tf.__version__)
    # print("\nTraining started     :")

    print("Repertoire : ", os.getcwd())
    open("../data/data_4/bob.txt", "a")

    start_time = time.time()

    os.makedirs("./img", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    trainer = ModelTrainer(args.epochs, args.batch_size)
    trainer.run()

    # print(f"Training finished    :")
    print(f"Training time        : {(time.time()-start_time):.3f} sec.")
    # print("\n<STOP>\n")


# num_classes = img_generator_flow_train.num_classes
# print(f"Number of classes: {num_classes}")

# Utilisation du modèle pré-entraîné sans fine-tuning
