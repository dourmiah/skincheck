import os
import sys
import time
import mlflow
import datetime
import argparse
import numpy as np
import mlflow.keras
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Prélude
k_DataDir = "../data/data/train"
k_NbClasses = 24
k_BatchSize = 32
k_Epochs = 25  # 25, 2 to debug, 8 to go fast...
k_StepsPerEpoch = 20
k_LearningRate = 0.001
k_RelativePath = "skin_check"
k_Img_Width = 224
k_Img_Height = 224


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class ModelTrainer:

    # -----------------------------------------------------------------------------
    def __init__(self, epochs, batch_size):
        # self.epochs = epochs
        self.epochs = k_Epochs  # Faster to set it that way

        # self.batch_size = batch_size
        self.batch_size = k_BatchSize

    # -----------------------------------------------------------------------------
    def load_data(self):
        start_time = time.time()
        # data = pd.read_csv(
        #     "https://skincheck-bucket.s3.eu-west-3.amazonaws.com/skincheck-dataset/california_housing_market.csv"
        # )
        mlflow.log_metric("load_data_time", time.time() - start_time)
        return

    # -----------------------------------------------------------------------------
    def preprocess_data(self):
        start_time = time.time()
        # This is where we split between train & validation.
        img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255, validation_split=0.3
        )

        # Create 2 flux : training, validation
        # TODO : should be a parameter?
        train_data_dir = k_DataDir

        img_generator_flow_train = img_generator.flow_from_directory(
            directory=train_data_dir,
            target_size=(k_Img_Height, k_Img_Width),
            batch_size=k_BatchSize,
            shuffle=True,
            subset="training",
        )

        class_indices = img_generator_flow_train.class_indices
        # Replace y_true, y_pred indices with names
        self.indices_to_class = {v: k for k, v in class_indices.items()}

        img_generator_flow_valid = img_generator.flow_from_directory(
            directory=train_data_dir,
            target_size=(k_Img_Height, k_Img_Width),
            batch_size=k_BatchSize,
            shuffle=True,
            subset="validation",
        )

        mlflow.log_metric("preprocess_data_time", time.time() - start_time)
        return img_generator_flow_train, img_generator_flow_valid

    # -----------------------------------------------------------------------------
    def build_model(self):
        start_time = time.time()
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
                    k_NbClasses,
                    activation="softmax",
                ),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=k_LearningRate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        mlflow.log_metric("build_model", time.time() - start_time)
        return model

    # -----------------------------------------------------------------------------
    # TODO : potasser les paramètres ci-dessous
    # model.fit(
    #       ...,
    #       validation_freq=5,          # only run validation every 5 epochs
    #       validation_steps=20,        # run validation on 20 batches
    #       validation_batch_size=16,   # set validation batch size
    #       ...,
    # )
    def train_model(self, model, img_generator_flow_train, img_generator_flow_valid):
        start_time = time.time()
        model.fit(
            img_generator_flow_train,
            validation_data=img_generator_flow_valid,
            steps_per_epoch=k_StepsPerEpoch,
            epochs=k_Epochs,
        )
        mlflow.log_metric("train_model_time", time.time() - start_time)
        return model

    # -----------------------------------------------------------------------------
    def evaluate_model(self, model, img_generator_flow_valid):
        start_time = time.time()

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
        mlflow.log_artifact(title)

        plt.figure()
        plt.plot(history_dict["loss"], c="r", label="Train Loss")
        plt.plot(history_dict["val_loss"], c="b", label="Validation Loss")
        plt.legend()
        plt.title("Loss vs epochs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title = f"./img/{timestamp}_loss.png"
        plt.savefig(title)
        mlflow.log_artifact(title)

        # Pour img_generator_flow_valid voir la ligne
        # img_generator_flow_valid = img_generator.flow_from_directory()
        y_true = img_generator_flow_valid.classes
        y_pred = np.argmax(model.predict(img_generator_flow_valid), axis=-1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall/Sensitivity", recall)
        mlflow.log_metric("F1 Score", f1)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12.0, 10.0))  # 1.618  plt.figure(figsize=(10, 8))
        class_names = [
            self.indices_to_class[i] for i in range(len(self.indices_to_class))
        ]
        heatmap = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10
        )
        heatmap.set_yticklabels(
            heatmap.get_yticklabels(), rotation=0, ha="right", fontsize=10
        )

        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title = f"./img/{timestamp}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(title)
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
        return

    # -----------------------------------------------------------------------------
    def log_parameters(self):
        mlflow.log_param("Classes", k_NbClasses)
        mlflow.log_param("Epochs", self.epochs)
        mlflow.log_param("Steps per epochs", k_StepsPerEpoch)
        mlflow.log_param("Learning rate", k_LearningRate)
        mlflow.log_param("Batch size", self.batch_size)
        mlflow.log_param("OS", sys.platform)
        mlflow.log_param("Python version", sys.version.split("|")[0])
        mlflow.log_param("mlflow version", mlflow.__version__)
        mlflow.log_param("TensorFlow version", tf.__version__)

    # -----------------------------------------------------------------------------
    def log_model(self, model, img_generator_flow_train):
        start_time = time.time()
        example_input, _ = next(img_generator_flow_train)
        example_output = model.predict(example_input)
        signature = infer_signature(example_input, example_output)
        mlflow.keras.log_model(
            model=model,
            artifact_path=k_RelativePath,
            registered_model_name="keras_sequential",
            signature=signature,
        )
        mlflow.log_metric("log_model", time.time() - start_time)

    # -----------------------------------------------------------------------------
    def run(self):
        with mlflow.start_run():
            total_start_time = time.time()
            # self.load_data()                            # not needed with this model
            img_generator_flow_train, img_generator_flow_valid = self.preprocess_data()
            self.log_parameters()
            model = self.build_model()
            model = self.train_model(
                model, img_generator_flow_train, img_generator_flow_valid
            )
            self.evaluate_model(model, img_generator_flow_valid)
            self.log_model(model, img_generator_flow_train)
            mlflow.log_metric("total_run_time", time.time() - total_start_time)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    start_time = time.time()

    os.makedirs("./img", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    trainer = ModelTrainer(args.epochs, args.batch_size)
    trainer.run()

    print(f"Training time        : {(time.time()-start_time):.3f} sec.")
