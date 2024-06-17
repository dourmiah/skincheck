import os
import sys
import time
import random
import mlflow
import datetime
import argparse
import numpy as np
import mlflow.keras
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

# Read : https://medium.com/stackademic/the-ultimate-guide-to-python-logging-simple-effective-and-powerful-9dbae53d9d6d
# Keep in mind : DEBUG INFO WARNING ERROR CRITICAL
import logging
import logging.config

from typing import Tuple
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# TODO :
# __init__   self.epochs = k_Epochs
# train_data_dir = k_DataDir
# parameters mentionnés par Raphael
# Read : https://stackoverflow.com/questions/66908259/how-to-fine-tune-inceptionv3-in-keras


# -----------------------------------------------------------------------------
# Prélude
# About k_DataDir, see volumes:[..] in MLproject file
# Don't forget that the app run in /home/app and it needs to access data that are in /home/data/train
# Do not change the path here. Update it in MLproject file
k_DataDir = "../data/train"
k_NbClasses = 24
k_L2 = 0.01
k_BatchSize = 64
k_Epochs = 50  # 25, 2 to debug, 8 to go fast...
# k_StepsPerEpoch = 20  # len(dataset)//batch_size
k_LearningRate = 0.001
k_RelativePath = "skin_check"
k_Img_Width = 299  # https://keras.io/api/applications/inceptionv3/
k_Img_Height = 299
k_Author = "Philippe"
k_XpPhase = "Investigation"


# -----------------------------------------------------------------------------
class ModelTrainer:

    # -----------------------------------------------------------------------------
    def __init__(self, epochs: int, batch_size: int) -> None:
        # self.epochs = epochs
        self.epochs = k_Epochs  # Faster to set it that way

        # self.batch_size = batch_size
        self.batch_size = k_BatchSize

    # -----------------------------------------------------------------------------
    def preprocess_data(
        self,
    ) -> Tuple[
        tf.keras.preprocessing.image.DirectoryIterator,
        tf.keras.preprocessing.image.DirectoryIterator,
    ]:
        """
        Preprocesses image data for training and validation.

        Returns:
            Tuple[tf.keras.preprocessing.image.DirectoryIterator, tf.keras.preprocessing.image.DirectoryIterator]:
                Tuple containing the training and validation data generators.
        """

        # -----------------------------------------------------------------------------
        # Mix data keeping coherence
        def shuffle_data(generator):
            indices = np.arange(generator.n)
            np.random.shuffle(indices)
            generator.index_array = indices

        start_time = time.time()

        # This is where we split between train & validation.
        img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255, validation_split=0.3
        )

        # TODO : should it be a parameter?
        train_data_dir = k_DataDir

        # Create 2 flux : training, validation
        # Flow from directory for training data
        img_generator_flow_train = img_generator.flow_from_directory(
            directory=train_data_dir,
            target_size=(k_Img_Height, k_Img_Width),
            batch_size=k_BatchSize,
            # shuffle=True,
            shuffle=False,  # see
            subset="training",
        )
        shuffle_data(img_generator_flow_train)

        # Map class indices to class names
        class_indices = img_generator_flow_train.class_indices
        self.indices_to_class = {v: k for k, v in class_indices.items()}

        # Flow from directory for validation data
        img_generator_flow_validation = img_generator.flow_from_directory(
            directory=train_data_dir,
            target_size=(k_Img_Height, k_Img_Width),
            batch_size=k_BatchSize,
            # shuffle=True,
            shuffle=False,
            subset="validation",
        )
        shuffle_data(img_generator_flow_validation)

        # Log preprocessing time using MLflow
        mlflow.log_metric("preprocess_data_time", round(time.time() - start_time, 2))

        # Return both training and validation data generators
        return img_generator_flow_train, img_generator_flow_validation

    # -----------------------------------------------------------------------------
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile a Keras sequential model.
        Returns:
            tf.keras.Model: Compiled Keras model.
        """

        start_time = time.time()

        base_model = tf.keras.applications.InceptionV3(
            input_shape=(k_Img_Height, k_Img_Width, 3),
            include_top=False,
            weights="imagenet",
        )
        # ! ATTENTION
        base_model.trainable = False

        # print("\n\nSummary about InceptionV3 : ")
        # base_model.summary()
        # print("End of summary\n\n")

        # tf.keras.utils.plot_model(
        #     # base_model, show_shapes=True, expand_nested=True, show_dtype=True
        #     base_model,
        #     to_file="inceptionv3.png",
        #     show_shapes=True,
        #     show_layer_names=True,
        # )

        # # Find out indexes of interrest
        # layer_names = [layer.name for layer in base_model.layers]
        # mixed9_index = layer_names.index("mixed9")
        # mixed10_index = layer_names.index("mixed10")

        # # Freeze all layers
        # for layer in base_model.layers:
        #     layer.trainable = False

        # # Unfreeze layers ]mixed9, mixed10]
        # # Layers between ]mixed9, mixed10] are now trainable
        # for layer in base_model.layers[mixed9_index + 1 : mixed10_index + 1]:
        #     layer.trainable = True

        # Get the number of layers
        # total_layers = len(base_model.layers)
        # print(f"Number of layers : {total_layers}")
        # layers_to_freeze = int(total_layers * 0.9)  # Geler 90% des couches

        # Frozen layers
        # for layer in base_model.layers[:layers_to_freeze]:
        #     layer.trainable = False

        # Unfreeze layers
        # for layer in base_model.layers[layers_to_freeze:]:
        #     layer.trainable = True

        model = tf.keras.Sequential(
            [
                base_model,
                # Suite discussion Colin
                # Enlever GlobalAveragePooling2D et remplacer par Flatten
                # 14/06/24 Du coup suite autres lectures et discussion on le remet :-)
                tf.keras.layers.GlobalAveragePooling2D(),
                # tf.keras.layers.Flatten(),
                # Ajouter 2 couches dense. Comme le model sort en 5x5x2048 on met 1024 et 512
                # Je ne sais pas trop si il faut une régule L2 partout ou juste sur la dernière couche
                # tf.keras.layers.Dense(
                #     1024,
                #     activation="relu",
                #     kernel_regularizer=tf.keras.regularizers.l2(k_L2),
                # ),
                # # 14/06/24 : on ne garde plus qu'une couche et on enlève la régularisation L2
                # tf.keras.layers.Dense(
                #     512,
                #     activation="relu",
                #     kernel_regularizer=tf.keras.regularizers.l2(k_L2),
                # ),
                tf.keras.layers.Dense(
                    # k_nb_classes classes à prédire
                    k_NbClasses,
                    activation="softmax",
                    # kernel_regularizer=tf.keras.regularizers.l2(k_L2),
                ),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=k_LearningRate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        mlflow.log_metric("build_model", round(time.time() - start_time, 2))
        return model

    # -----------------------------------------------------------------------------
    # TODO : potasser les paramètres ci-dessous (Raphael) pour aller plus vite
    # model.fit(
    #       ...,
    #       validation_freq=5,          # only run validation every 5 epochs
    #       validation_steps=20,        # run validation on 20 batches
    #       validation_batch_size=16,   # set validation batch size
    #       ...,
    # )
    def train_model(
        self,
        model: tf.keras.Model,
        img_generator_flow_train: tf.keras.preprocessing.image.DirectoryIterator,
        img_generator_flow_validation: tf.keras.preprocessing.image.DirectoryIterator,
    ) -> tf.keras.Model:
        """
        Train the model on the training data.
        Logs the time taken to train the model.
        Args:
            model (tf.keras.Model): The compiled Keras model.
            img_generator_flow_train (tf.keras.preprocessing.image.DirectoryIterator): Training dataset.
            img_generator_flow_validation (tf.keras.preprocessing.image.DirectoryIterator): Validation dataset.
        Returns:
            tf.keras.Model: The trained model
        """

        start_time = time.time()

        # Configurer le callback EarlyStopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",  # stop when the loss stop to diminish
            patience=10,  # nb epoch without improvement after we stop the training
            min_delta=0.001,  # smallest improvement
            restore_best_weights=True,  # restore model's weghts before best perf
        )

        nb_image_train = img_generator_flow_train.n

        model.fit(
            img_generator_flow_train,
            validation_data=img_generator_flow_validation,
            # steps_per_epoch=k_StepsPerEpoch,
            steps_per_epoch=nb_image_train // k_BatchSize,
            epochs=k_Epochs,
            callbacks=[early_stopping],
        )
        mlflow.log_metric("train_model_time", round(time.time() - start_time, 2))
        return model

    # -----------------------------------------------------------------------------
    def evaluate_model_1(self, model, img_generator_flow_validation):
        # Cette ligne récupère directement les classes vraies de l'ensemble de validation.
        y_true = img_generator_flow_validation.classes

        # Les prédictions sont obtenues en passant tout l'ensemble de validation à la méthode predict du modèle.
        y_pred = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12.0, 10.0))

        class_names = list(img_generator_flow_validation.class_indices.keys())

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title = f"./img/{timestamp}_confusion_matrix_1.png"
        plt.savefig(title)
        mlflow.log_artifact(title)
        return

    # -----------------------------------------------------------------------------
    # def evaluate_model_2(self, model, img_generator_flow_validation):

    #     y_true = []
    #     y_pred = []
    #     nb_image_val = img_generator_flow_validation.n
    #     batch_size = img_generator_flow_validation.batch_size

    #     for _ in range(nb_image_val // batch_size):
    #         # Ici, les étiquettes vraies et les prédictions sont récupérées par lots, en itérant sur l'ensemble de validation.
    #         # Peut causer des incohérences si le générateur d'images ne retourne pas les étiquettes dans le même ordre que img_generator_flow_validation.classes.
    #         val_imgs, val_targets = next(iter(img_generator_flow_validation))
    #         # prédit par lots, ce qui peut entraîner des variations si les lots sont traités de manière différente (par exemple, si le générateur n'est pas parfaitement synchronisé).
    #         predictions = model.predict(val_imgs, verbose=0)
    #         predicted_classes = np.argmax(predictions, axis=1)

    #         # gestion explicite pour les étiquettes en format multilabel, ce qui peut être crucial si vos étiquettes sont dans ce format.
    #         if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
    #             true_classes = np.argmax(val_targets, axis=1)
    #         else:
    #             true_classes = val_targets

    #         y_pred += list(predicted_classes)
    #         y_true += list(true_classes)

    #     cm = confusion_matrix(y_true, y_pred)
    #     plt.figure(figsize=(12.0, 10.0))

    #     class_names = list(img_generator_flow_validation.class_indices.keys())

    #     sns.heatmap(
    #         cm,
    #         annot=True,
    #         fmt="d",
    #         cmap="Blues",
    #         cbar=False,
    #         xticklabels=class_names,
    #         yticklabels=class_names,
    #     )
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     title = f"./img/{timestamp}_confusion_matrix_2.png"
    #     plt.savefig(title)
    #     mlflow.log_artifact(title)
    #     return

    # -----------------------------------------------------------------------------
    def evaluate_model(
        self,
        model: tf.keras.Model,
        img_generator_flow_validation: tf.keras.preprocessing.image.DirectoryIterator,
    ) -> None:
        """
        Evaluate the model on the test data.
        Logs the time taken to evaluate the model and the test loss.
        Args:
            model (tf.keras.Model): The trained Keras model.
            img_generator_flow_validation (tf.keras.preprocessing.image.DirectoryIterator) : Validation set
        """
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

        # Pour img_generator_flow_validation voir la ligne
        # img_generator_flow_validation = img_generator.flow_from_directory()
        y_true = img_generator_flow_validation.classes
        y_pred = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        mlflow.log_metric("Accuracy", round(accuracy, 2))
        mlflow.log_metric("Precision", round(precision, 2))
        mlflow.log_metric("Recall/Sensitivity", round(recall, 2))
        mlflow.log_metric("F1 Score", round(f1, 2))

        self.evaluate_model_1(model, img_generator_flow_validation)
        # self.evaluate_model_2(model, img_generator_flow_validation)

        mlflow.log_metric("evaluate_model_time", round(time.time() - start_time, 2))
        return

    # -----------------------------------------------------------------------------
    def log_tags_parameters(self) -> None:
        """
        Log some tags and the training parameters to mlflow.
        """

        mlflow.log_param("Classes", k_NbClasses)
        mlflow.log_param("Epochs", self.epochs)
        # mlflow.log_param("Steps per epochs", k_StepsPerEpoch)
        mlflow.log_param("L2", k_L2)
        mlflow.log_param("Learning rate", k_LearningRate)
        mlflow.log_param("Batch size", self.batch_size)

        mlflow.set_tag("Author", k_Author)
        mlflow.set_tag("Experiment phase", k_XpPhase)
        mlflow.set_tag("OS", sys.platform)
        mlflow.set_tag("Python version", sys.version.split("|")[0])
        mlflow.set_tag("mlflow version", mlflow.__version__)
        mlflow.set_tag("TensorFlow version", tf.__version__)

    # -----------------------------------------------------------------------------
    def log_model(
        self,
        model: tf.keras.Model,
        img_generator_flow_train: tf.keras.preprocessing.image.DirectoryIterator,
    ) -> None:
        """
        Log the trained model to mlflow with a signature.
        Args:
            model (tf.keras.Model): The trained Keras model.
            img_generator_flow_train (tf.keras.preprocessing.image.DirectoryIterator): Training dataset.
        """
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
        mlflow.log_metric("log_model", round(time.time() - start_time, 2))

    # -----------------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the full training and evaluation pipeline:
        - Load data
        - Preprocess data
        - Log tags & parameters
        - Build model
        - Train model
        - Evaluate model
        - Log model
        - Log total runtime
        """
        with mlflow.start_run():
            total_start_time = time.time()
            # self.load_data()                            # not needed with this model
            img_generator_flow_train, img_generator_flow_validation = (
                self.preprocess_data()
            )
            self.log_tags_parameters()
            model = self.build_model()
            model = self.train_model(
                model, img_generator_flow_train, img_generator_flow_validation
            )
            self.evaluate_model(model, img_generator_flow_validation)
            self.log_model(model, img_generator_flow_train)
            mlflow.log_metric(
                "total_run_time", round(time.time() - total_start_time, 2)
            )


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    start_time = time.time()

    # Load the logging configuration from the file
    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger(__name__)

    # Setup random generators such that we can compare runs over time
    seed = 0
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs("./img", exist_ok=True)

    logger.info(f"Training started")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    trainer = ModelTrainer(args.epochs, args.batch_size)
    trainer.run()

    logger.info(f"Training time        : {(time.time()-start_time):.3f} sec.")
    logger.info(f"Training stopped")
