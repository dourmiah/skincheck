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
# -----------------------------------------------------------------------------
# Prélude
# About k_DataDir, see volumes:[..] in MLproject file
# Don't forget that the app run in /home/app and it needs to access data that are in /home/data/train
# Do not change the path here. Update it in MLproject file
k_DataDir = "../data/train"
k_NbClasses = 4
k_L2 = 0.01
k_BatchSize = 32
k_Epochs = 50  # 25, 2 to debug, 8 to go fast...
k_StepsPerEpoch = 20
k_LearningRate = 0.001
k_RelativePath = "skin_check"
k_Img_Width = 299  # https://keras.io/api/applications/inceptionv3/
k_Img_Height = 299
k_Author = "Philippe"
k_XpPhase = "Investigation"


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class ModelTrainer:

    # -----------------------------------------------------------------------------
    def __init__(self, epochs: int, batch_size: int) -> None:
        # self.epochs = epochs
        self.epochs = k_Epochs  # Faster to set it that way

        # self.batch_size = batch_size
        self.batch_size = k_BatchSize

    # -----------------------------------------------------------------------------
    # def load_data(self) -> pd.DataFrame:
    #     """
    #     Load the dataset from a local CSV file or from S3.
    #     Logs the time taken to load the data.
    #     Returns:
    #         pd.DataFrame: Loaded data.
    #     """
    #     try:
    #         start_time = time.time()
    #         # Uncomment the line below to load data from S3
    #         # data = pd.read_csv("https://skincheck-bucket.s3.eu-west-3.amazonaws.com/skincheck-dataset/california_housing_market.csv")
    #         data = pd.read_csv("./data/california_housing_market.csv")
    #         mlflow.log_metric("load_data_time", time.time() - start_time)
    #         return data
    #     except Exception as e:
    #         logger.error(f"Error loading data: {e}")
    #         raise

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
        # Mixt data keeping coherence
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
        mlflow.log_metric("preprocess_data_time", time.time() - start_time)

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
        # base_model.trainable = False

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

        # Find out indexes of interrest
        layer_names = [layer.name for layer in base_model.layers]
        mixed9_index = layer_names.index("mixed9")
        mixed10_index = layer_names.index("mixed10")

        # Freeze all layers
        for layer in base_model.layers:
            layer.trainable = False

        # Unfreeze layers ]mixed9, mixed10]
        # Layers between ]mixed9, mixed10] are now trainable
        for layer in base_model.layers[mixed9_index + 1 : mixed10_index + 1]:
            layer.trainable = True

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
                tf.keras.layers.Dense(
                    1024,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(k_L2),
                ),
                # 14/06/24 : on ne garde plus qu'une couche et on enlève la régularisation L2
                tf.keras.layers.Dense(
                    512,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(k_L2),
                ),
                tf.keras.layers.Dense(
                    # k_nb_classes classes à prédire
                    k_NbClasses,
                    activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(k_L2),
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
            patience=5,  # nb epoch without improvement after we stop the training
            min_delta=0.001,  # smallest improvement
            restore_best_weights=True,  # restore model's weghts before best perf
        )

        model.fit(
            img_generator_flow_train,
            validation_data=img_generator_flow_validation,
            steps_per_epoch=k_StepsPerEpoch,
            epochs=k_Epochs,
            callbacks=[early_stopping],
        )
        mlflow.log_metric("train_model_time", time.time() - start_time)
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
    # def evaluate_model_1(self, model, img_generator_flow_validation):

    #     # y_true = img_generator_flow_validation.classes
    #     # model.predict() retourne un tableau de forme (num_images, num_classes)
    #     # num_images est le nombre d'images dans l'ensemble de validation et num_classes est le nombre de classes de classification.
    #     # Chaque entrée predictions[i] est donc un vecteur de probabilités pour les différentes classes pour l'image i.
    #     # axis=-1 spécifie l'axe le plus interne.
    #     # Pour le tableau 2D en sortie de model.predict() cela signifie qu'il trouvera l'indice de la valeur maximale le long de la dernière dimension
    #     # => pour chaque vecteur de classes
    #     # y_pred = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

    #     # Pas besoin de gestion explicite multilabel
    #     # labels instead of classes for multilabel
    #     y_true = img_generator_flow_validation.classes
    #     y_pred = model.predict(img_generator_flow_validation)

    #     if len(y_true.shape) > 1 and y_true.shape[1] > 1:
    #         # Convert multilabel to single label
    #         logger.info(f"evaluate_model_1 - multi-label")

    #         y_true = np.argmax(y_true, axis=1)
    #         y_pred = np.argmax(y_pred, axis=1)
    #     else:
    #         y_pred = np.argmax(y_pred, axis=-1)
    #         logger.info(f"evaluate_model_1 - no multi-label")

    #     cm = confusion_matrix(y_true, y_pred)
    #     plt.figure(figsize=(12.0, 10.0))

    #     class_names = list(img_generator_flow_validation.class_indices.keys())
    #     # class_names = [self.indices_to_class[i] for i in range(len(self.indices_to_class))]
    #     # logger.debug(f"evaluate_model_1 - class_names  = {class_names}")
    #     # 2024-06-15 17:27:36,060 - __main__ - DEBUG - evaluate_model_1 - class_names  = ['acne_and_rosacea', 'actinic_keratosis', 'atopic_dermatitis', 'healthy_skins']
    #     # 2024-06-15 17:28:04,353 - __main__ - DEBUG - evaluate_model_2 - class_names  = ['acne_and_rosacea', 'actinic_keratosis', 'atopic_dermatitis', 'healthy_skins']

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
    #     title = f"./img/{timestamp}_confusion_matrix_1.png"
    #     plt.savefig(title)
    #     mlflow.log_artifact(title)
    #     return

    # -----------------------------------------------------------------------------
    def evaluate_model_2(self, model, img_generator_flow_validation):

        y_true = []
        y_pred = []
        nb_image_val = img_generator_flow_validation.n
        batch_size = img_generator_flow_validation.batch_size

        for _ in range(nb_image_val // batch_size):
            # Ici, les étiquettes vraies et les prédictions sont récupérées par lots, en itérant sur l'ensemble de validation.
            # Peut causer des incohérences si le générateur d'images ne retourne pas les étiquettes dans le même ordre que img_generator_flow_validation.classes.
            val_imgs, val_targets = next(iter(img_generator_flow_validation))
            # prédit par lots, ce qui peut entraîner des variations si les lots sont traités de manière différente (par exemple, si le générateur n'est pas parfaitement synchronisé).
            predictions = model.predict(val_imgs, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            # gestion explicite pour les étiquettes en format multilabel, ce qui peut être crucial si vos étiquettes sont dans ce format.
            if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
                true_classes = np.argmax(val_targets, axis=1)
            else:
                true_classes = val_targets

            y_pred += list(predicted_classes)
            y_true += list(true_classes)

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
        title = f"./img/{timestamp}_confusion_matrix_2.png"
        plt.savefig(title)
        mlflow.log_artifact(title)
        return

    # -----------------------------------------------------------------------------
    # Dans evaluate_model_1, les étiquettes vraies sont récupérées directement de img_generator_flow_validation.classes.
    # Dans evaluate_model_2, les étiquettes vraies sont récupérées par lots via l'itérateur du générateur d'images.
    # Cela peut causer des incohérences si le générateur d'images ne retourne pas les étiquettes dans le même ordre que img_generator_flow_validation.classes.
    # evaluate_model_1 utilise une prédiction globale sur tout l'ensemble de validation.
    # evaluate_model_2 prédit par lots, ce qui peut entraîner des variations si les lots sont traités de manière différente
    # par exemple, si le générateur n'est pas parfaitement synchronisé
    # def evaluate_model_2(self, model, img_generator_flow_validation):

    #     y_true = []
    #     y_pred = []

    #     # logger.debug(f"Nb images valiadation = {img_generator_flow_validation.n}")
    #     # nb_image_val = 1114
    #     nb_image_val = img_generator_flow_validation.n
    #     batch_size = img_generator_flow_validation.batch_size

    #     # Gestion explicite multi-label
    #     for _ in range(nb_image_val // batch_size):
    #         val_imgs, val_targets = next(iter(img_generator_flow_validation))
    #         predictions = model.predict(val_imgs, verbose=0)
    #         predicted_classes = np.argmax(predictions, axis=1)

    #         # Convert val_targets to class indices if it's in multilabel-indicator format
    #         # Confirmé : on est en multi label
    #         if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
    #             true_classes = np.argmax(val_targets, axis=1)
    #             # logger.info(f"evaluate_model_2 - multi-label")
    #         else:
    #             true_classes = val_targets
    #             # logger.info(f"evaluate_model_2 - no multi-label")

    #         y_pred += list(predicted_classes)
    #         y_true += list(true_classes)

    #     # Sans gestion explicite multilabel => MARCHE PAS
    #     # for _ in range(nb_image_val // batch_size):
    #     #     val_imgs, val_targets = next(iter(img_generator_flow_validation))
    #     #     predictions = model.predict(val_imgs, verbose=0)
    #     #     predicted_classes = np.argmax(predictions, axis=-1)

    #     #     # Directly append the true classes and predicted classes
    #     #     y_true += list(val_targets)
    #     #     y_pred += list(predicted_classes)

    #     cm = confusion_matrix(y_true, y_pred)
    #     plt.figure(figsize=(12.0, 10.0))

    #     class_names = list(img_generator_flow_validation.class_indices.keys())
    #     # logger.debug(f"evaluate_model_2 - class_names  = {class_names}")
    #     # 2024-06-15 17:27:36,060 - __main__ - DEBUG - evaluate_model_1 - class_names  = ['acne_and_rosacea', 'actinic_keratosis', 'atopic_dermatitis', 'healthy_skins']
    #     # 2024-06-15 17:28:04,353 - __main__ - DEBUG - evaluate_model_2 - class_names  = ['acne_and_rosacea', 'actinic_keratosis', 'atopic_dermatitis', 'healthy_skins']

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
        self.evaluate_model_2(model, img_generator_flow_validation)

        # cm = confusion_matrix(y_true, y_pred)
        # plt.figure(figsize=(12.0, 10.0))  # 1.618
        # class_names = [
        #     self.indices_to_class[i] for i in range(len(self.indices_to_class))
        # ]
        # heatmap = sns.heatmap(
        #     cm,
        #     annot=True,
        #     fmt="d",
        #     cmap="Blues",
        #     cbar=False,
        #     xticklabels=class_names,
        #     yticklabels=class_names,
        # )
        # heatmap.set_xticklabels(
        #     heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10
        # )
        # heatmap.set_yticklabels(
        #     heatmap.get_yticklabels(), rotation=0, ha="right", fontsize=10
        # )

        # plt.xlabel("Predicted labels")
        # plt.ylabel("True labels")
        # plt.title("Confusion Matrix")
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # title = f"./img/{timestamp}_confusion_matrix.png"
        # plt.tight_layout()
        # plt.savefig(title)
        # mlflow.log_artifact(title)

        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # predictions_array = []
        # true_array = []

        # for _ in range(1114 // k_BatchSize):
        #     val_imgs, val_targets = next(iter(img_generator_flow_validation))
        #     predictions = model.predict(val_imgs, verbose=0)
        #     predicted_classes = np.argmax(predictions, axis=1)

        #     # Convert val_targets to class indices if it's in multilabel-indicator format
        #     if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
        #         true_classes = np.argmax(val_targets, axis=1)
        #     else:
        #         true_classes = val_targets

        #     predictions_array += list(predicted_classes)
        #     true_array += list(true_classes)

        # class_labels = list(img_generator_flow_validation.class_indices.keys())
        # cm2 = confusion_matrix(true_array, predictions_array)
        # plt.figure(figsize=(12.0, 10.0))
        # sns.heatmap(
        #     cm2,
        #     annot=True,
        #     fmt="g",
        #     cmap="Blues",
        #     xticklabels=class_labels,
        #     yticklabels=class_labels,
        # )
        # plt.xlabel("Predicted labels")
        # plt.ylabel("True labels")
        # plt.title("Confusion Matrix")
        # timestamp2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # title2 = f"./img/{timestamp2}_confusion_matrix2.png"
        # plt.tight_layout()
        # plt.savefig(title2)
        # mlflow.log_artifact(title2)

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

    # ------------------------------------------------------------------------------
    # def evaluate_model(self, model, img_generator_flow_validation):
    #     mlflow.set_tag("debug", "confusion matrix")
    #     self.evaluate_model_1(model, img_generator_flow_validation)
    #     self.evaluate_model_2(model, img_generator_flow_validation)
    #     return

    # -----------------------------------------------------------------------------
    # def evaluate_model(
    #     self,
    #     model: tf.keras.Model,
    #     img_generator_flow_validation: tf.keras.preprocessing.image.DirectoryIterator,
    # ) -> None:
    #     """
    #     Evaluate the model on the test data.
    #     Logs the time taken to evaluate the model and the test loss.
    #     Args:
    #         model (tf.keras.Model): The trained Keras model.
    #         img_generator_flow_validation (tf.keras.preprocessing.image.DirectoryIterator) : Validation set
    #     """
    #     start_time = time.time()

    #     history_dict = model.history.history

    #     plt.figure()
    #     plt.plot(history_dict["categorical_accuracy"], c="r", label="Train Accuracy")
    #     plt.plot(
    #         history_dict["val_categorical_accuracy"], c="b", label="Validation Accuracy"
    #     )
    #     plt.legend()
    #     plt.title("Accuracy vs epochs")
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     title = f"./img/{timestamp}_accuracy.png"
    #     plt.savefig(title)
    #     mlflow.log_artifact(title)

    #     plt.figure()
    #     plt.plot(history_dict["loss"], c="r", label="Train Loss")
    #     plt.plot(history_dict["val_loss"], c="b", label="Validation Loss")
    #     plt.legend()
    #     plt.title("Loss vs epochs")
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     title = f"./img/{timestamp}_loss.png"
    #     plt.savefig(title)
    #     mlflow.log_artifact(title)

    #     # Pour img_generator_flow_validation voir la ligne
    #     # img_generator_flow_validation = img_generator.flow_from_directory()
    #     y_true = img_generator_flow_validation.classes
    #     y_pred = np.argmax(model.predict(img_generator_flow_validation), axis=-1)

    #     accuracy = accuracy_score(y_true, y_pred)
    #     precision = precision_score(y_true, y_pred, average="weighted")
    #     recall = recall_score(y_true, y_pred, average="weighted")
    #     f1 = f1_score(y_true, y_pred, average="weighted")
    #     mlflow.log_metric("Accuracy", accuracy)
    #     mlflow.log_metric("Precision", precision)
    #     mlflow.log_metric("Recall/Sensitivity", recall)
    #     mlflow.log_metric("F1 Score", f1)

    #     cm = confusion_matrix(y_true, y_pred)
    #     plt.figure(figsize=(12.0, 10.0))  # 1.618
    #     class_names = [
    #         self.indices_to_class[i] for i in range(len(self.indices_to_class))
    #     ]
    #     heatmap = sns.heatmap(
    #         cm,
    #         annot=True,
    #         fmt="d",
    #         cmap="Blues",
    #         cbar=False,
    #         xticklabels=class_names,
    #         yticklabels=class_names,
    #     )
    #     heatmap.set_xticklabels(
    #         heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10
    #     )
    #     heatmap.set_yticklabels(
    #         heatmap.get_yticklabels(), rotation=0, ha="right", fontsize=10
    #     )

    #     plt.xlabel("Predicted labels")
    #     plt.ylabel("True labels")
    #     plt.title("Confusion Matrix")
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     title = f"./img/{timestamp}_confusion_matrix.png"
    #     plt.tight_layout()
    #     plt.savefig(title)
    #     mlflow.log_artifact(title)

    #     # -----------------------------------------------------------------------------
    #     # -----------------------------------------------------------------------------
    #     # -----------------------------------------------------------------------------
    #     # -----------------------------------------------------------------------------
    #     # -----------------------------------------------------------------------------
    #     # -----------------------------------------------------------------------------
    #     predictions_array = []
    #     true_array = []

    #     for _ in range(1114 // k_BatchSize):
    #         val_imgs, val_targets = next(iter(img_generator_flow_validation))
    #         predictions = model.predict(val_imgs, verbose=0)
    #         predicted_classes = np.argmax(predictions, axis=1)

    #         # Convert val_targets to class indices if it's in multilabel-indicator format
    #         if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
    #             true_classes = np.argmax(val_targets, axis=1)
    #         else:
    #             true_classes = val_targets

    #         predictions_array += list(predicted_classes)
    #         true_array += list(true_classes)

    #     class_labels = list(img_generator_flow_validation.class_indices.keys())
    #     cm2 = confusion_matrix(true_array, predictions_array)
    #     plt.figure(figsize=(12.0, 10.0))
    #     sns.heatmap(
    #         cm2,
    #         annot=True,
    #         fmt="g",
    #         cmap="Blues",
    #         xticklabels=class_labels,
    #         yticklabels=class_labels,
    #     )
    #     plt.xlabel("Predicted labels")
    #     plt.ylabel("True labels")
    #     plt.title("Confusion Matrix")
    #     timestamp2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     title2 = f"./img/{timestamp2}_confusion_matrix2.png"
    #     plt.tight_layout()
    #     plt.savefig(title2)
    #     mlflow.log_artifact(title2)

    #     # Voir plus tard
    #     # average=None => pour chaque classe
    #     # recall = recall_score(y_true, y_pred, average=None)
    #     # specificity = []
    #     # for i in range(num_classes):
    #     #     tn = np.sum(cm) - (np.sum(cm[:, i]) + np.sum(cm[i, :]) - cm[i, i])
    #     #     fp = np.sum(cm[:, i]) - cm[i, i]
    #     #     specificity.append(tn / (tn + fp))
    #     #
    #     # for i in range(num_classes):
    #     #     print(f"Class {i}: Sensitivity (Recall) = {recall[i]}, Specificity = {specificity[i]}")

    #     mlflow.log_metric("evaluate_model_time", time.time() - start_time)
    #     return

    # -----------------------------------------------------------------------------
    def log_tags_parameters(self) -> None:
        """
        Log some tags and the training parameters to mlflow.
        """

        mlflow.log_param("Classes", k_NbClasses)
        mlflow.log_param("Epochs", self.epochs)
        mlflow.log_param("Steps per epochs", k_StepsPerEpoch)
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
        mlflow.log_metric("log_model", time.time() - start_time)

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
            mlflow.log_metric("total_run_time", time.time() - total_start_time)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")

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
