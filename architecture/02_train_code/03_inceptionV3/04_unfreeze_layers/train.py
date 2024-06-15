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


from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# https://stackoverflow.com/questions/66908259/how-to-fine-tune-inceptionv3-in-keras


#  Code de Dominique pour afficher la matrice de confusion "correctement"
# 1162 = nb images train
# 8 = batch size

# predictionsarray = []
# true_array = []
# for  in range(1162//8):
#     val_imgs, val_targets = next(directory_generator_val)
#     predictions = modelconv.predict(val_imgs)
#     predicted_classes = np.argmax(predictions, axis=1)
#     # true_classes = directory_generator_val.classes
#     true_classes = val_targets
#     predictions_array += list(predicted_classes)
#     true_array += list(true_classes)

# class_labels = list(directory_generator_val.class_indices.keys())
# cm = confusion_matrix(true_array, predictions_array)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Prélude
# See volumes:[..] in MLproject file
# Don't forget that the app run in /home/app and it needs to access data that are in /home/data/train
# Do not change the path here. Change it in MLproject
k_DataDir = "../data/train"
k_NbClasses = 4
k_L2 = 0.01
k_BatchSize = 32
k_Epochs = 2  # 25, 2 to debug, 8 to go fast...
k_StepsPerEpoch = 20
k_LearningRate = 0.001
k_RelativePath = "skin_check"
k_Img_Width = 299
k_Img_Height = 299
k_Author = "Philippe"
k_XpPhase = "Investigation"


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
        # ! ATTENTION
        # base_model.trainable = False

        # print("\n\nSummary about InceptionV3 : ")
        # base_model.summary()
        # print("End of summary\n\n")

        # from tensorflow.keras.utils import plot_model
        # from IPython.display import Image
        # plot_model(model, to_file='convnet.png', show_shapes=True,show_layer_names=True)
        # Image(filename='convnet.png')

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
                tf.keras.layers.GlobalAveragePooling2D(),
                # tf.keras.layers.Flatten(),
                # Ajouter 2 couches dense. Comme le model sort en 5x5x2048 on met 1024 et 512
                # Je ne sais pas trop si il faut une régule L2 partout ou juste sur la dernière couche
                # tf.keras.layers.Dense(
                #     1024,
                #     activation="relu",
                #     kernel_regularizer=tf.keras.regularizers.l2(k_L2),
                # ),
                tf.keras.layers.Dense(
                    512,
                    activation="relu",
                    # kernel_regularizer=tf.keras.regularizers.l2(k_L2),
                ),
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
    def train_model(self, model, img_generator_flow_train, img_generator_flow_valid):
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
            validation_data=img_generator_flow_valid,
            steps_per_epoch=k_StepsPerEpoch,
            epochs=k_Epochs,
            callbacks=[early_stopping],
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
        plt.figure(figsize=(12.0, 10.0))  # 1.618
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

        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        predictions_array = []
        true_array = []

        for _ in range(1114 // k_BatchSize):
            val_imgs, val_targets = next(iter(img_generator_flow_valid))
            predictions = model.predict(val_imgs, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            # Convert val_targets to class indices if it's in multilabel-indicator format
            if len(val_targets.shape) > 1 and val_targets.shape[1] > 1:
                true_classes = np.argmax(val_targets, axis=1)
            else:
                true_classes = val_targets

            predictions_array += list(predicted_classes)
            true_array += list(true_classes)

        class_labels = list(img_generator_flow_valid.class_indices.keys())
        cm2 = confusion_matrix(true_array, predictions_array)
        plt.figure(figsize=(12.0, 10.0))
        sns.heatmap(
            cm2,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        timestamp2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        title2 = f"./img/{timestamp2}_confusion_matrix2.png"
        plt.tight_layout()
        plt.savefig(title2)
        mlflow.log_artifact(title2)

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

    # Définir les seeds pour obtenir des résultats reproductibles
    seed = 0
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()

    os.makedirs("./img", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    trainer = ModelTrainer(args.epochs, args.batch_size)
    trainer.run()

    print(f"Training time        : {(time.time()-start_time):.3f} sec.")
