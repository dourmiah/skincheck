# Execute locally
# Read data (california_housing_market.csv) locally or from S3
# Save parameters and results to mlflow tracking
# Save artifacts to S3

import sys
import time
import mlflow
import argparse
import mlflow.keras
import pandas as pd
import tensorflow as tf


from typing import Tuple
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

# Uncomment the line below if and only if Tensorflow is availble in the virtual env
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# https://medium.com/stackademic/the-ultimate-guide-to-python-logging-simple-effective-and-powerful-9dbae53d9d6d
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Artifacts will be stored under: skincheck-artifacts/2/118b36ff7c8f440db1a1c2bdb98d1008/artifacts/<k_RelativePath>/
# skincheck-artifacts               : S3 bucket
# 2                                 : Experiment ID (defined by mlflow)
# 118b36ff7c8f440db1a1c2bdb98d1008  : Run ID (defined by mlflow)
# artifacts/<k_RelativePath>/       : relative path see artifact_path param below
k_RelativePath = "modeling_housing_market"


# -----------------------------------------------------------------------------
class ModelTrainer:
    # -------------------------------------------------------------------------
    def __init__(self, epochs: int, batch_size: int) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

    # -------------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from a local CSV file or from S3.
        Logs the time taken to load the data.
        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            start_time = time.time()
            # Uncomment the line below to load data from S3
            # data = pd.read_csv("https://skincheck-bucket.s3.eu-west-3.amazonaws.com/skincheck-dataset/california_housing_market.csv")
            data = pd.read_csv("./data/california_housing_market.csv")
            mlflow.log_metric("load_data_time", time.time() - start_time)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    # -------------------------------------------------------------------------
    def preprocess_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Preprocess the dataset.
        Splits the data into training and testing sets.
        Scales the features using StandardScaler.
        Logs the time taken to preprocess the data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Processed data splits.
        """
        start_time = time.time()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlflow.log_metric("preprocess_data_time", time.time() - start_time)
        return X_train_scaled, X_test_scaled, y_train, y_test

    # -------------------------------------------------------------------------
    def build_model(self, input_shape: int) -> tf.keras.Model:
        """
        Build and compile a Keras sequential model.
        Args:
            input_shape (int): The shape of the input data.
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
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

    # -------------------------------------------------------------------------
    def train_model(
        self, model: tf.keras.Model, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Train the model on the training data.
        Logs the time taken to train the model.
        Args:
            model (tf.keras.Model): The compiled Keras model.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        Returns:
            Tuple[tf.keras.Model, tf.keras.callbacks.History]: The trained model and training history.
        """
        start_time = time.time()

        # Uncomment the line below if and only if Tensorflow is availble in the virtual env
        # checkpoint = ModelCheckpoint(
        #     "best_model.h5", save_best_only=True, monitor="val_loss"
        # )
        # early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

        history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            # Uncomment the line below if and only if Tensorflow is availble in the virtual env
            # callbacks=[checkpoint, early_stopping],
        )
        mlflow.log_metric("train_model_time", time.time() - start_time)
        return model, history

    # -------------------------------------------------------------------------
    def evaluate_model(
        self, model: tf.keras.Model, X_test: pd.DataFrame, y_test: pd.Series
    ) -> float:
        """
        Evaluate the model on the test data.
        Logs the time taken to evaluate the model and the test loss.
        Args:
            model (tf.keras.Model): The trained Keras model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
        Returns:
            float: The loss on the test data.
        """
        start_time = time.time()
        loss = model.evaluate(X_test, y_test)
        mlflow.log_metric("evaluate_model_time", time.time() - start_time)
        mlflow.log_metric("test_loss", loss)
        return loss

    # -------------------------------------------------------------------------
    def log_parameters(self) -> None:
        """
        Log the training parameters to mlflow.
        """
        mlflow.log_param("epochs", self.epochs)
        mlflow.log_param("batch_size", self.batch_size)

    # -------------------------------------------------------------------------
    def log_model(self, model: tf.keras.Model, X_train: pd.DataFrame) -> None:
        """
        Log the trained model to mlflow with a signature.
        Args:
            model (tf.keras.Model): The trained Keras model.
            X_train (pd.DataFrame): Training features.
        """
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.keras.log_model(
            model=model,
            artifact_path=k_RelativePath,
            registered_model_name="keras_sequential",
            signature=signature,
        )

    # -------------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the full training and evaluation pipeline:
        - Load data
        - Preprocess data
        - Log parameters
        - Build model
        - Train model
        - Evaluate model
        - Log model
        - Log total runtime
        """
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


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n<START>\n")
    print("OS                   : ", sys.platform)
    print("Python version       : ", sys.version.split("|")[0])
    print("mlflow version       : ", mlflow.__version__)
    print("TensorFlow version   : ", tf.__version__)
    print("\nTraining started     :")

    logger.info("<START>")
    logger.info(f"OS                   : {sys.platform}")
    logger.info(f"Python version       : {sys.version.split('|')[0]}")
    logger.info(f"mlflow version       : {mlflow.__version__}")
    logger.info(f"TensorFlow version   : {tf.__version__}")
    logger.info("Training started     :")

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    trainer = ModelTrainer(args.epochs, args.batch_size)
    trainer.run()

    logger.info(f"Training finished    :")
    logger.info(f"Training time        : {(time.time()-start_time):.3f} sec.")
    logger.info(f"<STOP>")
