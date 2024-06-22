import sys
import argparse
import pandas as pd
import time
import mlflow
import sklearn

from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, r2_score
import mlflow.sklearn

# Artifacts will be stored under    : skincheck-artifacts/2/118b36ff7c8f440db1a1c2bdb98d1008/artifacts/<k_RelativePath>/
# skincheck-artifacts               : Compartiment S3
# 2                                 : Experiment ID
# 118b36ff7c8f440db1a1c2bdb98d1008  : Run ID
# artifacts/<k_RelativePath>/       : relative path see artifact_path param below
k_RelativePath = "modeling_housing_market"


class ModelTrainer:
    def __init__(self, n_estimators, min_samples_split):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split

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
        mlflow.log_metric("preprocess_data_time", time.time() - start_time)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        start_time = time.time()
        model = Pipeline(
            steps=[
                ("standard_scaler", StandardScaler()),
                (
                    "Regressor",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        min_samples_split=self.min_samples_split,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        mlflow.log_metric("train_model_time", time.time() - start_time)
        return model

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        start_time = time.time()
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        mlflow.log_metric("evaluate_model_time", time.time() - start_time)

        return {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "signature": infer_signature(X_train, train_predictions),
        }

    def log_parameters(self):
        mlflow.log_param("n_estimators", self.n_estimators)
        mlflow.log_param("min_samples_split", self.min_samples_split)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            if key != "signature":
                mlflow.log_metric(key, value)

    def log_model(self, model, signature):
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=k_RelativePaths,
            registered_model_name="random_forest",
            signature=signature,
        )

    def run(self):
        with mlflow.start_run():
            total_start_time = time.time()
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            self.log_parameters()
            model = self.train_model(X_train, y_train)
            metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            self.log_metrics(metrics)
            self.log_model(model, metrics["signature"])
            mlflow.log_metric("total_run_time", time.time() - total_start_time)


if __name__ == "__main__":
    print("\n\n<START>\n")
    print("OS                   : ", sys.platform)
    print("Python version       : ", sys.version.split("|")[0])
    print("mlflow version       : ", mlflow.__version__)
    print("scikit-learn version : ", sklearn.__version__)
    print("\nTraining started     :")

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, required=True)
    parser.add_argument("--min_samples_split", type=int, required=True)
    args = parser.parse_args()

    trainer = ModelTrainer(args.n_estimators, args.min_samples_split)
    trainer.run()

    print(f"Training finished    :")
    print(f"Training time        : {(time.time()-start_time):.3f} sec.")
    print("\n<STOP>\n\n")
