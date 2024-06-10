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


if __name__ == "__main__":

    print()
    print()
    print("<START>")
    print()

    print("OS                   : ", sys.platform)
    print("Python version       : ", sys.version.split("|")[0])
    print("mlflow version       : ", mlflow.__version__)
    print("scikit-learn version : ", sklearn.__version__)
    print()

    print("Training started     :")

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, required=True)
    parser.add_argument("--min_samples_split", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_csv(
        "https://skincheck-bucket.s3.eu-west-3.amazonaws.com/skincheck-dataset/california_housing_market.csv"
    )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_estimators = args.n_estimators
    min_samples_split = args.min_samples_split

    model = Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            (
                "Regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators, min_samples_split=min_samples_split
                ),
            ),
        ]
    )

    # Log experiment to MLFlow
    with mlflow.start_run():
        # Call mlflow autolog inside the run context
        # mlflow.sklearn.autolog(log_models=False)

        # Enregistrer les paramètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_split", min_samples_split)

        model.fit(X_train, y_train)
        # predictions = model.predict(X_train)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Calculer les métriques
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        # Enregistrer les métriques
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)

        # Log model separately to have more flexibility on setup
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     # Les artifacts will be under       : skincheck-artifacts/2/118b36ff7c8f440db1a1c2bdb98d1008/artifacts/modeling_housing_market/
        #     # skincheck-artifacts               : Compartiment S3
        #     # 2                                 : Experiment ID
        #     # 118b36ff7c8f440db1a1c2bdb98d1008  : Run ID
        #     # artifacts/modeling_housing_market : relative path see artifact_path param
        #     artifact_path="modeling_housing_market",
        #     registered_model_name="random_forest",
        #     signature=infer_signature(X_train, predictions),
        # )
        signature = infer_signature(X_train, train_predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="modeling_housing_market",
            registered_model_name="random_forest",
            signature=signature,
        )

    print(f"Training finished    :")
    print(f"Training time        : {(time.time()-start_time):.3f} sec.")

    print()
    print("<STOP>")
    print()
    print()
