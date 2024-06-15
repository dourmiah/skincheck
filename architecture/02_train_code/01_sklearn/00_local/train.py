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
        mlflow.sklearn.autolog(log_models=False)

        model.fit(X_train, y_train)
        predictions = model.predict(X_train)

        # Log model separately to have more flexibility on setup
        mlflow.sklearn.log_model(
            sk_model=model,
            # Les artifacts will be under       : skincheck-artifacts/2/118b36ff7c8f440db1a1c2bdb98d1008/artifacts/modeling_housing_market/
            # skincheck-artifacts               : Compartiment S3
            # 2                                 : Experiment ID
            # 118b36ff7c8f440db1a1c2bdb98d1008  : Run ID
            # artifacts/modeling_housing_market : relative path see artifact_path param below
            artifact_path="modeling_housing_market",
            registered_model_name="random_forest",
            signature=infer_signature(X_train, predictions),
        )

    print(f"Training finished    :")
    print(f"Training time        : {(time.time()-start_time):.3f} sec.")

    print()
    print("<STOP>")
    print()
    print()
