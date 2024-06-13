from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "https://mlflow-jedha-app-ac2b4eb7451e.herokuapp.com"
MODEL_RUN_ID = "2b18238931bb4c4a9ad733c0e9ea2305"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
artifacts = client.list_artifacts(run_id=MODEL_RUN_ID)
for artifact in artifacts:
    print(artifact.path)