. "./secrets.ps1"

# If you want to pass parameters pick option 2
mlflow run --experiment-name $env:MLFLOW_EXPERIMENT_NAME .

# mlflow run --experiment-name $env:MLFLOW_EXPERIMENT_NAME -P epochs=10 -P batch_size=1000 . 