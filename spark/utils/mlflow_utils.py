import mlflow


def init_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Initialize MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
