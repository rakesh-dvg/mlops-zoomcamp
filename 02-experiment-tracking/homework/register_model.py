import os
import pickle
import click
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Set tracking URI and experiment names to match your running server
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
REG_EXPERIMENT_NAME = "random-forest-best-models"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(REG_EXPERIMENT_NAME)


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {}
        for param in ["max_depth", "n_estimators", "min_samples_split", "min_samples_leaf", "random_state"]:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate on Validation and Test sets using the updated RMSE function
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    help="Number of top models that will be evaluated to find the best one"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the HPO experiment metadata to get its ID
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    
    # Query the top_n runs sorted by the best validation RMSE
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Fetch the best model from our new registration experiment based on TEST RMSE
    select_experiment = client.get_experiment_by_name(REG_EXPERIMENT_NAME)
    best_runs = client.search_runs(
        experiment_ids=select_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )
    
    best_run = best_runs[0]

    # --- THE REGISTRATION CODE ---
    # This officially promotes the absolute best model run to the Model Registry
    mlflow.register_model(
        model_uri=f"runs/{best_run.info.run_id}/model",
        name="nyc-taxi-regressor"
    )
    
    print(f"Successfully registered run {best_run.info.run_id} with Test RMSE: {best_run.data.metrics['test_rmse']:.4f}")


if __name__ == '__main__':
    run_register_model()