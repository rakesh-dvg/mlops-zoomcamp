import os
import pickle
import click
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    help="Number of top models to evaluate that were saved from hpo.py"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top model runs from the hpo experiment sorted by rmse
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],  # FIXED HERE
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    # Load dataset
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    
    # Slice to match our lightweight training subset
    X_train, y_train = X_train[:10000], y_train[:10000]

    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    # Try loading test.pkl, fallback to val.pkl if test doesn't exist in homework output
    test_path = os.path.join(data_path, "test.pkl")
    X_test, y_test = load_pickle(test_path) if os.path.exists(test_path) else (X_val, y_val)

    # Train and register the top-scoring model
    for run in runs:
        best_run_id = run.info.run_id
        
        # Safely extract and parse parameters
        best_params = {}
        for k, v in run.data.params.items():
            if k in RF_PARAMS:
                best_params[k] = float(v) if '.' in str(v) else int(v)
        
        with mlflow.start_run():
            rf = RandomForestRegressor(**best_params)
            rf.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = rf.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)

            mlflow.log_params(best_params)
            mlflow.log_metric("test_rmse", rmse)
            
            # Register the model
            mlflow.sklearn.log_model(rf, artifact_path="model", registered_model_name="random-forest-best-taxi-model")
        
        print(f"Registered best run {best_run_id} with Test RMSE: {rmse}")
        break  # Register only the top 1

if __name__ == '__main__':
    run_register_model()