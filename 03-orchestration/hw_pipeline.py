import gc
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
from prefect import flow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-homework3")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@flow(name="HW3 Training Pipeline")
def main_flow():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    
    print("Loading data...")
    columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']
    
    # Read just a safe chunk/sample directly to avoid OOM
    df = pd.read_parquet(url, columns=columns)
    print(f"Total records loaded: {len(df):,}")
    
    # Sample down immediately before any heavy operation
    df = df.sample(n=500000, random_state=42).copy()
    
    print("Preparing data...")
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    print(f"Size after preparation: {len(df):,}")
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    y_train = df['duration'].values
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    del df
    gc.collect()
    
    print("Vectorizing and training model...")
    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    print(f"Model Intercept: {lr.intercept_:.2f}")
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("train_intercept", lr.intercept_)
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")
        
    print(f"Pipeline completed successfully! MLflow run ID: {run.info.run_id}")

if __name__ == "__main__":
    main_flow()