import os
import gc
import pickle
import pandas as pd
from datetime import datetime
from prefect import task, flow, get_run_logger

import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

@task(retries=3, retry_delay_seconds=10)
def read_dataframe(filename: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Loading data from: {filename}")
    
    df = pd.read_parquet(filename)
    
    # Question 3 Verification
    logger.info(f"Question 3 -> Number of records loaded: {len(df)}")
    print(f"Raw records loaded: {len(df)}")
    
    # Question 4 Data preparation logic
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    logger.info(f"Question 4 -> Size of filtered result: {len(df)}")
    return df

@task
def train_model(df: pd.DataFrame):
    logger = get_run_logger()
    
    # Connect to your local running Docker MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-homework-3")
    
    categorical = ['PULocationID', 'DOLocationID']
    
    logger.info("Converting target column to float32 to conserve memory...")
    y_train = df['duration'].values.astype('float32')
    
    logger.info("Extracting records array for feature engineering...")
    dicts = df[categorical].to_dict(orient='records')
    
    # RAM Lifesaver: Wipe out the dense dataframe before building the matrix
    del df
    gc.collect()
    
    logger.info("Fitting DictVectorizer sparse matrix...")
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(dicts)
    
    # Wipe out dictionaries to optimize RAM further
    del dicts
    gc.collect()
    
    logger.info("Training Linear Regression model...")
    with mlflow.start_run():
        lr = LinearRegression(n_jobs=-1)
        lr.fit(X_train, y_train)
        
        # Question 5 Output
        logger.info(f"Question 5 -> Model Intercept: {lr.intercept_}")
        print(f"\n======================================")
        print(f"SUCCESS! Model Intercept: {lr.intercept_}")
        print(f"======================================\n")
        
        # Track metric details
        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        # --- FIX FOR THE CRASH: Save and upload files directly ---
        logger.info("Saving model and vectorizer binaries locally...")
        with open("model.pkl", "wb") as f_model:
            pickle.dump(lr, f_model)
            
        with open("dict_vectorizer.bin", "wb") as f_dv:
            pickle.dump(dv, f_dv)
            
        logger.info("Uploading artifacts directly to MLflow to bypass API crash...")
        mlflow.log_artifact("model.pkl", artifact_path="model")
        mlflow.log_artifact("dict_vectorizer.bin", artifact_path="artifacts")
        
    return lr, dv

@flow(name="Taxi Homework Pipeline")
def main_flow():
    # Target dataset for Homework 3
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    
    df = read_dataframe(url)
    train_model(df)

if __name__ == "__main__":
    main_flow()