import mlflow
import mlflow.sklearn
import dagshub

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np

# Initialize DagsHub + MLflow
dagshub.init(
    repo_owner="khaledahamed8080",
    repo_name="MlFlow-Demo",
    mlflow=True
)

# Load regression dataset
X, y = fetch_california_housing(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="linear_regression_model",registered_model_name="HousePriceRegressor")

    print("Regression metrics:")
    print(f"MSE  : {mse}")
    print(f"RMSE : {rmse}")
    print(f"MAE  : {mae}")
    print(f"R2   : {r2}")
