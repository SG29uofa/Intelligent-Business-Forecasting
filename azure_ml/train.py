import argparse
import os
import pandas as pd

# Input from Azure ML

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Path to the raw forecasting data"
)

args = parser.parse_args()

# Build path to train.csv
train_file = os.path.join(args.data, "train.csv")

print("Training data location:")
print(train_file)

# Loading dataset

df = pd.read_csv(train_file)

df["date"] = pd.to_datetime(df["date"])


# Validation

print("\nDataset successfully loaded.")
print("Shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Stores:", df["store_nbr"].nunique())
print("Product families:", df["family"].nunique())
print("Missing values:", df.isnull().sum().sum())

# FE
# Sorting chronologically
df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

# Basic calendar features from the date
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

# Historical sales lag features

group_cols = ["store_nbr", "family"]

df["sales_lag_1"] = df.groupby(group_cols)["sales"].shift(1)
df["sales_lag_7"] = df.groupby(group_cols)["sales"].shift(7)
df["sales_lag_14"] = df.groupby(group_cols)["sales"].shift(14)
df["sales_lag_28"] = df.groupby(group_cols)["sales"].shift(28)

# Rolling sales features using only previous sales

shifted_sales = df.groupby(group_cols)["sales"].shift(1)

df["sales_rolling_mean_7"] = (
    shifted_sales
    .groupby([df["store_nbr"], df["family"]])
    .rolling(window=7)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

df["sales_rolling_mean_28"] = (
    shifted_sales
    .groupby([df["store_nbr"], df["family"]])
    .rolling(window=28)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

print("\nRolling sales features created.")

print("\nLag features created.")

# Remove rows that do not have complete lag history

lag_columns = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28"
]

df = df.dropna(subset=lag_columns).copy()

print("Rows after removing incomplete lag history:", len(df))

print("\nFeature engineering completed.")

print(
    df[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "onpromotion",
            "year",
            "month",
            "day",
            "day_of_week",
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_28"
        ]
    ].head()
)

test_days = 76

max_date = df["date"].max()
cutoff_date = max_date - pd.Timedelta(days=test_days)

train_df = df[df["date"] <= cutoff_date].copy()
test_df = df[df["date"] > cutoff_date].copy()

print("\nChronological split completed.")
print("Training period:", train_df["date"].min(), "to", train_df["date"].max())
print("Testing period:", test_df["date"].min(), "to", test_df["date"].max())

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# Prepare features for machine learning
feature_columns = [
    "store_nbr",
    "family",
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_mean_28"
]

target_column = "sales"

X_train = train_df[feature_columns].copy()
y_train = train_df[target_column].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df[target_column].copy()

print("\nMachine learning features prepared.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Encode the categorical product family column

X_train = pd.get_dummies(
    X_train,
    columns=["family"],
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=["family"],
    dtype=int
)

# Ensure train and test contain exactly the same feature columns
X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

print("\nCategorical encoding completed.")
print("Number of model features:", X_train.shape[1])
print("Train and test columns match:", X_train.columns.equals(X_test.columns))

# Random Forest model

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("\nRandom Forest model created.")

# Use a sample first to validate the training pipeline

sample_size = min(500000, len(X_train))

X_train_sample = X_train.sample(
    n=sample_size,
    random_state=42
)

y_train_sample = y_train.loc[X_train_sample.index]

print("\nTraining Random Forest...")
print("Training sample rows:", len(X_train_sample))

rf_model.fit(X_train_sample, y_train_sample)

print("Random Forest training completed.")

# Make predictions on the test period

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("\nGenerating predictions...")

predictions = rf_model.predict(X_test)

# Evaluate the model

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# MAPE only where actual sales are greater than zero
non_zero_mask = y_test > 0

mape = np.mean(
    np.abs(
        (y_test[non_zero_mask] - predictions[non_zero_mask])
        / y_test[non_zero_mask]
    )
) * 100

print("\nRandom Forest Evaluation")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("MAPE:", round(mape, 2), "%")

# Save the trained model as an Azure ML job output

import joblib

os.makedirs("outputs", exist_ok=True)

model_path = "outputs/random_forest_model.joblib"

joblib.dump(rf_model, model_path)

print("\nModel saved successfully.")
print("Model path:", model_path)


# Save the feature schema used by the trained model

feature_schema_path = "outputs/feature_columns.joblib"

joblib.dump(
    list(X_train.columns),
    feature_schema_path
)

print("Feature schema saved successfully.")
print("Feature schema path:", feature_schema_path)
print("Number of saved model features:", len(X_train.columns))

# Save model evaluation metrics

import json

metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "training_sample_rows": int(len(X_train_sample)),
    "testing_rows": int(len(X_test)),
    "model_features": int(X_train.shape[1])
}

metrics_path = "outputs/metrics.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nEvaluation metrics saved successfully.")
print("Metrics path:", metrics_path)

