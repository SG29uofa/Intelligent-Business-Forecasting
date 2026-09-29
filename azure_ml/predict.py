import argparse
import joblib
import pandas as pd

# Receive the registered model from Azure ML
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to the registered Random Forest model"
)

args = parser.parse_args()

print("Loading registered model from:")
print(args.model)

# Load model
model = joblib.load(args.model)

print("\nRegistered model loaded successfully.")
print("Expected features:", model.n_features_in_)

# Create one sample containing the exact 45 encoded features
feature_names = list(model.feature_names_in_)

sample = pd.DataFrame(
    [[0.0] * len(feature_names)],
    columns=feature_names
)

# Populate known numerical features with realistic example values
sample.loc[0, "store_nbr"] = 1
sample.loc[0, "onpromotion"] = 0
sample.loc[0, "year"] = 2017
sample.loc[0, "month"] = 8
sample.loc[0, "day"] = 16
sample.loc[0, "day_of_week"] = 2

sample.loc[0, "sales_lag_1"] = 100
sample.loc[0, "sales_lag_7"] = 95
sample.loc[0, "sales_lag_14"] = 90
sample.loc[0, "sales_lag_28"] = 85
sample.loc[0, "sales_rolling_mean_7"] = 98
sample.loc[0, "sales_rolling_mean_28"] = 92

# Use one encoded product-family column
family_columns = [
    col for col in feature_names
    if col.startswith("family_")
]

if family_columns:
    sample.loc[0, family_columns[0]] = 1

# Generate prediction
prediction = model.predict(sample)

print("\nInference completed successfully.")
print("Predicted sales:", round(float(prediction[0]), 2))
