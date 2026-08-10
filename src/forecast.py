import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_forecasting_model.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_features.pkl"
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "test.csv"
)

OIL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "oil.csv"
)

HOLIDAYS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "holidays_events.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "future_sales_forecast.csv"
)


# ---------------------------------------------------------
# LOAD SAVED PRODUCTION MODEL
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)


# ---------------------------------------------------------
# HISTORICAL DAILY SALES
# ---------------------------------------------------------

def load_historical_daily_sales():

    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["date", "sales"],
        parse_dates=["date"]
    )

    daily_sales = (
        train
        .groupby("date", as_index=False)["sales"]
        .sum()
        .rename(
            columns={
                "sales": "total_sales"
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_sales


# ---------------------------------------------------------
# FUTURE KNOWN FEATURES
# ---------------------------------------------------------

def prepare_future_features():

    # -------------------------
    # Future promotions
    # -------------------------

    future_test = pd.read_csv(
        TEST_PATH,
        usecols=[
            "date",
            "onpromotion"
        ],
        parse_dates=["date"]
    )

    future_promo = (
        future_test
        .groupby(
            "date",
            as_index=False
        )["onpromotion"]
        .sum()
        .rename(
            columns={
                "onpromotion":
                "total_onpromotion"
            }
        )
    )


    # -------------------------
    # Future calendar
    # -------------------------

    future_dates = pd.date_range(
        start=future_test["date"].min(),
        end=future_test["date"].max(),
        freq="D"
    )

    future_df = pd.DataFrame({
        "date": future_dates
    })

    future_df = future_df.merge(
        future_promo,
        on="date",
        how="left"
    )


    # -------------------------
    # Oil prices
    # -------------------------

    oil = pd.read_csv(
        OIL_PATH,
        parse_dates=["date"]
    )

    oil = (
        oil
        .set_index("date")
        .sort_index()
    )

    # Create a continuous daily oil series.
    # Forward-fill non-trading days.
    full_oil_dates = pd.date_range(
        start=oil.index.min(),
        end=oil.index.max(),
        freq="D"
    )

    oil = (
        oil
        .reindex(full_oil_dates)
        .ffill()
        .bfill()
    )

    oil.index.name = "date"

    oil = oil.reset_index()

    future_df = future_df.merge(
        oil,
        on="date",
        how="left"
    )


    # -------------------------
    # National holidays
    # -------------------------

    holidays = pd.read_csv(
        HOLIDAYS_PATH,
        parse_dates=["date"]
    )

    national_holidays = holidays[
        (holidays["locale"] == "National")
        & (holidays["transferred"] == False)
    ][["date"]].drop_duplicates()

    national_holidays[
        "is_national_holiday"
    ] = 1

    future_df = future_df.merge(
        national_holidays,
        on="date",
        how="left"
    )

    future_df[
        "is_national_holiday"
    ] = (
        future_df[
            "is_national_holiday"
        ]
        .fillna(0)
        .astype(int)
    )


    # -------------------------
    # Calendar features
    # -------------------------

    future_df["day_of_week"] = (
        future_df["date"].dt.dayofweek
    )

    future_df["month"] = (
        future_df["date"].dt.month
    )

    future_df["year"] = (
        future_df["date"].dt.year
    )

    future_df["day_of_month"] = (
        future_df["date"].dt.day
    )

    return future_df


# ---------------------------------------------------------
# GENERATE 16-DAY RANDOM FOREST FORECAST
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def generate_future_forecast():

    historical_sales = (
        load_historical_daily_sales()
    )

    future_df = (
        prepare_future_features()
    )

    # Historical actual sales through
    # the final training date.
    history = (
        historical_sales[
            "total_sales"
        ]
        .astype(float)
        .tolist()
    )

    predictions = []


    # Recursive multi-step forecasting
    for _, row in future_df.iterrows():

        feature_row = pd.DataFrame(
            [{
                "day_of_week":
                    row["day_of_week"],

                "month":
                    row["month"],

                "year":
                    row["year"],

                "day_of_month":
                    row["day_of_month"],

                "lag_1":
                    history[-1],

                "lag_7":
                    history[-7],

                "lag_14":
                    history[-14],

                "lag_28":
                    history[-28],

                "rolling_7":
                    np.mean(
                        history[-7:]
                    ),

                "rolling_30":
                    np.mean(
                        history[-30:]
                    ),

                "total_onpromotion":
                    row[
                        "total_onpromotion"
                    ],

                "dcoilwtico":
                    row["dcoilwtico"],

                "is_national_holiday":
                    row[
                        "is_national_holiday"
                    ]
            }]
        )

        # Exact same feature order
        # used during model training.
        feature_row = (
            feature_row[features]
        )

        prediction = model.predict(
            feature_row
        )[0]

        predictions.append(
            float(prediction)
        )

        # Critical for recursive forecasting:
        # tomorrow's lags can use today's
        # predicted value.
        history.append(
            float(prediction)
        )


    forecast = future_df[
        [
            "date",
            "total_onpromotion",
            "dcoilwtico",
            "is_national_holiday"
        ]
    ].copy()

    forecast[
        "predicted_sales"
    ] = np.round(
        predictions,
        2
    )

    return forecast


# ---------------------------------------------------------
# COMPATIBILITY FUNCTION FOR FLASK
# ---------------------------------------------------------

def load_future_forecast():

    return (
        generate_future_forecast()
        .copy()
    )


# ---------------------------------------------------------
# FORECAST SUMMARY
# ---------------------------------------------------------

def get_forecast_summary():

    forecast = (
        load_future_forecast()
    )

    average_sales = (
        forecast[
            "predicted_sales"
        ]
        .mean()
    )

    peak_row = forecast.loc[
        forecast[
            "predicted_sales"
        ].idxmax()
    ]

    lowest_row = forecast.loc[
        forecast[
            "predicted_sales"
        ].idxmin()
    ]

    return {

        "forecast_days":
            len(forecast),

        "average_forecast":
            round(
                float(
                    average_sales
                ),
                2
            ),

        "peak_date":
            peak_row[
                "date"
            ].strftime(
                "%Y-%m-%d"
            ),

        "peak_sales":
            round(
                float(
                    peak_row[
                        "predicted_sales"
                    ]
                ),
                2
            ),

        "lowest_date":
            lowest_row[
                "date"
            ].strftime(
                "%Y-%m-%d"
            ),

        "lowest_sales":
            round(
                float(
                    lowest_row[
                        "predicted_sales"
                    ]
                ),
                2
            )
    }


# ---------------------------------------------------------
# CHART DATA FOR FLASK
# ---------------------------------------------------------

def get_forecast_chart_data():

    forecast = (
        load_future_forecast()
    )

    dates = (
        forecast["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .tolist()
    )

    sales = (
        forecast[
            "predicted_sales"
        ]
        .round(2)
        .tolist()
    )

    return {
        "dates": dates,
        "sales": sales
    }


# ---------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    forecast = (
        generate_future_forecast()
    )

    # Save freshly generated
    # model predictions.
    forecast.to_csv(
        OUTPUT_PATH,
        index=False
    )

    summary = (
        get_forecast_summary()
    )

    print(
        "\nMODEL USED:"
    )

    print(
        type(model)
    )

    print(
        "\nFEATURES:"
    )

    print(
        features
    )

    print(
        "\nGENERATED FORECAST:"
    )

    print(
        forecast
    )

    print(
        "\nFORECAST SUMMARY:"
    )

    print(
        summary
    )

    print(
        "\nForecast generated "
        "directly by saved "
        "Random Forest model."
    )