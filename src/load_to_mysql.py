import pandas as pd
from sqlalchemy import create_engine


# -----------------------------
# MySQL connection
# -----------------------------

MYSQL_USER = "root"
MYSQL_PASSWORD = "$Weety29"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DATABASE = "intelligent_business_forecasting"


engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)


# -----------------------------
# File paths
# -----------------------------

RAW_PATH = "../data/raw/"


# -----------------------------
# Load oil prices
# -----------------------------

oil = pd.read_csv(
    RAW_PATH + "oil.csv",
    parse_dates=["date"]
)

print("Oil CSV shape:", oil.shape)
print("Missing oil prices:", oil["dcoilwtico"].isna().sum())


oil.to_sql(
    name="oil_prices",
    con=engine,
    if_exists="replace",
    index=False
)

print("oil_prices loaded successfully.")

# -----------------------------
# Load transactions
# -----------------------------

transactions = pd.read_csv(
    RAW_PATH + "transactions.csv",
    parse_dates=["date"]
)

print("\nTransactions CSV shape:", transactions.shape)

transactions.to_sql(
    name="transactions",
    con=engine,
    if_exists="replace",
    index=False
)

print("transactions loaded successfully.")


# -----------------------------
# Load holidays/events
# -----------------------------

holidays = pd.read_csv(
    RAW_PATH + "holidays_events.csv",
    parse_dates=["date"]
)

print("\nHolidays CSV shape:", holidays.shape)

holidays.to_sql(
    name="holidays_events",
    con=engine,
    if_exists="replace",
    index=False
)

print("holidays_events loaded successfully.")

# -----------------------------
# Load stores
# -----------------------------

stores = pd.read_csv(
    RAW_PATH + "stores.csv"
)

print("\nStores CSV shape:", stores.shape)

stores = stores.rename(columns={"type": "store_type"})

stores.to_sql(
    name="stores",
    con=engine,
    if_exists="replace",
    index=False
)

print("stores loaded successfully.")

# -----------------------------
# Load historical sales
# -----------------------------

TRAIN_PATH = RAW_PATH + "train.csv"

chunk_size = 100000
first_chunk = True
total_rows = 0

print("\nLoading historical sales...")

for chunk in pd.read_csv(
    TRAIN_PATH,
    parse_dates=["date"],
    chunksize=chunk_size
):
    chunk.to_sql(
        name="sales",
        con=engine,
        if_exists="replace" if first_chunk else "append",
        index=False,
        method="multi",
        chunksize=5000
    )

    first_chunk = False
    total_rows += len(chunk)

    print(f"Loaded {total_rows:,} rows")

print("\nSales loaded successfully.")