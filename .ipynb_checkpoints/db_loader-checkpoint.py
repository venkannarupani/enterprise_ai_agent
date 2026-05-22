import pandas as pd
from sqlalchemy import create_engine

# Read CSV
csv_path = "data/NTPC Installed Power Capacity.csv"
df = pd.read_csv(csv_path)

# Create SQLite DB
engine = create_engine("sqlite:///NTPC_database.db")

# Load table
df.to_sql(
    "NTPC_data",
    con=engine,
    if_exists="replace",
    index=False
)

print("Database created successfully!")