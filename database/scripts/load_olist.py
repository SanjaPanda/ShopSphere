from pathlib import Path
import os
import json

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, Integer, BigInteger, Float, String, Boolean, DateTime, Text


from shared.logging_config import setup_logger

# ------------------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------------------

logger = setup_logger(__name__)

# ------------------------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------------------------

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)

# ------------------------------------------------------------------------------
# Dataset Configuration
# ------------------------------------------------------------------------------

DATA_DIR = Path("data/processed")

FILES = {
    "customers": "customers_processed.csv",
    "orders": "orders_processed.csv",
    "order_items": "order_items_processed.csv",
    "payments": "payments_processed.csv",
    "products": "products_processed.csv",
    "reviews": "reviews_processed.csv",
    "sellers": "sellers_processed.csv",
    "geolocation": "geolocations_processed.csv",
    "category_translation": "translations_processed.csv",
}


TABLE_SCHEMA_DIR = Path("data/table_schema")

TABLE_SCHEMAS = {
    "customers": "customers_schema.json",
    "orders": "orders_schema.json",
    "order_items": "order_items_schema.json",
    "payments": "payments_schema.json", 
    "products": "products_schema.json",
    "reviews": "reviews_schema.json",
    "sellers": "sellers_schema.json",
    "geolocation": "geolocations_schema.json",
    "category_translation": "translations_schema.json",

}
# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

def map_dtype_to_sqlalchemy(dtype_str: str):
    """Map schema JSON 'Data Type' string to a SQLAlchemy column type."""
    dtype_str = dtype_str.lower()
    if 'datetime' in dtype_str:
        return DateTime()
    elif dtype_str in ('int64', 'int32', 'int'):
        return BigInteger()
    elif dtype_str in ('float64', 'float32', 'float'):
        return Float()
    elif dtype_str in ('bool', 'boolean'):
        return Boolean()
    elif dtype_str in ('str', 'string', 'object'):
        return Text()
    else:
        return Text()  # safe fallback for anything unmapped

def load_sql_schema_from_json(schema_path: str) -> dict:
    """Read a table's schema JSON file and build the dtype dict for to_sql()."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return {
        col['Column Name']: map_dtype_to_sqlalchemy(col['Data Type'])
        for col in schema
    }


def get_datetime_columns(schema: dict) -> list:
    """Read schema JSON and return any columns that should be parsed as dates."""
    return [
        col['Column Name']
        for col in schema
        if 'datetime' in str(col.get('Data Type', '')).lower()
    ]


def load_table(table_name: str):
    """Load a CSV file into PostgreSQL."""

    csv_path = DATA_DIR / FILES[table_name]
    schema_path = TABLE_SCHEMA_DIR / TABLE_SCHEMAS[table_name]

    logger.info("Loading table '%s'", table_name)

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return
    
    if not schema_path.exists():
        logger.error("Schema file not found: %s", schema_path)
        return

    try:
        schema = json.load(open(schema_path, 'r', encoding='utf-8'))
        datetime_columns = get_datetime_columns(schema)
        df = pd.read_csv(csv_path, parse_dates=datetime_columns if datetime_columns else None)
        

        logger.info(
            "Read %s rows and %s columns",
            f"{len(df):,}",
            len(df.columns),
        )

        sql_dtype_schema = load_sql_schema_from_json(schema_path)

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5000,
            dtype=sql_dtype_schema
        )

        logger.info(
            "Successfully loaded '%s' (%s rows)",
            table_name,
            f"{len(df):,}",
        )

    except Exception:
        logger.exception("Failed loading table '%s'", table_name)


def main():

    logger.info("=" * 70)
    logger.info("Starting Olist Data Load Pipeline")
    logger.info("=" * 70)

    for table_name in FILES:
        load_table(table_name)

    logger.info("=" * 70)
    logger.info("Data loading completed.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()