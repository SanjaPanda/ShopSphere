from pathlib import Path
from datetime import datetime
import json
import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from shared.logging_config import setup_logger

# ==============================================================================
# Logger
# ==============================================================================

logger = setup_logger(__name__)

# ==============================================================================
# Environment
# ==============================================================================

load_dotenv()

DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DATABASE_URL)

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path("data/processed")

REPORT_DIR = Path("database/reports/validation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

# ==============================================================================
# Report Object
# ==============================================================================

report = {
    "pipeline_run_id": str(uuid.uuid4()),
    "timestamp": datetime.now().isoformat(),
    "overall_status": "PASS",
    "summary": {
        "tables_checked": 0,
        "tables_passed": 0,
        "tables_failed": 0,
    },
    "tables": [],
}


# ==============================================================================
# Validation
# ==============================================================================

def normalize_dtype(dtype_value):
    dtype_str = str(dtype_value).lower()
    if dtype_str == "object":
        return "str"
    if dtype_str in {"string", "str"}:
        return "str"
    if dtype_str.startswith("int"):
        return "int64"
    if dtype_str.startswith("float"):
        return "float64"
    if dtype_str.startswith("datetime64"):
        return "datetime64[ns]"
    if dtype_str.startswith("bool"):
        return "bool"
    return dtype_str


def validate_table(table_name: str, filename: str):

    logger.info("=" * 70)
    logger.info("Validating table: %s", table_name)

    csv_path = DATA_DIR / filename

    result = {
        "table_name": table_name,
        "status": "PASS",
        "table_exists": False,
        "csv_rows": 0,
        "db_rows": 0,
        "row_match": False,
        "csv_columns": 0,
        "db_columns": 0,
        "column_count_match": False,
        "column_names_match": False,
        "schema_file_exists": False,
        "schema_columns": 0,
        "schema_match": False,
        "schema_issues": [],
        "message": "",
    }

    # ------------------------------------------------------------------
    # CSV Exists
    # ------------------------------------------------------------------

    if not csv_path.exists():

        result["status"] = "FAIL"
        result["message"] = f"CSV not found: {filename}"

        logger.error(result["message"])

        return result

    df = pd.read_csv(csv_path)

    csv_rows = len(df)
    csv_columns = list(df.columns)

    result["csv_rows"] = csv_rows
    result["csv_columns"] = len(csv_columns)

    schema_path = TABLE_SCHEMA_DIR / f"{Path(filename).stem.replace('_processed', '')}_schema.json"
    result["schema_file_exists"] = False
    result["schema_columns"] = 0
    result["schema_match"] = False
    result["schema_issues"] = []

    if not schema_path.exists():
        result["status"] = "FAIL"
        result["message"] = f"Schema file not found: {schema_path.name}"
        logger.error(result["message"])
        return result

    result["schema_file_exists"] = True

    try:
        schema_entries = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["status"] = "FAIL"
        result["message"] = f"Failed reading schema file: {exc}"
        logger.error(result["message"])
        return result

    schema_columns = [entry["Column Name"] for entry in schema_entries if "Column Name" in entry]
    schema_dtypes = {
        entry["Column Name"]: normalize_dtype(entry.get("Data Type", ""))
        for entry in schema_entries
        if "Column Name" in entry
    }
    schema_nullables = {
        entry["Column Name"]: bool(entry.get("Nullable", False))
        for entry in schema_entries
        if "Column Name" in entry
    }

    result["schema_columns"] = len(schema_columns)

    missing_in_csv = sorted(set(schema_columns) - set(csv_columns))
    extra_in_csv = sorted(set(csv_columns) - set(schema_columns))

    if missing_in_csv:
        result["schema_issues"].append(
            f"CSV missing schema columns: {missing_in_csv}"
        )

    if extra_in_csv:
        result["schema_issues"].append(
            f"CSV has extra columns not in schema: {extra_in_csv}"
        )

    inspector = inspect(engine)

    # ------------------------------------------------------------------
    # Table Exists
    # ------------------------------------------------------------------

    if not inspector.has_table(table_name):

        result["status"] = "FAIL"
        result["message"] = "Table does not exist."

        logger.error(result["message"])

        return result

    result["table_exists"] = True

    logger.info("✓ Table exists")

    # ------------------------------------------------------------------
    # Row Count
    # ------------------------------------------------------------------

    with engine.connect() as conn:

        db_rows = conn.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar()

    result["db_rows"] = db_rows
    result["row_match"] = csv_rows == db_rows

    if result["row_match"]:
        logger.info("✓ Row count matches (%s)", csv_rows)
    else:
        logger.error(
            "✗ Row mismatch | CSV=%s DB=%s",
            csv_rows,
            db_rows,
        )
        result["status"] = "FAIL"

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    db_columns = [c["name"] for c in inspector.get_columns(table_name)]

    result["db_columns"] = len(db_columns)
    result["column_count_match"] = (
        len(csv_columns) == len(db_columns)
    )

    missing_schema_db = sorted(set(schema_columns) - set(db_columns))
    extra_schema_db = sorted(set(db_columns) - set(schema_columns))

    if missing_schema_db:
        result["schema_issues"].append(
            f"Database missing schema columns: {missing_schema_db}"
        )

    if extra_schema_db:
        result["schema_issues"].append(
            f"Database has extra columns not in schema: {extra_schema_db}"
        )

    for column_name in schema_columns:
        if column_name not in csv_columns:
            continue

        expected_dtype = schema_dtypes.get(column_name)
        if expected_dtype == "datetime64[ns]":
            parsed_series = pd.to_datetime(df[column_name], errors="coerce")
            actual_csv_dtype = (
                "datetime64[ns]"
                if parsed_series.notna().any()
                else normalize_dtype(df[column_name].dtype)
            )
        else:
            actual_csv_dtype = normalize_dtype(df[column_name].dtype)

        if expected_dtype and actual_csv_dtype != expected_dtype:
            result["schema_issues"].append(
                f"Column {column_name} dtype mismatch: CSV={actual_csv_dtype} schema={expected_dtype}"
            )

        actual_nullable = df[column_name].isnull().any()
        expected_nullable = schema_nullables.get(column_name, False)
        if actual_nullable != expected_nullable:
            result["schema_issues"].append(
                f"Column {column_name} nullable mismatch: CSV={actual_nullable} schema={expected_nullable}"
            )

    result["schema_match"] = not result["schema_issues"]

    if not result["schema_match"]:
        logger.error("✗ Schema validation failed: %s", result["schema_issues"])
        result["status"] = "FAIL"

    if result["column_count_match"]:
        logger.info("✓ Column count matches")
    else:
        logger.error("✗ Column count mismatch")
        result["status"] = "FAIL"

    # ------------------------------------------------------------------
    # Column Names
    # ------------------------------------------------------------------

    missing = sorted(set(csv_columns) - set(db_columns))
    extra = sorted(set(db_columns) - set(csv_columns))

    result["column_names_match"] = not missing and not extra

    if result["column_names_match"]:
        logger.info("✓ Column names match")
    else:

        logger.error("Column names do not match")

        if missing:
            logger.error("Missing columns: %s", missing)

        if extra:
            logger.error("Unexpected columns: %s", extra)

        result["status"] = "FAIL"

    if result["status"] == "PASS":
        result["message"] = "Validation successful."

    return result


# ==============================================================================
# Main
# ==============================================================================

def main():

    logger.info("=" * 80)
    logger.info("Starting Validation Pipeline")
    logger.info("=" * 80)

    for table_name, filename in FILES.items():

        result = validate_table(table_name, filename)

        report["tables"].append(result)

        report["summary"]["tables_checked"] += 1

        if result["status"] == "PASS":
            report["summary"]["tables_passed"] += 1
        else:
            report["summary"]["tables_failed"] += 1
            report["overall_status"] = "FAIL"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = REPORT_DIR / f"validation_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    latest_report = REPORT_DIR / "validation_latest.json"

    with open(latest_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info("=" * 80)
    logger.info("Validation Complete")
    logger.info("Overall Status : %s", report["overall_status"])
    logger.info("Tables Checked : %s", report["summary"]["tables_checked"])
    logger.info("Tables Passed  : %s", report["summary"]["tables_passed"])
    logger.info("Tables Failed  : %s", report["summary"]["tables_failed"])
    logger.info("Report Saved   : %s", report_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()