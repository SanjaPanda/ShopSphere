from langchain_core.tools import tool

from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
import pandas as pd
from shared.logging_config import setup_logger

# ==============================================================================
# Logger
# ==============================================================================

logger = setup_logger(__name__)


load_dotenv()

DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

@tool
def execute_sql(sql: str):
    """
    Executes the provided SQL query against the PostgreSQL database.
    Input:
        sql (str): The SQL query to be executed.
    Output:
        dict: A dictionary containing the execution result, including:
            - success (bool): Indicates if the execution was successful.
            - df (list): The result of the query as a list of dictionaries (if successful).
            - final_sql (str): The executed SQL query (if successful).
            - error (str): The error message (if execution failed).
    
    """
    try:
        engine = create_engine(DATABASE_URL)

        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        return {
            "success": True,
            "df": df.to_dict(orient="records"),
            "final_sql": sql,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "df": None,
            "final_sql": None,
            "error": str(e),
        }


