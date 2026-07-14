import json
import os
from pydantic import BaseModel, Field, schema
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .tools import execute_sql
from pathlib import Path

from dotenv import load_dotenv
from shared.logging_config import setup_logger

load_dotenv()

class SQLGeneration(BaseModel):
    sql: str = Field(description="Generated SQL query")
    reasoning: str = Field(description="Reasoning behind the SQL generation")


class SummaryGeneration(BaseModel):
    summary: str = Field(description="Summary of the user question")
    reasoning: str = Field(description="Reasoning behind the summary")


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def retrieve_schema(state):
    logger = setup_logger("retrieve_schema")
    logger.info("retrieve_schema started")
    SCHEMA_PATH = Path(__file__).parent / "table_schema.json"
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    logger.info("retrieve_schema loaded schema from %s", SCHEMA_PATH)

    return {
        "schema": schema
    }


def generate_sql(state):
    logger = setup_logger("generate_sql")
    logger.info("generate_sql started")

    user_query = state.get("user_query", "Get all customers")
    schema = state.get("schema", {})

    SYSTEM_PROMPT = f"""
        You are an expert PostgreSQL SQL developer.

        Your task is to generate ONLY valid PostgreSQL SQL.

        Rules:
        - Use only the provided schema.
        - Never invent tables or columns.
        - Use proper joins.
        - Return a SELECT statement only.
        - Do not use UPDATE, DELETE, INSERT, DROP or ALTER.

        Table Schema:
        {schema}
        """
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)]
    
    sql_llm = llm.with_structured_output(SQLGeneration).with_config(run_name="SQL_Generation")    
    response = sql_llm.invoke(messages)    
    

    logger.info("generate_sql produced SQL: %s", response.sql)

    return {
        "generated_sql": response.sql
    }

def sql_corrector(state):
    logger = setup_logger("sql_corrector")
    logger.info("sql_corrector started")

    generated_sql = state.get("generated_sql", "")
    sql_error = state.get("sql_error", "")
    schema = state.get("schema", {})

    if not sql_error:
        logger.info("sql_corrector skipped because no SQL error was present")
        return {
            "executed_sql": generated_sql
        }
    
    SYSTEM_PROMPT = f"""
        You are an expert PostgreSQL SQL developer.

        Your task is to fix the provided SQL query and generate ONLY valid PostgreSQL SQL.

        Rules:
        - Use only the provided schema.
        - Never invent tables or columns.
        - Use proper joins.
        - Return a SELECT statement only.
        - Do not use UPDATE, DELETE, INSERT, DROP or ALTER.

        Table Schema:
        {schema}

        The following SQL query has an error:
        {generated_sql}

        The error is:   
        {sql_error} 

        """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]
    response = sql_llm.invoke(messages)

    logger.info("sql_corrector generated corrected SQL: %s", response.sql)

    return {
        "executed_sql": response.sql,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def execute_generated_sql(state):
    logger = setup_logger("execute_generated_sql")
    logger.info("execute_generated_sql started")

    sql_error = state.get("sql_error")
    
    if not sql_error:
        generated_sql = state.get("generated_sql", "")
        
    else:
        generated_sql = state.get("executed_sql", "")

    if not generated_sql.strip().upper().startswith("SELECT"):
        logger.warning("execute_generated_sql rejected non-SELECT SQL")
        return {
            "sql_result": None,
            "sql_error": "Generated SQL does not contain a SELECT statement."
        }
    
    # execution_result = execute_sql(generated_sql)
    execution_result = execute_sql.invoke({"sql": generated_sql})

    if execution_result["success"]:
        logger.info("execute_generated_sql completed successfully")
        return {
            "executed_sql": execution_result["final_sql"],
            "sql_result": execution_result["df"],
            "sql_error": None
        }
    else:
        logger.error("execute_generated_sql failed: %s", execution_result["error"])
        return {
            "executed_sql": None,
            "sql_result": None,
            "sql_error": execution_result["error"]
        }


def summarize(state):
    logger = setup_logger("summarize")
    logger.info("summarize started")

    user_query = state["user_query"]
    sql_error = state.get("sql_error")
    executed_sql = state.get("executed_sql")
    sql_result = state.get("sql_result")

    if sql_error is None:
        SYSTEM_PROMPT = f"""
        You are a smart result summariser.
        Answer user question based on the result from SQL executed.

        Structure your answer in following manner:

        Business Summary -- insights from the result of SQL executed
        ==============================================
        Calculation method

        ==============================================
        Follow up question (if applicable)


        For calculation method use:
        {executed_sql}

        Result:
        {sql_result}
        """

        messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)]

        summary_llm = llm.with_structured_output(SummaryGeneration).with_config(run_name="SQLSummary")    
        response = summary_llm.invoke(messages)

        logger.info("summarize generated response")
        return {"answer": response.summary}
        
    else:
        summary = "Sorry, please try again later"
        logger.warning("summarize returning fallback message due to SQL error")
        return {"answer": summary}


