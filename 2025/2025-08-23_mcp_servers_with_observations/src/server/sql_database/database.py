import os
from langchain_openai import AzureChatOpenAI

from src.server.sql_database.tools import (
    SQLDatabaseWithSchema,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)

from src.server.constants import DATABASE_PATH
from src.server.sql_database.schema_descriptions import (
    CUSTOMERS_TABLE_DESC,
    CUSTOMERS_TABLE_INFO_DESC,
)

import argparse
from InquirerPy import inquirer
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)



db = SQLDatabaseWithSchema.from_uri(
    f"duckdb:///{DATABASE_PATH}",
    include_tables=["CUSTOMERS"], # exclude other system tables such as COMMENTS
    sample_rows_in_table_info=3, # give 3 examples of each table
    custom_table_descriptions={
        "CUSTOMERS": CUSTOMERS_TABLE_DESC, 
        # add more tables here if needed
    },
    custom_table_info={
        "CUSTOMERS": CUSTOMERS_TABLE_INFO_DESC, 
        # add more tables here if needed
    }
)

llm = AzureChatOpenAI(
    azure_endpoint= os.environ.get("AZURE_OPENAI_ENDPOINT"),
    openai_api_key= os.environ.get("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_BASIC_DEPLOYMENT_NAME"),
    temperature=os.environ.get("AZURE_OPENAI_TEMPERATURE"),
    model_name=os.environ.get("AZURE_OPENAI_BASIC_MODEL_NAME"),
    timeout= 60000,
)

# source: 
# https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/tools/sql_database/tool.py

list_sql_database_tool = ListSQLDatabaseTool(
    db=db,
    name="list_customer_tables",
    description="Input is an empty string. Output is the list of table names with their associated descriptions.",
)

info_sql_database_tool_description = (
    "Input to this tool is a comma-separated list of tables, output is the "
    "schema and sample rows for those tables. "
    "Be sure that the tables actually exist by calling "
    f"'{list_sql_database_tool.name}' tool first! "
    "Example Input: table1, table2, table3"
)
info_sql_database_tool = InfoSQLDatabaseTool(
    db=db, 
    name = "get_customer_table_schema",
    description=info_sql_database_tool_description
)

query_sql_database_tool_description = (
    "Input to this tool is a detailed and correct SQL query, output is a "
    "result from the database in markdown format. If the query is not correct, an error message "
    "will be returned. If an error is returned, rewrite the query, check the "
    "query, and try again. If you encounter an issue with Unknown column "
    f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
    "to query the correct table fields."
)
query_sql_database_tool = QuerySQLDatabaseTool(
    db=db, 
    name = "query_customer_table_data",
    description=query_sql_database_tool_description
)

query_sql_checker_tool_description = (
    "Use this tool to double check if your query is correct before executing "
    "it. Always use this tool before executing a query with "
    f"{query_sql_database_tool.name}!"
)
query_sql_checker_tool = QuerySQLCheckerTool(
    db=db,
    name = "verify_customer_table_query",
    llm=llm, description=query_sql_checker_tool_description
)



async def execute_tool(tool_name: str, args: dict):
    """Execute the selected tool with the provided arguments."""
    if tool_name == "query_customer_table_data":
        result = query_sql_database_tool._run(query=args.get("query"))
    elif tool_name == "get_customer_table_schema":
        result = info_sql_database_tool._run(table_names=args.get("table_names"))
    elif tool_name == "list_customer_tables":
        result = list_sql_database_tool._run(tool_input="")
    elif tool_name == "verify_customer_table_query":
        result = await query_sql_checker_tool._arun(query=args.get("query"))
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")

    print(f"Result from {tool_name}: {result}")


def main():
    """Interactive mode to select and execute a tool."""
    tools = [
        "query_customer_table_data",
        "get_customer_table_schema",
        "list_customer_tables",
        "verify_customer_table_query",
    ]

    selected_tool = inquirer.select(
        message="Select a tool to execute:",
        choices=tools,
    ).execute()

    args = {}
    if selected_tool == "query_customer_table_data":
        args["query"] = inquirer.text(message="Enter the SQL query:").execute()
        # SELECT * FROM CUSTOMERS WHERE Name = 'Alice';
    elif selected_tool == "get_customer_table_schema":
        args["table_names"] = inquirer.text(
            message="Enter a comma-separated list of table names:"
        ).execute()
        # CUSTOMERS
    elif selected_tool == "verify_customer_table_query":
        args["query"] = inquirer.text(message="Enter the SQL query to check:").execute()
    asyncio.run(execute_tool(selected_tool, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQL database tools")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.interactive:
        main()
    else:
        print("Please use --interactive to run the script interactively.")
