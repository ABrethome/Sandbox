"""Test MCP server with this client. """

import os
import argparse
import asyncio
import logging
from InquirerPy import inquirer
from httpx import ConnectError

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# oauth for clients
# https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#oauth-authentication-for-clients

BUILT_IN_QUESTIONS = [
    "What is the country of my customer called Alice?",
    "List all my customers ID, name and age who are aged above 40. Those are key customers.", 
]

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)


async def call_mcp_server_async(
    user_input: str,
    debug: bool = False,
):
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting async workflow with user input: {user_input}")
    
    settings =  {
        "customers_data": {
            # make sure you start your server on port 3002
            "url": "http://localhost:3002/mcp",
            "transport": "streamable_http"
        }
    }

    client = MultiServerMCPClient(settings)
    try:
        tools = await client.get_tools()
    except ExceptionGroup as eg:
        # Handle specific exceptions within the ExceptionGroup
        for exc in eg.exceptions:
            if isinstance(exc, ConnectError):
                logger.error(f"Failed to connect to MCP server. Please ensure the MCP server is running on http://localhost:3000. Error: {exc}")
                return
        # Re-raise if no specific handling is done
        raise
    
    logger.info(f"Retrieved tools: {tools}")

    # Check if required environment variables are defined
    required_env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_BASIC_DEPLOYMENT_NAME",
        "AZURE_OPENAI_BASIC_MODEL_NAME",
    ]

    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error(f"The following required environment variables are missing: {', '.join(missing_vars)}. Please, load them before running this script using `source ./load_env.sh`.")
        return
    
    agent = create_agent(
        name="Agent",
        model=AzureChatOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.environ.get("AZURE_OPENAI_BASIC_DEPLOYMENT_NAME"),
            temperature=float(os.environ.get("AZURE_OPENAI_TEMPERATURE", 0.7)),
            model_name=os.environ.get("AZURE_OPENAI_BASIC_MODEL_NAME"),
            timeout=60000,
        ),
        tools=tools,
        system_prompt="""
        You are an expert in SQL language and database operations. Follow these steps strictly and in the exact order provided to answer the user's question:

        1. **Retrieve the list of available tables**: Use the 'list_customer_tables' tool to obtain the most up-to-date list of tables accessible to you.

        2. **Retrieve the schema of the tables**: Use the 'get_customer_table_schema' tool to get the detailed schema description of the tables retrieved in step 1.

        3. **Formulate and execute a SQL query**:
            - Create a SQL query based on the user's question and the schema information obtained in step 2.
            - In your query, prefer selecting the column on which you filter.
            - Use the 'verify_customer_table_query' tool to validate your SQL query for correctness before executing it.
            - Execute the validated SQL query using the 'query_customer_table_data' tool to retrieve the required data.

            Example query:
            SELECT * FROM CUSTOMERS WHERE Name = 'Alice';

        4. **Handle errors or incomplete results**:
            - If the query execution results in an error, repeat step 3 by revising your SQL query, validating it, and executing it again.
            - If the data returned does not fully or adequately answer the user's question, refine your SQL query, validate it, and execute it again until the user's question is fully addressed.

        5. Once you have enough data, you must answer the question of the user.

        Ensure that each step is completed successfully before proceeding to the next. Do not skip or reorder these steps under any circumstances.
        """
    )

    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
    }
    # RunnableConfig is a TypedDict here
    config = {
        "configurable": {
            "thread_id": "default",
        },
        "recursion_limit": 100, # recursion limit in ReAct agents
    }
    async for event_type, event_data in agent.astream(
        initial_state,
        config=config,
        stream_mode=["updates"],
        subgraphs=False,
    ):
        try:
            if event_type in ["updates"] and isinstance(event_data, dict):
                # state updates
                for agent_name in event_data.keys():
                    # added messages
                    agent_info = event_data.get(agent_name) or {}
                    if (messages := agent_info.get("messages")):
                        for message_chunk in messages:
                            if isinstance(message_chunk, ToolMessage):
                                # Tool Message - Return the result of the tool call
                                # sometimes tools have empty name made by LLM
                                if message_chunk.name:
                                    message_chunk.pretty_print()
                                continue
                            elif isinstance(message_chunk, AIMessage):
                                # AI Message - Raw message tokens
                                message_chunk.pretty_print()
                                continue
                            elif isinstance(message_chunk, HumanMessage):
                                # Human Message - Raw message tokens
                                message_chunk.pretty_print()
                                continue
                            print(f"Unrecognized chunk type {type(message_chunk)}\n{message_chunk.content}")         
                        continue
                    #print(f"UNCAPTURED STATE EVENT: {event_data}")
                continue
            print(f"Unknown event type {event_type}.")
        except Exception as e:
            logger.error(f"Error processing stream output: {e}")
            print(f"Error processing output: {str(e)}")

    logger.info("Async workflow completed successfully")


def ask(
    question,
    debug=False,
):
    """Run the MCP workflow on a given tool.

    Args:
        question: The user's query or request
        debug: If True, enables debug level logging
    """
    asyncio.run(
        call_mcp_server_async(
            user_input=question,
            debug=debug,
        )
    )


def main(
    debug=False,
):
    """Interactive mode with built-in questions.

    Args:
        debug: If True, enables debug level logging
    """
    questions = BUILT_IN_QUESTIONS
    ask_own_option = "[Ask my own question]"

    # Select a question
    initial_question = inquirer.select(
        message="What do you want to search?",
        choices=[ask_own_option] + questions,
    ).execute()

    if initial_question == ask_own_option:
        initial_question = inquirer.text(
            message="What do you want to search?",
        ).execute()

    # Pass all parameters to ask function
    ask(
        question=initial_question,
        debug=debug,
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument("query", nargs="*", help="The query to process")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with built-in questions",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.interactive:
        # Pass command line arguments to main function
        main(
            debug=args.debug
        )
    else:
        # Parse user input from command line arguments or user input
        if args.query:
            user_query = " ".join(args.query)
        else:
            user_query = input("Enter your query: ")

        # Run the agent workflow with the provided parameters
        ask(
            question=user_query,
            debug=args.debug,
        )
