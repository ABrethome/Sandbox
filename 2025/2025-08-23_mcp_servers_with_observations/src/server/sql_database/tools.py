"""Tools for interacting with a SQL database."""

from typing import Any, Dict, Optional, Sequence, Type, Union, Literal

from sqlalchemy.engine import Result

from pydantic import BaseModel, Field, model_validator, ConfigDict

from langchain_core._api.deprecation import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase, truncate_word
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_community.utilities import SQLDatabase
from typing import Optional, List, Iterable
from sqlalchemy.engine import URL, Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.expression import Executable
from sqlalchemy.types import NullType
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    select,
    text,
)
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

class Observation(BaseModel):
    """Describe an observation used to ground an answer."""
    id: str = None
    type: str = "Sql"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SQLDatabaseWithSchema(SQLDatabase):
    """Include schema description of SQL database.

    Source:
    https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/utilities/sql_database.py#L23
    """
    def __init__(
            self, 
            engine, 
            schema = None, 
            metadata = None, 
            ignore_tables = None, 
            include_tables = None, 
            sample_rows_in_table_info = 3, 
            indexes_in_table_info = False,
            custom_table_descriptions = None,
            custom_table_info = None, 
            view_support = False, 
            max_string_length = 300, 
            lazy_table_reflection = False
        ):

        super().__init__(engine, schema, metadata, ignore_tables, include_tables, sample_rows_in_table_info, indexes_in_table_info, custom_table_info, view_support, max_string_length, lazy_table_reflection)

        self._custom_table_descriptions = custom_table_descriptions
        if self._custom_table_descriptions:
            if not isinstance(self._custom_table_descriptions, dict):
                raise TypeError(
                    "custom_table_descriptions must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_descriptions).intersection(self._all_tables)
            self._custom_table_descriptions = dict(
                (table, self._custom_table_descriptions[table])
                for table in self._custom_table_descriptions
                if table in intersection
            )

        
    def get_usable_table_descriptions(self, get_tbl_comments: bool = False) -> str:
        """Get descriptions of tables available."""
        if self._include_tables:
            table_names = sorted(self._include_tables)
        else:
            table_names = sorted(self._all_tables - self._ignore_tables)

        descriptions = []
        for name in table_names:
            if get_tbl_comments and name in self._custom_table_descriptions:
                descriptions.append(f"{name} -- {self._custom_table_descriptions[name]}")
            else:
                descriptions.append(name)
        return "\n".join(descriptions)

    def get_usable_table_descriptions_no_throw(self, get_tbl_comments: bool = False) -> str:
        """Get information about specified tables."""
        try:
            return self.get_usable_table_descriptions(get_tbl_comments=get_tbl_comments)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"
        
    def get_table_info(
        self, table_names: Optional[List[str]] = None, get_col_comments: bool = False
    ) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
        to_reflect = set(all_table_names) - set(metadata_table_names)
        if to_reflect:
            self._metadata.reflect(
                views=self._view_support,
                bind=self._engine,
                only=list(to_reflect),
                schema=self._schema,
            )

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # Ignore JSON datatyped columns - SQLAlchemy v1.x compatibility
            try:
                # For SQLAlchemy v2.x
                for k, v in table.columns.items():
                    if type(v.type) is NullType:
                        table._columns.remove(v)
            except AttributeError:
                # For SQLAlchemy v1.x
                for k, v in dict(table.columns).items():
                    if type(v.type) is NullType:
                        table._columns.remove(v)

            # add create table command
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"

            # Add column comments as dictionary
            if get_col_comments:
                try:
                    column_comments_dict = {}
                    for column in table.columns:
                        if column.comment:
                            column_comments_dict[column.name] = column.comment

                    if column_comments_dict:
                        table_info += (
                            f"\n\n/*\nColumn Comments: {column_comments_dict}\n*/"
                        )
                except Exception:
                    raise ValueError(
                        "Column comments are available on PostgreSQL, MySQL, Oracle"
                    )

            has_extra_info = (
                self._indexes_in_table_info or self._sample_rows_in_table_info
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                table_info += f"\n{self._get_sample_rows(table)}\n"
            if has_extra_info:
                table_info += "*/"
            tables.append(table_info)
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str
    
    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None, get_col_comments: bool = False) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names, get_col_comments=get_col_comments)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run(
        self,
        command: Union[str, Executable],
        fetch: Literal["all", "one", "cursor"] = "all",
        include_columns: bool = False,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result[Any]]:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        result = self._execute(
            command, fetch, parameters=parameters, execution_options=execution_options
        )

        if fetch == "cursor":
            return result

        res = [
            {
                column: truncate_word(value, length=self._max_string_length)
                for column, value in r.items()
            }
            for r in result
        ]

        if not include_columns:
            res = [tuple(row.values()) for row in res]  # type: ignore[misc]

        if not res:
            return ""
        else:
            return res
        
    def run_no_throw(
        self,
        command: str,
        fetch: Literal["all", "one"] = "all",
        include_columns: bool = False,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result[Any]]:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(
                command,
                fetch,
                parameters=parameters,
                execution_options=execution_options,
                include_columns=include_columns,
            )
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"
    
class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabaseWithSchema = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QuerySQLDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


class QuerySQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    args_schema: Type[BaseModel] = _QuerySQLDatabaseToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        logger.info(f"{self.name} tool: {query}")
        result = self.db.run_no_throw(query, include_columns=True)
        #TODO: implement blocking of unwanted SQL operations such as DELETE, UPDATE, etc. for safety
        logger.debug(result)
        if isinstance(result, list) and all(isinstance(row, dict) for row in result):
            df = pd.DataFrame(result)
            result = df.to_markdown(index=False)
        return Observation(
            type="Sql",
            content=result,
            metadata = {
                "query": query,
                "result": result
            }
        )


class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        logger.info(f"{self.name} tool: {table_names}")
        result= self.db.get_table_info_no_throw(
            [t.strip() for t in table_names.split(",")],
            get_col_comments=True
        )
        return Observation(
            type="String",
            content=result,
            metadata = {
                "query": table_names,
                "result": result
            }
        )


class _ListSQLDatabaseToolInput(BaseModel):
    tool_input: str = Field("", description="An empty string")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting tables names."""

    name: str = "sql_db_list_tables"
    description: str = "Input is an empty string. Output is the list of table names with their associated descriptions."
    args_schema: Type[BaseModel] = _ListSQLDatabaseToolInput

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a list of table names."""
        logger.info(f"{self.name} tool.")
        result = self.db.get_usable_table_descriptions_no_throw(get_tbl_comments=True)
        return Observation(
            type="String",
            content=result,
            metadata = {}
        )

class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed SQL query to be checked before execution.")


class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    @model_validator(mode="before")
    @classmethod
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Any:
        if "llm_chain" not in values:
            values["llm_chain"] = PromptTemplate(
                template=QUERY_CHECKER, input_variables=["dialect", "query"]
            ) | values.get("llm")

        if isinstance(values["llm_chain"].steps[0], PromptTemplate):
            input_variables = values["llm_chain"].steps[0].input_variables
            if input_variables != ["dialect", "query"]:
                raise ValueError(
                    "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
                )

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        logger.info(f"{self.name} tool: {query}.")
        message = self.llm_chain.invoke(
            input={
                "query": query,
                "dialect": self.db.dialect
            },
            callbacks=run_manager.get_child() if run_manager else None,
        )
        return Observation(
            type="Sql",
            content=message.content,
            metadata = {
                "query": query,
                "result": message.content
            }
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        logger.info(f"{self.name} tool: {query}.")
        message = await self.llm_chain.ainvoke(
            input={
                "query": query,
                "dialect": self.db.dialect
            },
            callbacks=run_manager.get_child() if run_manager else None,
        )
        return Observation(
            type="Sql",
            content=message.content,
            metadata = {
                "query": query,
                "result": message.content
            }
        )
