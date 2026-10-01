from pathlib import Path

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI


def get_database() -> SQLDatabase:
    db_path = Path(__file__).resolve().parents[2] / "ecommerce.db"
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")

def get_sql_tools() -> list:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    toolkit = SQLDatabaseToolkit(db=get_database(), llm=llm)
    return toolkit.get_tools()

__all__ = ["get_sql_tools", "get_database"]