from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

def build_db_tools(llm, postgres_url):
    db = SQLDatabase.from_uri(postgres_url)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()
