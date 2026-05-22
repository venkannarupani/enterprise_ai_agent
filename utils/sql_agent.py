from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Database connection
engine = create_engine("sqlite:///Installed_Power_database.db")

# LangChain SQL DB
sql_db = SQLDatabase(engine)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    db=sql_db,
    verbose=True,
    agent_type="openai-tools"
)


def ask_sql_agent(question):
    response = agent_executor.invoke({"input": question})
    return response["output"]