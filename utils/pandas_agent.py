import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Load CSV
csv_path = "data/Installed Power Capacity.csv"
df = pd.read_csv(csv_path)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Create Pandas Agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)


def ask_pandas_agent(question):
    response = agent.invoke(question)
    return response["output"]