import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain.utilities import PandasAgent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import PandasToolkit
from langchain.agents import create_pandas_agent

# Load environment variables
load_dotenv()

# Initialize LLM with API key from environment variables
llm = OpenAI(
    temperature=0,
    verbose=True,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Load CSV file into Pandas DataFrame
csv_file_path = 'your_csv_file.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Setup the Pandas agent toolkit
toolkit = PandasToolkit(df=df, llm=llm)

# Create the Pandas agent
agent_executor = create_pandas_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Streamlit app interface
st.title("LLM Agent Bot")

# Display the DataFrame
st.write("Data Preview:", df.head())

user_question = st.text_input("Ask a question about the data:")

if user_question:
    # Use the agent to get the answer from the DataFrame
    answer = agent_executor.invoke(user_question)
    
    if answer:
        st.success(f"Answer: {answer}")
    else:
        st.warning("Sorry, I couldn't find an answer to your question in the data.")
