import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType

# Function to load data from a CSV file
def load_data(data):
    return pd.read_csv(data)

# Function to stream text with a delay
def stream_text(text: str, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# Function to stream DataFrame rows with a delay
def stream_df(df, delay: float = 0.02):
    for i in range(len(df)):
        yield df.iloc[[i]]
        time.sleep(delay)

# Load CSV data
df = load_data("data/llm_clean_list.csv")
df_leaderboard = load_data("data/llm_clean_leaderboard.csv")

# About page
def about_page():
    st.title("About")
    st.write("This app provides insights into Large Language Models (LLMs) and allows you to ask questions directly to the models.")

# Ask LLM page
def ask_llm_page():
    load_dotenv()  # Load environment variables

    # Initialize LLM with API key from environment variables
    llm = OpenAI(
        temperature=0,
        verbose=True,
        openai_api_key=os.getenv("GHTvgvbjnkml,poijhgvfcdxcfgvbhnjmkjhgftdsfghujn"),
    )

    # Streamlit app interface
    st.title("LLM Agent Bot")

    # Display the DataFrame
    st.write("Data Preview:", df.head())

    # User input for question
    user_question = st.text_input("Ask a question about the data:")

    if user_question:
        # Create the agent with the DataFrame and LLM
        agent_executor = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        
        # Use the agent to get the answer from the DataFrame
        try:
            answer = agent_executor.invoke(user_question)
            if answer:
                st.success(f"Answer: {answer}")
            else:
                st.warning("Sorry, I couldn't find an answer to your question in the data.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# LLM List page
def llm_list_page():
    st.title("LLM List")
    st.write("Explore the list of Large Language Models.")
    quick_filters = st.multiselect("Filters", df.columns.tolist())
    if quick_filters:
        st.dataframe(df[quick_filters])
    else:
        st.dataframe(df)

# LLM Stats page
def llm_stats_page():
    st.title("LLM Stats")
    st.write("Analyze the statistics of different LLMs.")

    metric = st.selectbox("Metric", ["Downloads", "Likes", "Context Length"])

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("LLM Stats"):
            show_llm_stats_as_table()

    with col2:
        if st.button("LLM List"):
            show_llm_list_as_table()

    with col3:
        if st.button("LLM Leaderboard"):
            show_stats_as_table_for_leaderboard()

    # Select the top 10 models by metric
    top_models = df.nlargest(columns=metric, n=10)[["Model Name", "Maintainer", metric]]

    # Bar Chart
    bar_fig = px.bar(top_models, x="Model Name", y=metric, color="Maintainer", title=f"Top 10 Models by {metric} - Bar Chart")
    st.plotly_chart(bar_fig)

    # Pie Chart
    pie_fig = px.pie(top_models, names="Model Name", values=metric, color="Maintainer", title=f"Top 10 Models by {metric} - Pie Chart")
    st.plotly_chart(pie_fig)

    # Scatter Chart
    scatter_fig = px.scatter(top_models, x="Model Name", y=metric, color="Maintainer", title=f"Top 10 Models by {metric} - Scatter Plot")
    st.plotly_chart(scatter_fig)

# LLM Leaderboard page
def llm_leaderboard_page():
    st.title("LLM Leaderboard")
    st.write("View the leaderboard of top-performing LLMs.")

    quick_filters = st.multiselect(
        "Filters ",
        [
            "Model Name",
            "Maintainer",
            "License",
            "Context Length",
            "Mt Bench",
            "Humaneval",
            "Input Priceusd/1M Tokens",
        ],
        default=["Model Name", "Maintainer", "License", "Context Length", "Humaneval"],
    )

    if quick_filters:
        st.dataframe(df_leaderboard[quick_filters])
    else:
        st.dataframe(df_leaderboard)

    metric = st.selectbox("Metric", ["Context Length", "Humaneval", "MT Bench"])

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("LLM Leaderboard Stats "):
            show_stats_as_table_for_leaderboard()

    # Top N Models
    # Select the top 10 models by metric
    top_models = df_leaderboard.nlargest(columns=metric, n=10)[
        ["Model Name", "Maintainer", metric]
    ]

    # Bar Chart
    bar_fig = px.bar(top_models, x="Model Name", y=metric, color="Maintainer", title=f"Top 10 Models by {metric} - Bar Chart")
    st.plotly_chart(bar_fig)

    # Pie Chart
    pie_fig = px.pie(top_models, names="Model Name", values=metric, color="Maintainer", title=f"Top 10 Models by {metric} - Pie Chart")
    st.plotly_chart(pie_fig)

    # Scatter Chart
    scatter_fig = px.scatter(top_models, x="Model Name", y=metric, color="Maintainer", title=f"Top 10 Models by {metric} - Scatter Plot")
    st.plotly_chart(scatter_fig)

# Dialogs for showing tables
@st.dialog("LLM List - Stats")
def show_llm_stats_as_table():
    metric = st.selectbox("Metric ", ["Downloads", "Likes", "Context Length"])
    # Select the top 10 models by metric
    top_models = df.nlargest(columns=metric, n=10)[["Model Name", "Maintainer", metric]]
    st.dataframe(top_models)

@st.dialog("LLM List ")
def show_llm_list_as_table():
    st.dataframe(df)

@st.dialog("LLM Leaderboard - Stats")
def show_stats_as_table_for_leaderboard():
    metric = st.selectbox("Metric  ", ["Context Length", "Humaneval", "MT Bench"])
    # Select the top 10 models by metric
    top_models = df_leaderboard.nlargest(columns=metric, n=10)[
        ["Model Name", "Maintainer", metric]
    ]
    st.dataframe(top_models)

# Define Streamlit pages
about = st.Page(about_page, title="About", icon=":material/info:")
ask_llm = st.Page(ask_llm_page, title="Ask LLM", icon=":material/chat:")
llm_stats = st.Page(llm_stats_page, title="LLM Stats", icon=":material/list:")
llm_list = st.Page(llm_list_page, title="LLM List", icon=":material/analytics:")
llm_leaderboard = st.Page(
    llm_leaderboard_page, title="Leaderboard", icon=":material/favorite:"
)

# Navigation setup
pg = st.navigation(
    {"Home": [llm_list, ask_llm, llm_stats, llm_leaderboard], "About": [about]}
)

# Run the application
pg.run()
