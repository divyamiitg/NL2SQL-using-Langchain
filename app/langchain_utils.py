# import os
# from dotenv import load_dotenv

# load_dotenv()

# db_user = os.getenv("db_user")
# db_password = os.getenv("db_password")
# db_host = os.getenv("db_host")
# db_name = os.getenv("db_name")

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain.chains import create_sql_query_chain
# # Replace ChatOpenAI with ChatGoogleGenerativeAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from langchain.memory import ChatMessageHistory

# from operator import itemgetter

# from langchain_core.output_parsers import StrOutputParser

# from langchain_core.runnables import RunnablePassthrough
# # Remove the second import of ChatOpenAI
# # from langchain_openai import ChatOpenAI

# # Assuming these modules are also consistent with Gemini or are generic LangChain components
# from table_details import table_chain as select_table
# from prompts import final_prompt, answer_prompt

# import streamlit as st

# @st.cache_resource
# def get_chain():
#     print("Creating chain")
#     # Add SSL parameters if needed for your Aiven database
#     db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

#     # Replace ChatOpenAI initialization with ChatGoogleGenerativeAI
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0)

#     generate_query = create_sql_query_chain(llm, db, final_prompt)
#     execute_query = QuerySQLDataBaseTool(db=db)
#     rephrase_answer = answer_prompt | llm | StrOutputParser()

#     chain = (
#         RunnablePassthrough.assign(table_names_to_use=select_table) |
#         RunnablePassthrough.assign(query=generate_query).assign(
#             result=itemgetter("query") | execute_query
#         )
#         | rephrase_answer
#     )

#     return chain

# def create_history(messages):
#     history = ChatMessageHistory()
#     for message in messages:
#         if message["role"] == "user":
#             history.add_user_message(message["content"])
#         else:
#             history.add_ai_message(message["content"])
#     return history

# def invoke_chain(question, messages):
#    chain = get_chain()
#    history = create_history(messages)
#    # Ensure "messages" key is passed if final_prompt expects it
#    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
#    history.add_user_message(question)
#    history.add_ai_message(response)
#    return response

# langchain_utils.py (Corrected)

import os
from dotenv import load_dotenv
from operator import itemgetter

import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the necessary components from your other local files
# --- CHANGE: Added get_table_details to the import ---
from table_details import get_table_details, table_chain as select_table
from examples import get_example_selector
from prompts import answer_prompt, example_prompt

# Load environment variables from .env file
load_dotenv()

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_name = os.getenv("db_name")
db_port = os.getenv("db_port")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@st.cache_resource
def get_chain():
    """Creates and caches the main LangChain runnable."""
    print("Creating chain...")
    db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    db = SQLDatabase.from_uri(db_uri)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0
    )

    example_selector = get_example_selector()

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input", "top_k"],
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specified.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
            ),
            few_shot_prompt,
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ]
    )
    
    # --- CHANGE: Load the table descriptions from the CSV file ---
    table_details = get_table_details()

    generate_query = create_sql_query_chain(llm, db, final_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    rephrase_answer = answer_prompt | llm | StrOutputParser()

    # --- CHANGE: Inject the loaded table_details into the chain's input ---
    chain = (
        RunnablePassthrough.assign(table_details=lambda x: table_details)
        | RunnablePassthrough.assign(table_names_to_use=select_table)
        | RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
    )
    
    print("Chain created successfully!")
    return chain


def create_history(messages):
    """Creates a chat history object from a list of messages."""
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


def invoke_chain(question, messages):
    """Invokes the main chain with the user's question and chat history."""
    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke(
        {"question": question, "top_k": 3, "messages": history.messages}
    )
    history.add_user_message(question)
    history.add_ai_message(response)
    return response