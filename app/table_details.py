import pandas as pd
import streamlit as st
from operator import itemgetter
# Remove OpenAI imports
# from langchain.chains.openai_tools import create_extraction_chain_pydantic
# from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI # Import Gemini Chat model
import os # Import os to access environment variables

# Initialize Gemini LLM (using environment variable for API key)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"], temperature=0)


from typing import List

@st.cache_data
def get_table_details():
    # Read the CSV file into a DataFrame
    # IMPORTANT: Ensure 'database_table_descriptions.csv' is uploaded to /content/ or accessible
    # Dynamically resolve absolute path for Streamlit Cloud
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "database_table_descriptions.csv")
    table_description = pd.read_csv(CSV_PATH)
    table_docs = []

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""
    # Updated BaseModel to match the structure for with_structured_output
    name: List[str] = Field(description="List of Name of tables in SQL database.")

def get_tables(table_response: Table) -> List[str]:
    """
    Extracts the list of table names from a Table object returned by with_structured_output.
    """
    return table_response.name


# table_names = "\n".join(db.get_usable_table_names())
table_details = get_table_details()

# Use ChatPromptTemplate for better prompt formatting with chat models
from langchain_core.prompts import ChatPromptTemplate

table_details_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """Return the names of ALL the SQL tables that MIGHT be relevant to the user question.
                          The tables are:

                          {table_details}

                          Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""),
            ("human", "{question}")
        ]
    )

# Create the chain using with_structured_output
table_chain = {"question": itemgetter("question"), "table_details": itemgetter("table_details")} | table_details_prompt | llm.with_structured_output(Table) | get_tables
