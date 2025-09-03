import sys

try:
    # Force Chroma to use pysqlite3 instead of the default sqlite3
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass


import os
# from dotenv import load_dotenv
import streamlit as st
import asyncio # <--- 1. Import asyncio

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load_dotenv()

examples = [
    {
        "input": "List all customers in France with a credit limit over 20,000.",
        "query": "SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;"
    },
    {
        "input": "Get the highest payment amount made by any customer.",
        "query": "SELECT MAX(amount) FROM payments;"
    },
    {
        "input": "Show product details for products in the 'Motorcycles' product line.",
        "query": "SELECT * FROM products WHERE productLine = 'Motorcycles';"
    },
    {
        "input": "Retrieve the names of employees who report to employee number 1002.",
        "query": "SELECT firstName, lastName FROM employees WHERE reportsTo = 1002;"
    },
    {
        "input": "List all products with a stock quantity less than 7000.",
        "query": "SELECT productName, quantityInStock FROM products WHERE quantityInStock < 7000;"
    },
    {
     'input':"what is price of `1968 Ford Mustang`",
     "query": "SELECT `buyPrice`, `MSRP` FROM products  WHERE `productName` = '1968 Ford Mustang' LIMIT 1;"
    }
]

@st.cache_resource
def get_example_selector():
    # --- FIX: Create and set an event loop for this thread ---
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    # --- END FIX ---

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GEMINI_API_KEY"]),
        Chroma,
        k=2,
        input_keys=["input"],
    )
    return example_selector

