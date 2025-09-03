import streamlit as st
# from dotenv import load_dotenv
import os # Make sure os is imported

# Load variables from .env file
# load_dotenv()

# --- DEBUGGING STEP ---
# Check if the API key was loaded successfully
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🔴 ERROR: GEMINI_API_KEY not found!")
    st.info("Please make sure you have a .env file in the same directory as main.py with the line: GEMINI_API_KEY='your_api_key'")
    st.stop() # Stop the app if the key is missing
# --- END DEBUGGING STEP ---

# If the script continues, it means the key was found.
# We only import the rest of our app if the key exists.
from langchain_utils import invoke_chain

st.title("Langchain NL2SQL Chatbot") # Update title to reflect Gemini usage

# Remove OpenAI API key setting and client initialization
# client = OpenAI(api_key="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")

# Remove setting a default OpenAI model
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            # Directly use invoke_chain which is now Gemini-configured
            response = invoke_chain(prompt,st.session_state.messages)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
