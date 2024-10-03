import streamlit as st
from prompt import generate_category
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()  # take environment variables from .env

# UI based Categorization
# User input for transaction and user ID
st.title("Explainability of Transaction categorization using LLM")
transaction_desc = st.text_input("Enter Transaction Description:")

prompt_template = """
        You are CategorizePro - I will give you details about
        a Transaction: {transaction} and you will give me your best guess
        for what accounting category it belongs in. 
        Give a brief description Of why you choose that category
        in 50 words or less. Please add a line break for every sentence.
        """
 # Set up LangChain's prompt template system
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, max_tokens=100, max_retries=2, verbose=True)

if st.button("Explain transaction categorization"):
    # Generate and explain category using the LLM
    content, total_tokens, model_name = generate_category(transaction_desc,prompt=prompt)
    st.write(f"Explaination of model's categorization: {content}")