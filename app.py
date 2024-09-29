import streamlit as st
from csv_handling.csv_storage import load_user_personalized_categories, store_user_probable_category
from prompt import generate_category
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

# Load CSV data (or create if it doesn't exist)
csv_file = 'data/testing_data_approach_1.csv'

# UI based Categorization
# User input for transaction and user ID
st.title("Auto-Categorization of Transactions")

user_id = st.text_input("Enter User ID:")
transaction_desc = st.text_input("Enter Transaction Description:")

if st.button("Generate Category"):
    # Load user preferences from CSV
    user_personal_categories = load_user_personalized_categories(csv_file, user_id)
    print(user_personal_categories)
    
    # Generate a category using the LLM
    category = generate_category(transaction_desc, user_personal_categories)
    
    st.write(f"Suggested Category: {category}")
    
    store_user_probable_category(csv_file, user_id, transaction_desc, category)

