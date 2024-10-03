import os
import pickle
import numpy as np
import time
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss
from csv_handling.csv_storage import move_inaccurate_categories, create_map_user_preferred_category
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

user_category_file = 'approach_using_LLM_FAISS/data/user_category.json'
csv_file = 'approach_using_LLM_FAISS/data/input/testing_data_approach_3_run1.csv'
output_csv_file = 'approach_using_LLM_FAISS/data/processed/testing_data_approach_3_run1_output.csv'
difference = 'approach_using_LLM_FAISS/data/recon/testing_data_approach_3_run1_mismatch.csv'
threshold = 1

# Initialize the encoder
encoder = SentenceTransformer("all-mpnet-base-v2")
user_category = create_map_user_preferred_category(user_category_file)

# Generates FAISS indices for user categories
def generate_faiss_indices(json_file_path):
    # Load user categories from JSON file
    with open(json_file_path, 'r') as file:
        user_category = json.load(file)

    # Iterate through each user in the JSON file
    for user_id, categories in user_category.items():
        # Initialize FAISS index for the user
        faiss_index = None
        # Encode each category and add to the FAISS index
        category_vectors = [encoder.encode([category])[0] for category in categories]

        # Convert list of vectors to numpy array
        category_vectors = np.array(category_vectors)

        # Create FAISS index with correct dimensions
        faiss_index = faiss.IndexFlatL2(category_vectors.shape[1])
        faiss_index.add(category_vectors)

        # Save the FAISS index to a pickle file
        pickle_file_path = f"approach_using_LLM_FAISS/faiss_pickle/faiss_store_{user_id}.pkl"
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump((faiss_index, categories), pickle_file)

        print(f"FAISS index for user {user_id} saved to {pickle_file_path}")

generate_faiss_indices(user_category_file)

# Set up LangChain's prompt template system
prompt_template = """
You are CategorizePro - Categorize the following transaction and provide only the best guess for what accounting category it belongs in:
Transaction: {transaction}
Provide only the best category name for this transaction without any explanation.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50, max_retries=1, verbose=True)

# Function to categorize a transaction
def generate_category(transaction_desc):
    chain = prompt | llm
    response = chain.invoke(
        {
            "output_language": "English",
            "transaction": transaction_desc,
        }
    )
    if hasattr(response, 'content'):
        content = response.content.strip()
    else:
        content = response.strip()
    
    # Assuming the response contains token usage information in the same format
    total_tokens = response.response_metadata['token_usage']['total_tokens']
    model_name = response.response_metadata['model_name']
    return content, total_tokens, model_name

# Search for a category vector in the user's FAISS index
def search_in_user_pickle(user_id, category_vector):
    # Load the user's FAISS index from the pickle file
    pickle_file_path = f"approach_using_LLM_FAISS/faiss_pickle/faiss_store_{user_id}.pkl"
    with open(pickle_file_path, 'rb') as pickle_file:
        faiss_index, categories = pickle.load(pickle_file)

    # Ensure the category_vector is in the correct shape
    category_vector = np.array(category_vector).reshape(1, -1)

    # Perform the search in the FAISS index
    distances, indices = faiss_index.search(category_vector, 1)
    return distances, indices

def get_matched_category(user_id, matched_index):
    # Load the user's categories from the pickle file
    pickle_file_path = f"approach_using_LLM_FAISS/faiss_pickle/faiss_store_{user_id}.pkl"
    with open(pickle_file_path, 'rb') as pickle_file:
        faiss_index, categories = pickle.load(pickle_file)

    # Retrieve the category corresponding to the matched_index
    matched_category = categories[matched_index]
    return matched_category

def update_all_personalised_categories(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    total_tokens_used = 0

    # Loop over each row
    for index, row in df.iterrows():
        user_id = row['UserID']
        description = row['Description']
        
        # Generate a category using the LLM
        llm_category, total_tokens, model_name = generate_category(description)
        total_tokens_used += total_tokens

        print(f"LLMCategory: {llm_category}")
        category_vec = encoder.encode([llm_category])[0]

        distances, I = search_in_user_pickle(user_id, category_vec)
        print(distances, I)

        if distances[0][0] < threshold:

            # I contains the index of the closest match in the FAISS index
            matched_index = I[0][0]  # Get the index of the best match
            # Fetch the corresponding PersonalizedCategory using the index
            personalised_category = get_matched_category(user_id, matched_index)
            print(f"PersonalisedCategory: {personalised_category}")
        else:
            personalised_category = llm_category
            print("PersonalisedCategory set as LLM category due to threshold breach ", personalised_category)

        # Add a new column to the DataFrame
        df.at[index, 'PersonalisedCategory'] = personalised_category

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_file, index=False)

    total_cost = (total_tokens_used / 1000) * .03
    print(f"Total cost: {total_cost}")

update_all_personalised_categories(csv_file)
move_inaccurate_categories(output_csv_file, difference, user_category)