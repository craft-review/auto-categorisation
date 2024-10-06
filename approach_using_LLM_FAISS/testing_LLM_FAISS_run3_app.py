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

user_category_cache = 'approach_using_LLM_FAISS/data/user_category.json'
input_trx_csv_file = 'approach_using_LLM_FAISS/data/input/testing_data_run3_distant_trx.csv'
output_trx_csv_file = 'approach_using_LLM_FAISS/data/processed/testing_data_run3_distant_trx_output.csv'
recon_csv_file = 'approach_using_LLM_FAISS/data/recon/testing_data_run3_distant_trx_recon.csv'
eucledian_dist_threshold = .7

# Define the cost per 1000 tokens for each model
model_cost_dict = {
    "gpt-4o-mini-2024-07-18": 0.000750,
    "gpt-4o-2024-08-06": 0.02000,
}

# initialize the model
model = 'gpt-4o-mini'

# Initialize the encoder
hugging_face_encoder = SentenceTransformer("all-mpnet-base-v2")
user_category_map = create_map_user_preferred_category(user_category_cache)

# Generates FAISS indices for user categories
def generate_faiss_indices(user_category_cache):
    # Load user categories from JSON file
    with open(user_category_cache, 'r') as file:
        user_preferred_categories = json.load(file)

    # Iterate through each user in the JSON file
    for user_id, preferred_categories in user_preferred_categories.items():
        # Initialize FAISS index for the user
        faiss_index = None
        # Encode each category and add to the FAISS index
        preferred_category_embedded_vec = [hugging_face_encoder.encode([preferred_category])[0] for preferred_category in preferred_categories]

        # Convert list of vectors to numpy array
        preferred_category_embedded_vec = np.array(preferred_category_embedded_vec)

        # Create FAISS index with correct dimensions
        faiss_index = faiss.IndexFlatL2(preferred_category_embedded_vec.shape[1])
        faiss_index.add(preferred_category_embedded_vec)

        # Save the FAISS index to a pickle file
        pickle_file_path = f"approach_using_LLM_FAISS/faiss_pickle/faiss_store_{user_id}.pkl"
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump((faiss_index, preferred_categories), pickle_file)

        print(f"User {user_id} preferred_categories saved to {pickle_file_path} in FAISS vector index")

# if pickle file not present, then only generate it
if not os.path.exists('approach_using_LLM_FAISS/faiss_pickle'):
    os.makedirs('approach_using_LLM_FAISS/faiss_pickle')
    generate_faiss_indices(user_category_cache)

# Set up LangChain's prompt template system
prompt_template = """
You are CategorizePro - Categorize the following transaction and provide only the best guess for what accounting category it belongs in:
Transaction: {transaction}
Provide only the best category name for this transaction without any explanation.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model=model, temperature=0, max_tokens=100, max_retries=1, verbose=True)

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
def search_in_user_pickle(user_id, category_svec):
    # Load the user's FAISS index from the pickle file
    pickle_file_path = f"approach_using_LLM_FAISS/faiss_pickle/faiss_store_{user_id}.pkl"
    with open(pickle_file_path, 'rb') as pickle_file:
        faiss_index, categories = pickle.load(pickle_file)

    # Ensure the category_vector is in the correct shape
    category_svec = np.array(category_svec).reshape(1, -1)

    # Perform the search in the FAISS index
    distances, indices = faiss_index.search(category_svec, 1)
    return distances, indices

def get_matched_category(user_id, matched_index):
    # Load the user's categories from the pickle file
    pickle_file_path = f"approach_using_LLM_FAISS/faiss_pickle/faiss_store_{user_id}.pkl"
    with open(pickle_file_path, 'rb') as pickle_file:
        faiss_index, categories = pickle.load(pickle_file)

    # Retrieve the category corresponding to the matched_index
    matched_category = categories[matched_index]
    return matched_category

def update_all_personalised_categories(input_trx_csv_file):
    # Read the CSV file
    df = pd.read_csv(input_trx_csv_file)
    total_tokens_used = 0

    # Loop over each row
    for index, row in df.iterrows():
        user_id = row['UserID']
        description = row['Description']
        
        # Generate a category using the LLM
        llm_category, total_tokens, model_name = generate_category(description)
        total_tokens_used += total_tokens

        print(f"Transaction: {description}, UserID: {user_id}")
        print(f"LLMCategory: {llm_category}")
        category_svec = hugging_face_encoder.encode([llm_category])[0]

        distances, I = search_in_user_pickle(user_id, category_svec)
        print(f"Eucledian Distance: {distances[0][0]:.6f}, Index: {I[0][0]}")

        # I contains the index of the closest match in the FAISS index
        matched_index = I[0][0]  # Get the index of the best match
        # Fetch the corresponding PersonalizedCategory using the index
        personalised_category = get_matched_category(user_id, matched_index)
        
        if distances[0][0] < eucledian_dist_threshold:
            print(f"PersonalisedCategory: {personalised_category}")
        else:
            print(f"Due to threshold breach- PersonalisedCategory: {personalised_category} --> LLM category: {llm_category}")
            personalised_category = llm_category

        # Add a new column to the DataFrame
        df.at[index, 'PersonalisedCategory'] = personalised_category
        print()

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_trx_csv_file, index=False)

    total_cost = (total_tokens_used / 1000) * model_cost_dict[model_name]
    print(f"Total cost: {total_cost:.6f}")

update_all_personalised_categories(input_trx_csv_file)
move_inaccurate_categories(output_trx_csv_file, recon_csv_file, user_category_map)