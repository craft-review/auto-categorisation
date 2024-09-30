import os
import pickle
import numpy as np
import time
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss
from csv_handling.csv_storage import store_user_probable_category, move_non_matching_categories

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)
file_path = "faiss_store_categories_2.pkl"
csv_file = "data/testing_data_approach_3_run1.csv"
output_file = "data/testing_data_approach_3_run2.csv"
user_map = {'U001': 'Akhil', 'U002': 'Rahul', 'U003': 'Sachin', 'U004': 'Rohit', 'U005': 'Virat'}

# load data
# Function to extract fields from page_content
def extract_values_from_page_content(page_content, field_name):
    lines = page_content.split("\n")
    for line in lines:
        if line.startswith(f"{field_name}:"):
            return line.split(": ")[1].strip()
    return None

loader = CSVLoader(csv_file, source_column="UserID")
data = loader.load()
# Initialize the encoder and FAISS index
encoder = SentenceTransformer("all-mpnet-base-v2")
faiss_index = None
# Store the PersonalizedCategory for each vector in a list
personalized_category_list = []

# Extract 'source' and 'PersonalizedCategory' for each document
for document in data:
    source = document.metadata['source']  # This corresponds to the "UserID"
    page_content = document.page_content  # Multi-line content for each row
    
    # Extract the PersonalizedCategory from the page_content
    personalized_category = extract_values_from_page_content(page_content, "PersonalizedCategory")
    
    if personalized_category:
        # Initialize FAISS index if not already initialized
        category_vec = encoder.encode([personalized_category])[0]
        user_id_vec = encoder.encode([user_map.get(source)])[0]
        vectors = np.concatenate((category_vec, user_id_vec)).reshape(1, -1)

        if faiss_index is None:
            faiss_index = faiss.IndexFlatL2(vectors.shape[1])  # Create FAISS index with correct dimensions

        # Add vectors to the FAISS index
        faiss_index.add(vectors)

        # Store the corresponding PersonalizedCategory in the list
        personalized_category_list.append(personalized_category)

        # Log or display progress
        print(f"Source (UserID): {source}, PersonalizedCategory: {personalized_category}")
    else:
        print(f"Source (UserID): {source} has no valid PersonalizedCategory, skipping...")

time.sleep(2)

# Save the FAISS index to a pickle file
with open(file_path, "wb") as f:
    pickle.dump(faiss_index, f)
    # pickle.dump(vectorstore_openai, f)

# Set up LangChain's prompt template system
prompt_template = """
You are CategorizePro - Categorize the following transaction and provide only the best guess for what accounting category it belongs in:
Transaction: {transaction}
Provide only the best category name for this transaction without any explanation.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50, max_retries=1, verbose=True)

# Example function to categorize a transaction
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

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
        total_tokens_used = 0

        for document in data:
            source = document.metadata['source']  # This corresponds to the "UserID"
            page_content = document.page_content  # Multi-line content for each row
            
            # Extract the PersonalizedCategory from the page_content
            personalized_category = extract_values_from_page_content(page_content, "PersonalizedCategory")
            user_id = extract_values_from_page_content(page_content, "UserID")
            transaction_desc = extract_values_from_page_content(page_content, "Description")
            # print(user_id)
            # Generate a category using the LLM
            llm_category, total_tokens, model_name = generate_category(transaction_desc)

            total_tokens_used += total_tokens
            
            print(f"LLMCategory: {llm_category}")
            category_vec = encoder.encode([llm_category])[0]
            user_id_vec = encoder.encode([user_map.get(user_id)])[0]
            svec = np.concatenate((category_vec, user_id_vec)).reshape(1, -1)
            
            distances, I = faiss_index.search(svec, 1) 
            # print(distances, I)

            # I contains the index of the closest match in the FAISS index
            matched_index = I[0][0]  # Get the index of the best match

            # Fetch the corresponding PersonalizedCategory using the index
            if matched_index != -1:  # Ensure a match was found
                matched_category = personalized_category_list[matched_index]
                print(f"PersonalizedCategory: {matched_category}")
            else:
                print("No match found.")

            store_user_probable_category(csv_file, user_id, transaction_desc, matched_category)
        
        total_cost = (total_tokens_used / 1000) * .03
        print(total_cost)
        move_non_matching_categories(csv_file, output_file)
