from csv_handling.csv_storage import load_user_personalized_categories, store_user_probable_category, move_non_matching_categories
from prompt import generate_category
import pandas as pd

# Load CSV data (or create if it doesn't exist)
csv_file = 'data/testing_data_approach_2_run1.csv'
output_file = 'data/testing_data_approach_2_run2.csv'

def update_all_probable_categories(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    total_tokens_used = 0
    
    # Loop over each row
    for index, row in df.iterrows():
        user_id = row['UserID']
        description = row['Description']
        
        # Load user preferences from CSV
        user_personal_categories = load_user_personalized_categories(csv_file, user_id)
        print(user_personal_categories)
        
        # Generate a category using the LLM
        probable_category, tokens, model_name = generate_category(description, user_personal_categories)
        total_tokens_used += tokens
        
        # Call the function to update the category
        store_user_probable_category(csv_file, user_id, description, probable_category)
    total_cost = (total_tokens_used / 1000) * .03
    print(total_cost)

update_all_probable_categories(csv_file)
move_non_matching_categories(csv_file, output_file)