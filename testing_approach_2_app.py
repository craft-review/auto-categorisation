from csv_handling.csv_storage import load_user_personalized_categories, store_user_probable_category
from prompt import generate_category
import pandas as pd

# Load CSV data (or create if it doesn't exist)
csv_file = 'data/testing_data_approach_2_run1.csv'

def update_all_probable_categories(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Loop over each row
    for index, row in df.iterrows():
        user_id = row['UserID']
        description = row['Description']
        
        # Load user preferences from CSV
        user_personal_categories = load_user_personalized_categories(csv_file, user_id)
        print(user_personal_categories)
        
        # Generate a category using the LLM
        probable_category = generate_category(description, user_personal_categories)
        
        # Call the function to update the category
        store_user_probable_category(csv_file, user_id, description, probable_category)

update_all_probable_categories(csv_file)