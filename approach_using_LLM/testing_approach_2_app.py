from csv_handling.csv_storage import load_user_personalized_categories, store_user_personalised_category, move_non_matching_categories, create_map_user_preferred_category
from prompt import generate_category
import pandas as pd
import json

# Generate user category from the CSV file
user_category_file = 'approach_using_LLM/data/user_category.json'
csv_file = 'approach_using_LLM/data/input/testing_data_approach_2_run1.csv'
output_csv_file = 'approach_using_LLM/data/processed/testing_data_approach_2_run1_output.csv'
difference = 'approach_using_LLM/data/recon/testing_data_approach_2_run1_mismatch.csv'

user_category = create_map_user_preferred_category(user_category_file)

def update_all_personalised_categories(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    total_tokens_used = 0

    # Loop over each row
    for index, row in df.iterrows():
        user_id = row['UserID']
        description = row['Description']
        
        # Generate a category using the LLM
        personalised_category, tokens, model_name = generate_category(description, user_category.get(user_id))
        print(personalised_category)
        total_tokens_used += tokens

        # Add a new column to the DataFrame
        df.at[index, 'PersonalisedCategory'] = personalised_category
        
        # Call the function to update the category
        #store_user_personalised_category(csv_file, user_id, description, personalised_category, output_csv_file)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_file, index=False)
    total_cost = (total_tokens_used / 1000) * .03
    print(total_cost)

update_all_personalised_categories(csv_file)
move_non_matching_categories(output_csv_file, difference, user_category)