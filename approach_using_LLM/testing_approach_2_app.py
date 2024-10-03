from csv_handling.csv_storage import move_inaccurate_categories, create_map_user_preferred_category
from prompt import generate_category
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

# Generate user category from the CSV file
user_category_cache = 'approach_using_LLM/data/user_category.json'
input_trx_csv_file = 'approach_using_LLM/data/input/testing_data_approach_2_run1.csv'
output_trx_csv_file = 'approach_using_LLM/data/processed/testing_data_approach_2_run1_output.csv'
recon_csv_file = 'approach_using_LLM/data/recon/testing_data_approach_2_run1_mismatch.csv'

# Define the cost per 1000 tokens for each model
model_cost_dict = {
    "gpt-4o-mini-2024-07-18": 0.000750,
    "gpt-4o-2024-08-06": 0.02000,
}

user_category_map = create_map_user_preferred_category(user_category_cache)

def update_all_personalised_categories(input_trx_csv_file):
    # Read the CSV file
    df = pd.read_csv(input_trx_csv_file)
    total_tokens_used = 0

    # Loop over each row
    for index, row in df.iterrows():
        user_id = row['UserID']
        description = row['Description']
        
        # Generate a category using the LLM
        personalised_category, tokens, model_name = generate_category(description, user_category_map.get(user_id), model="gpt-4o-mini")
        print(personalised_category)
        total_tokens_used += tokens

        # Add a new column to the DataFrame
        df.at[index, 'PersonalisedCategory'] = personalised_category
        
        # Call the function to update the category
        #store_user_personalised_category(csv_file, user_id, description, personalised_category, output_csv_file)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_trx_csv_file, index=False)
    total_cost = (total_tokens_used / 1000) * model_cost_dict[model_name]
    print(total_cost)

update_all_personalised_categories(input_trx_csv_file)
move_inaccurate_categories(output_trx_csv_file, recon_csv_file, user_category_map)