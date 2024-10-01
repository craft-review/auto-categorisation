import pandas as pd
from collections import defaultdict
import json

def create_map_user_preferred_category(user_category_file):
    # Read the JSON file
    with open(user_category_file, 'r') as json_file:
        data = json.load(json_file)
    
    # Initialize the hashmap
    hashmap = defaultdict(set)
    
    # Iterate through the JSON data
    for user_id, categories in data.items():
        for category in categories:
            hashmap[user_id].add(category)
    
    # Convert defaultdict to a regular dictionary if needed
    hashmap = dict(hashmap)
    
    return hashmap

# Load CSV containing personalized categories
def load_user_personalized_categories(csv_file, user_id):
    df = pd.read_csv(csv_file)
    user_data = df[df['UserID'] == user_id]
    return user_data['PersonalizedCategory'].tolist() if not user_data.empty else None

# Append new feedback into the CSV
def append_user_probable_category(csv_file, user_id, transaction, category):
    df = pd.read_csv(csv_file)
    new_row = {'UserID': user_id, 'TransactionID': transaction, 'ProbableCategory': category}
    df = df._append(new_row, ignore_index=True)
    df.to_csv(csv_file, index=False)
    print("UserId", user_id, "TransactionID", transaction, "ProbableCategory", category)

def store_user_personalised_category(csv_file, user_id, transaction_desc, category, output_csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if the row with matching UserID and TransactionID exists
    mask = (df['UserID'] == user_id) & (df['Description'] == transaction_desc)

    
    if not df[mask].empty:
        # Update the ProbableCategory in the same row
        df.loc[mask, 'PersonalisedCategory'] = category
        df.to_csv(output_csv_file, index=False)
        print("PersonalisedCategory updated for UserID:", user_id, ", Description:", transaction_desc, ", PersonalisedCategory:", category)
    else:
        print("No matching record found for UserID:", user_id, "Description:", transaction_desc)

def move_non_matching_categories(output_csv_file, difference, user_category):
    # Load the CSV into a DataFrame
    df = pd.read_csv(output_csv_file)
    
    # Initialize an empty DataFrame for mismatched records
    mismatched_df = pd.DataFrame(columns=df.columns)
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        user_id = row['UserID']
        category = row['PersonalisedCategory']
        
        # Check if the user exists in user_category and the category doesn't match
        if user_id in user_category and category not in user_category[user_id]:
            # Append the row to the mismatched DataFrame
            mismatched_df = pd.concat([mismatched_df, pd.DataFrame([row])], ignore_index=True)
    
    # Save the mismatched records to a new CSV file
    mismatched_df.to_csv(difference, index=False)
    
    # Print the success message
    print(f"Mismatched records have been moved to {difference}")