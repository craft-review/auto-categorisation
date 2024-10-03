import json
import pandas as pd

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Save updated JSON
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to handle 1-to-1 mapping
def handle_1to1_mapping(json_data, csv_file):
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        user_id = row['UserID']
        personalised_category = row['PersonalisedCategory']
        feedback_category = row['FeedbackCategory']
        
        # Update category in the user preferences
        if user_id in json_data and personalised_category in json_data[user_id]:
            json_data[user_id].remove(personalised_category)
            json_data[user_id].append(feedback_category)
            print(f"Updated {personalised_category} to {feedback_category} for user {user_id} (1-to-1 mapping)")

# Function to handle n-to-1 mapping
def handle_nto1_mapping(json_data, csv_file):
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        user_id = row['UserID']
        personalised_category = row['PersonalisedCategory']
        feedback_category = row['FeedbackCategory']

        # Update category in the user preferences
        if user_id in json_data and personalised_category in json_data[user_id]:
            json_data[user_id] = [feedback_category if cat == personalised_category else cat for cat in json_data[user_id]]
            print(f"Updated {personalised_category} to {feedback_category} for user {user_id} (n-to-1 mapping)")

def handle_1ton_mapping(json_data, csv_file):
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        user_id = row['UserID']
        personalised_category = row['PersonalisedCategory']
        feedback_category = row['FeedbackCategory']
        
        # Remove the personalised category if it exists and add the feedback categories as subcategories
        if user_id in json_data:
            # Remove the personalised category
            if personalised_category in json_data[user_id]:
                json_data[user_id].remove(personalised_category)
                
            # If feedback_category is not present, add it
            if feedback_category not in json_data[user_id]:
                json_data[user_id].append(feedback_category)
                
            print(f"Removed {personalised_category} and added {feedback_category} for user {user_id} (1-to-n mapping)")

    # Handle the case where the FeedbackCategory maps multiple times for the same PersonalisedCategory
    feedback_groups = df.groupby(['UserID', 'PersonalisedCategory'])['FeedbackCategory'].apply(set).reset_index()

    # Iterate over the feedback_groups to check for multiple mappings
    for _, row in feedback_groups.iterrows():
        user_id = row['UserID']
        personalised_category = row['PersonalisedCategory']
        feedback_categories = row['FeedbackCategory']
        
        # Ensure sub-categories for 1-to-n mappings
        for feedback_category in feedback_categories:
            if feedback_category not in json_data[user_id]:
                json_data[user_id].append(feedback_category)
                print(f"Added {feedback_category} as a sub-category for user {user_id}")

# Main function to orchestrate the updates
def update_user_categories(json_file, csv_1to1, csv_nto1, csv_1ton):
    # Load the JSON data
    user_categories = load_json(json_file)

    # Process each CSV file for the respective mappings
    handle_1to1_mapping(user_categories, csv_1to1)
    handle_nto1_mapping(user_categories, csv_nto1)
    handle_1ton_mapping(user_categories, csv_1ton)

    # Save the updated JSON data back to the file
    save_json(user_categories, json_file)
    print("User category preferences have been updated.")

# Define file paths
json_file = 'approach_using_LLM/data/user_category.json'
csv_1to1 = 'approach_using_LLM/data/input/feedback-1to1.csv'
csv_nto1 = 'approach_using_LLM/data/input/feedback-nto1.csv'
csv_1ton = 'approach_using_LLM/data/input/feedback-1ton.csv'

# Run the update
update_user_categories(json_file, csv_1to1, csv_nto1, csv_1ton)