import pandas as pd

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

def store_user_probable_category(csv_file, user_id, transaction_desc, category):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if the row with matching UserID and TransactionID exists
    mask = (df['UserID'] == user_id) & (df['Description'] == transaction_desc)
    
    if not df[mask].empty:
        # Update the ProbableCategory in the same row
        df.loc[mask, 'ProbableCategory'] = category
        df.to_csv(csv_file, index=False)
        print("ProbableCategory updated for UserID:", user_id, ", Description:", transaction_desc, ", ProbableCategory", category)
    else:
        print("No matching record found for UserID:", user_id, "Description:", transaction_desc)


