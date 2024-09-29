import pandas as pd
from transformers import pipeline

# Load the previously generated transactions without categories
transactions_df = pd.read_csv('data/quickbooks_transactions_no_categories.csv')

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Define a function to generate a category based on the description
def generate_category(description):
    prompt = f"Categorize the following transaction based on the transaction description: '{description}'. Just provide only the best category name for this transaction without any explanation."
    response = generator(prompt, max_length=50, num_return_sequences=1, temperature=1.0)
    return response[0]['generated_text'].strip()

# Apply the model to generate categories
transactions_df['Category'] = transactions_df['Description'].apply(generate_category)

# Save the updated DataFrame with categories
transactions_df.to_csv('data/quickbooks_transactions_with_categories.csv', index=False)

# Display the first few rows of the updated DataFrame
print(transactions_df.head())
