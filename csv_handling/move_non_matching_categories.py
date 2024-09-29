import pandas as pd

def move_non_matching_categories(csv_file, output_file):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Filter rows where PersonalizedCategory does not match ProbableCategory
    non_matching_df = df[df['PersonalizedCategory'] != df['ProbableCategory']]
    
    # Save the filtered DataFrame to a new CSV
    non_matching_df.to_csv(output_file, index=False)
    
    print(f"Non-matching categories have been moved to {output_file}")

# Define the input and output file paths
csv_file = '/Users/akhiljain/Documents/workspace/Intuit/auto-categorisation/data/testing_data_approach_2_run1.csv'  # Your input CSV file
output_file = '/Users/akhiljain/Documents/workspace/Intuit/auto-categorisation/data/testing_data_approach_2_run2.csv'  # New CSV for non-matching rows

# Call the function to move non-matching categories
move_non_matching_categories(csv_file, output_file)
