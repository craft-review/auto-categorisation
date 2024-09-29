import pandas as pd
import random

# Load the previously generated transactions
transactions_df = pd.read_csv("data/quickbooks_transactions.csv")

# Synonym categories for 3 different users
synonymous_categories = {
    "Office Supplies": ["Office Essentials", "Stationery", "Office Items"],
    "Travel": ["Business Travel", "Trips", "Work Travel"],
    "Utilities": ["Bills", "Utility Payments", "Utilities Expenses"],
    "Insurance": ["Insurance Costs", "Insurance Premiums", "Insurance Payments"],
    "Meals and Entertainment": ["Meals", "Dining", "Food & Beverages"],
    "Rent": ["Office Rent", "Building Rent", "Lease Payment"],
    "Marketing": ["Advertising", "Campaigns", "Promotions"],
    "Consulting Services": ["Consulting", "Advisory Services", "Consulting Fees"],
    "Fuel": ["Gasoline", "Fuel Costs", "Petrol"],
    "Shipping": ["Shipping Costs", "Delivery", "Logistics"],
    "Advertising": ["Ads", "Advertisements", "Marketing Campaign"],
    "Repairs and Maintenance": ["Maintenance", "Fixes", "Repairs"],
    "Professional Services": ["Expert Services", "Consultancy", "Specialized Services"],
    "Software": ["Software Subscriptions", "Apps", "Tools"],
    "Taxes": ["Tax Payments", "Government Fees", "Tax Costs"],
    "Training": ["Workshops", "Employee Training", "Skills Development"],
    "Subscription": ["Subscriptions", "Recurring Fees", "Memberships"],
    "Legal Services": ["Legal Fees", "Law Services", "Legal Counsel"],
    "Telecommunications": ["Phone Bills", "Internet Services", "Telecom"],
    "Payroll": ["Employee Salaries", "Salaries", "Wages"],
    "Employee Benefits": ["Benefits", "Employee Perks", "Staff Benefits"],
    "Miscellaneous": ["Other Expenses", "Misc", "Sundry"],
    "Credit Card Payment": ["Card Payment", "Credit Payment", "Card Balance"],
    "Loan Payment": ["Loan Repayment", "Loan Installment", "Debt Repayment"],
    "Bank Charges": ["Bank Fees", "Service Fees", "Bank Service Charges"]
}

# Function to generate a personalized CSV for each user with synonymous categories
# def generate_personalized_csv(user_num, synonymous_categories):
#     user_data = transactions_df.copy()
    
#     # Map the categories to personalized categories for each user
#     user_data['PersonalizedCategory'] = user_data['Category'].apply(
#         lambda x: synonymous_categories.get(x, [x])[user_num - 1]
#     )
    
#     # Save to CSV
#     csv_file = f"data/quickbooks_user_{user_num}_transactions.csv"
#     user_data.to_csv(csv_file, index=False)
#     return csv_file

# Generate CSVs for three users
# user_1_csv = generate_personalized_csv(1, synonymous_categories)
# user_2_csv = generate_personalized_csv(2, synonymous_categories)
# user_3_csv = generate_personalized_csv(3, synonymous_categories)

# user_1_csv, user_2_csv, user_3_csv





# Load the previously generated transaction data
transactions_df = pd.read_csv('data/quickbooks_transactions.csv')

# Define 5 users
users = ["User1", "User2", "User3", "User4", "User5"]

# Create a mapping of synonymous categories for each user
synonymous_categories_v2 = {
    "Office Supplies": ["Stationery", "Workplace Essentials", "Office Needs", "Supplies", "Workplace Goods"],
    "Travel": ["Business Travel", "Trip Expenses", "Traveling", "Journeys", "Travel Costs"],
    "Utilities": ["Energy Bills", "Power & Water", "Electricity", "Gas & Water", "Basic Utilities"],
    "Insurance": ["Protection Plans", "Coverage Fees", "Insurance Payments", "Risk Management", "Safeguard Fees"],
    "Meals and Entertainment": ["Food & Fun", "Dining", "Food & Drink", "Client Entertainment", "Social Expenses"],
    "Rent": ["Lease Payment", "Office Lease", "Rental Payment", "Space Rent", "Building Lease"],
    "Marketing": ["Promotion", "Advertising Campaigns", "Branding", "Client Outreach", "Sales Promotion"],
    "Consulting Services": ["Advisory Services", "Expert Consultation", "Consulting Fees", "Professional Advice", "Business Consulting"],
    "Fuel": ["Gasoline", "Vehicle Fuel", "Car Gas", "Fuel Charges", "Transport Fuel"],
    "Shipping": ["Delivery Costs", "Shipping Fees", "Postage", "Freight", "Courier Service"],
    "Advertising": ["Ads", "Promotional Ads", "Online Ads", "Marketing Ads", "Brand Ads"],
    "Repairs and Maintenance": ["Maintenance", "Fixes", "Repair Work", "Building Maintenance", "Upkeep"],
    "Professional Services": ["Business Services", "Expert Services", "Professional Help", "Specialist Services", "Consultancy"],
    "Software": ["IT Tools", "Business Software", "Digital Tools", "Software Subscriptions", "Tech Software"],
    "Taxes": ["Tax Payments", "Government Fees", "Dues", "Tax Obligations", "State Dues"],
    "Training": ["Skill Development", "Employee Training", "Learning", "Staff Development", "Workforce Training"],
    "Subscription": ["Memberships", "Service Subscriptions", "Recurring Payments", "Monthly Fees", "Regular Services"],
    "Legal Services": ["Lawyer Fees", "Legal Advice", "Court Fees", "Legal Consultancy", "Law Services"],
    "Telecommunications": ["Phone & Internet", "Communication Bills", "Telecom", "Mobile Bills", "Network Costs"],
    "Payroll": ["Employee Pay", "Wages", "Salaries", "Staff Payment", "Workforce Pay"],
    "Employee Benefits": ["Staff Perks", "Employee Rewards", "Workplace Benefits", "Employee Extras", "Worker Benefits"],
    "Miscellaneous": ["Other Expenses", "Various Costs", "Misc. Expenses", "Other Items", "Sundry Costs"],
    "Credit Card Payment": ["Card Payment", "Credit Balance", "Card Charges", "Card Bill", "Credit Payment"],
    "Loan Payment": ["Debt Repayment", "Loan Balance", "Debt Payment", "Loan Installment", "Loan Charges"],
    "Bank Charges": ["Service Fees", "Account Fees", "Banking Fees", "Financial Fees", "Bank Charges"]
}

# Generate user-specific categories based on the synonym mapping
def assign_synonymous_category(category, user_index):
    # Each user gets a different synonym of the original category
    if category in synonymous_categories_v2:
        return synonymous_categories_v2[category][user_index]
    else:
        return category  # If no synonym found, return the same category

# Add user and personalized category to each transaction
personalized_data = []

for idx, row in transactions_df.iterrows():
    user = random.choice(users)  # Randomly assign a user
    user_index = users.index(user)  # Get the index for that user
    personalized_category = assign_synonymous_category(row['Category'], user_index)  # Get user-specific category
    
    personalized_data.append({
        "UserID": user,
        "TransactionID": row['TransactionID'],
        "Description": row['Description'],
        "Amount": row['Amount'],
        "OriginalCategory": row['Category'],
        "PersonalizedCategory": personalized_category
    })

# Create a new DataFrame for personalized transactions
personalized_df = pd.DataFrame(personalized_data)

# Save as CSV
personalized_csv = "data/personalized_quickbooks_transactions.csv"
personalized_df.to_csv(personalized_csv, index=False)

personalized_df.head()  # Display a few rows to check the output