import pandas as pd

# Load only the first N rows from the original Excel file
input = '/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/development_doc/bq-results-20250527-055335-1748325234441.csv'
df = pd.read_csv(input, nrows=100)  # change 10 to however many rows you want

# Save to a new Excel file
df.to_csv("/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/development_doc/examples_100.csv", index=False)