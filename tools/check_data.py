import pandas as pd
import sqlite3
# Set display options
pd.set_option('display.max_columns', None)       # Show all columns
pd.set_option('display.max_rows', None)          # Show all rows (use with care)
pd.set_option('display.max_colwidth', None)      # Show full content of each cell
pd.set_option('display.expand_frame_repr', False)  # Don't wrap lines
# Connect and read data into a DataFrame
sql_url = "/Users/shuhangge/Desktop/my_projects/Sekai/DataAgent/DataProcess/event_analysis.db"
conn = sqlite3.connect(sql_url)
df = pd.read_sql_query("SELECT * FROM device_event_dictionaries", conn)

print(df.head())  # Show first few rows
df.to_csv("output.csv", index=False, encoding='utf-8')
conn.close()