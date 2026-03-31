"""
bronze_to_silver.py
-------------------
Phase 3 - Data Cleaning and Transformation Script
Azure Banking Data Platform

What this script does in simple terms:
- Opens the raw fraud CSV files from the bronze folder in Azure
- Cleans them up (removes bad rows, fixes dates, adds useful columns)
- Saves the cleaned result into the silver folder in Azure
"""

import os
import pandas as pd
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


# ------------------------------------
# STEP 1 - Load our Azure password
# ------------------------------------
# We stored our Azure connection string in the .env file
# so it never gets uploaded to GitHub by accident.
# This line reads that secret and loads it into the script.
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# If the connection string is missing, stop the script and tell us
if not connection_string:
    raise ValueError("Azure connection string not found. Check your .env file.")

print("Step 1 done - Azure password loaded successfully")


# ------------------------------------
# STEP 2 - Connect to Azure Storage
# ------------------------------------
# This is like logging into your Azure storage account from Python.
# We create a client which is basically a connection handle
# that lets us read and write files in our storage account.
blob_service = BlobServiceClient.from_connection_string(connection_string)

# Get a handle to the bronze container (where raw files live)
bronze_container = blob_service.get_container_client("bronze")

# Get a handle to the silver container (where cleaned files will go)
silver_container = blob_service.get_container_client("silver")

print("Step 2 done - Connected to Azure storage account")


# ------------------------------------
# STEP 3 - Read the CSV files from bronze
# ------------------------------------
# This helper function downloads a single CSV file from Azure
# and loads it into pandas as a table called a DataFrame
def read_csv_from_blob(container_client, blob_path):
    print(f"  Downloading {blob_path} from Azure...")
    blob_client = container_client.get_blob_client(blob_path)
    data = blob_client.download_blob().readall()
    df = pd.read_csv(BytesIO(data))
    print(f"  Downloaded successfully - {len(df):,} rows loaded")
    return df

print("\nStep 3 - Downloading files from bronze container...")

# Download the training file (the big one - about 1.2M rows)
df_train = read_csv_from_blob(bronze_container, "fraud-transactions/fraudTrain.csv")

# Download the test file (about 550K rows)
df_test = read_csv_from_blob(bronze_container, "fraud-transactions/fraudTest.csv")


# ------------------------------------
# STEP 4 - Combine both files into one
# ------------------------------------
# We have two separate files but we want one big dataset.
# pd.concat stacks them on top of each other like two spreadsheets merged.
# ignore_index=True just resets the row numbers starting from 0.
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"\nStep 4 done - Combined both files into one dataset")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")


# ------------------------------------
# STEP 5 - Clean up the data
# ------------------------------------
print("\nStep 5 - Cleaning the data...")

# The first column is just a row number from Kaggle - we do not need it
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("  Removed the unnecessary row number column")

# Make all column names lowercase with underscores instead of spaces
# For example: "Transaction Amount" becomes "transaction_amount"
# This makes it easier to type and work with in SQL later
df.columns = (df.columns
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("(", "")
              .str.replace(")", ""))
print("  Fixed all column names to be lowercase with underscores")

# Remove rows where the most important columns are empty or missing
# We cannot use a transaction that has no date, amount, fraud label, or date of birth
critical_cols = ["trans_date_trans_time", "amt", "is_fraud", "dob"]
rows_before = len(df)
df = df.dropna(subset=critical_cols)
rows_after = len(df)
print(f"  Removed {rows_before - rows_after:,} rows that had missing critical values")

# Remove exact duplicate transactions - same transaction ID appearing twice
rows_before = len(df)
df = df.drop_duplicates(subset=["trans_num"])
rows_after = len(df)
print(f"  Removed {rows_before - rows_after:,} duplicate transaction rows")


# ------------------------------------
# STEP 6 - Extract useful time information
# ------------------------------------
# The transaction timestamp looks like "2019-01-01 00:00:18"
# We want to pull out the hour, day, month and year separately
# so analysts can ask questions like which hour of day has most fraud
print("\nStep 6 - Pulling out time details from the transaction timestamp...")

# Convert the text timestamp into a proper datetime object Python understands
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

# Now extract each piece separately as new columns
df["transaction_date"] = df["trans_date_trans_time"].dt.date       # just the date e.g. 2019-01-01
df["transaction_hour"] = df["trans_date_trans_time"].dt.hour       # hour 0-23 e.g. 14 means 2pm
df["day_of_week"]      = df["trans_date_trans_time"].dt.day_name() # e.g. Monday or Tuesday
df["month"]            = df["trans_date_trans_time"].dt.month      # 1 to 12
df["year"]             = df["trans_date_trans_time"].dt.year       # e.g. 2019

print("  Added new columns: transaction_date, transaction_hour, day_of_week, month, year")


# ------------------------------------
# STEP 7 - Calculate customer age
# ------------------------------------
# The dataset has a date of birth column but not the customer age.
# We calculate age by finding the difference between transaction date and birth date.
print("\nStep 7 - Calculating customer age from date of birth...")

# Convert date of birth to a proper date format Python understands
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

# Calculate age in years at the time of each transaction
df["age"] = df.apply(
    lambda row: (row["trans_date_trans_time"] - row["dob"]).days // 365
    if pd.notnull(row["dob"]) else None,
    axis=1
)

# Group ages into buckets - much easier to analyse than individual ages
# For example instead of saying customer was 34 we say they are in the 26-35 group
def assign_age_group(age):
    if pd.isnull(age):  return "Unknown"
    if age < 18:        return "Under 18"
    if age <= 25:       return "18-25"
    if age <= 35:       return "26-35"
    if age <= 50:       return "36-50"
    if age <= 65:       return "51-65"
    return "65+"

df["age_group"] = df["age"].apply(assign_age_group)

print("  Added new columns: age and age_group")
print(f"  Age group breakdown: {df['age_group'].value_counts().to_dict()}")


# ------------------------------------
# STEP 8 - Label transactions by amount size
# ------------------------------------
# Instead of just having a raw dollar amount we add a label
# so analysts can easily filter for example show me all Premium transactions
print("\nStep 8 - Labelling transactions by amount size...")

def assign_amount_tier(amt):
    if pd.isnull(amt): return "Unknown"
    if amt < 50:       return "Low"       # under $50
    if amt < 200:      return "Mid"       # $50 to $199
    if amt < 500:      return "High"      # $200 to $499
    return "Premium"                       # $500 and above

df["amount_tier"] = df["amt"].apply(assign_amount_tier)

# Also add a simple True or False flag for transactions over $500
# Banks pay special attention to these as they often trigger a manual review
df["high_value_transaction"] = df["amt"] > 500

print("  Added new columns: amount_tier and high_value_transaction")
print(f"  Amount tier breakdown: {df['amount_tier'].value_counts().to_dict()}")


# ------------------------------------
# STEP 9 - Print a summary of what we built
# ------------------------------------
print("\n" + "="*50)
print("CLEANED DATASET SUMMARY")
print("="*50)
print(f"Total rows:             {len(df):,}")
print(f"Total columns:          {len(df.columns)}")
print(f"Fraud transactions:     {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}% of all transactions)")
print(f"Date range:             {df['transaction_date'].min()} to {df['transaction_date'].max()}")
print(f"Transaction amount:     ${df['amt'].min():.2f} to ${df['amt'].max():.2f}")
print(f"High value over $500:   {df['high_value_transaction'].sum():,} transactions")
print("="*50)


# ------------------------------------
# STEP 10 - Save the cleaned data to silver
# ------------------------------------
# We save as Parquet format instead of CSV because:
# Parquet is compressed so the file is much smaller (400MB CSV becomes around 50MB)
# Parquet is much faster to read when you run SQL queries in Synapse
# It is the industry standard format used in all real data lakes
print("\nStep 10 - Saving cleaned data to silver container...")

# Write the DataFrame into memory as a Parquet file
output_buffer = BytesIO()
df.to_parquet(output_buffer, index=False, engine="pyarrow")
output_buffer.seek(0)  # Go back to the start so Azure can read it from the beginning

# Upload that Parquet file to the silver container in Azure
silver_blob_path = "fraud-transactions/fraud_cleaned.parquet"
silver_container.get_blob_client(silver_blob_path).upload_blob(
    output_buffer,
    overwrite=True
)

file_size_mb = output_buffer.tell() / 1024 / 1024
print(f"Saved to: silver/{silver_blob_path}")
print(f"File size: {file_size_mb:.1f} MB")
print("\nPhase 3 complete! Cleaned data is now in the silver layer and ready for Synapse.")
