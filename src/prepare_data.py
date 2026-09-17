import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output files
INPUT_FILE = BASE_DIR / "data" / "final_public.tsv"
OUTPUT_FILE = BASE_DIR / "data" / "pairs.csv"

print("Loading dataset...")

# Read TSV file
df = pd.read_csv(INPUT_FILE, sep="\t")

print(f"Original rows: {len(df)}")
print(f"Unique pair IDs: {df['pair_id'].nunique()}")

# Keep only what we need
pairs = df[["pair_id", "email_body", "email_reply"]].copy()

# Keep one row for each email/reply pair
pairs = pairs.drop_duplicates(subset=["pair_id"])

# Remove missing values
pairs = pairs.dropna(subset=["email_body", "email_reply"])

# Remove empty emails/replies
pairs = pairs[
    (pairs["email_body"].str.strip() != "") &
    (pairs["email_reply"].str.strip() != "")
]

# Add numeric ID
pairs.insert(0, "id", range(1, len(pairs) + 1))

# Save clean dataset
pairs.to_csv(OUTPUT_FILE, index=False)

print(f"Clean pairs: {len(pairs)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst 5 examples:")
print(pairs.head().to_string(index=False))