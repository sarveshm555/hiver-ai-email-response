import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "pairs.csv"

TRAIN_FILE = BASE_DIR / "data" / "retrieval_pairs.csv"
TEST_FILE = BASE_DIR / "data" / "test_pairs.csv"


# Load dataset
df = pd.read_csv(INPUT_FILE)

print(f"Total pairs: {len(df)}")


# Shuffle deterministically so results are reproducible
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# 70% retrieval / 30% evaluation
split_index = int(len(df) * 0.70)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


# Save
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


print(f"Retrieval pairs: {len(train_df)}")
print(f"Test pairs: {len(test_df)}")

print(f"\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)