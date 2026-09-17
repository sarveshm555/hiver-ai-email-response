import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "pairs.csv"


print("Loading email-response dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} email-response pairs.")


# Load lightweight embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Convert emails into embeddings
print("Creating embeddings...")

embeddings = model.encode(
    df["email_body"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"Embedding shape: {embeddings.shape}")


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings.astype("float32"))

print(f"FAISS index contains {index.ntotal} emails.")


def retrieve_similar_emails(query, top_k=3):

    # Create embedding for incoming email
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search
    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        results.append({
            "email": df.iloc[idx]["email_body"],
            "reply": df.iloc[idx]["email_reply"],
            "similarity": float(score)
        })

    return results


if __name__ == "__main__":

    test_email = "I just received a promotion at work!"

    print("\nIncoming email:")
    print(test_email)

    print("\nSimilar historical emails:\n")

    results = retrieve_similar_emails(test_email)

    for i, result in enumerate(results, start=1):

        print(f"--- Result {i} ---")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Historical email: {result['email']}")
        print(f"Historical reply: {result['reply']}")
        print()
        