import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "retrieval_pairs.csv"


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} email-response pairs.")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

print("Creating embeddings...")

embeddings = model.encode(
    df["email_body"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)


# --------------------------------------------------
# Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings.astype("float32"))


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve_similar_emails(query, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

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


# --------------------------------------------------
# Generate response
# --------------------------------------------------

def generate_reply(incoming_email):

    retrieved = retrieve_similar_emails(
        incoming_email,
        top_k=3
    )

    examples = ""

    for i, item in enumerate(retrieved, start=1):

        examples += f"""
Example {i}

Historical email:
{item["email"]}

Historical reply:
{item["reply"]}
"""


    prompt = f"""
You are an AI email assistant.

Generate a suggested reply to the incoming email.

Use the historical email/reply examples as guidance for
tone, style, and how similar messages were answered.

Do NOT copy a historical reply blindly.

Do NOT invent facts, promises, policies, prices, dates,
or actions that are not supported by the incoming email
or historical examples.

Keep the response natural, concise, polite, and relevant.

Historical examples:
{examples}

Incoming email:
{incoming_email}

Write ONLY the suggested reply.
"""


    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text, retrieved


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    incoming_email = "I just received a promotion at work!"

    print("\nIncoming email:")
    print(incoming_email)

    reply, retrieved = generate_reply(incoming_email)

    print("\nRetrieved examples:")

    for i, item in enumerate(retrieved, start=1):

        print(f"\n--- Example {i} ---")
        print(f"Similarity: {item['similarity']:.3f}")
        print(f"Email: {item['email']}")
        print(f"Reply: {item['reply']}")

    print("\n==============================")
    print("SUGGESTED REPLY")
    print("==============================")
    print(reply)