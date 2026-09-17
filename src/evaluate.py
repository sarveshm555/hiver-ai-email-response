import pandas as pd
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "data" / "test_pairs.csv"
RESULT_FILE = BASE_DIR / "results" / "evaluation.json"


# --------------------------------------------------
# OpenAI
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

df = pd.read_csv(TEST_FILE)

print(f"Loaded {len(df)} test examples.")


# --------------------------------------------------
# Evaluation prompt
# --------------------------------------------------

def evaluate_response(
    email,
    reference_reply,
    generated_reply
):

    prompt = f"""
You are evaluating an AI-generated email response.

Evaluate the generated response against the incoming email
and the reference response.

IMPORTANT:
The reference response is a useful comparison point, but
the generated response does NOT need to match it word-for-word.
A different response can still be good if it appropriately
answers the email.

Score each criterion from 0 to 10.

Criteria:

1. Relevance:
Does the generated reply address the incoming email?

2. Correctness:
Does it avoid incorrect claims or invented facts?

3. Groundedness:
Is the response consistent with the available context
and the style/content of the reference response?

4. Completeness:
Does it sufficiently address the main point of the email?

5. Tone:
Is the response natural, polite, and appropriate?

Incoming email:
{email}

Reference reply:
{reference_reply}

Generated reply:
{generated_reply}

Return ONLY valid JSON:

{{
    "relevance": 0,
    "correctness": 0,
    "groundedness": 0,
    "completeness": 0,
    "tone": 0,
    "reason": "short explanation"
}}
"""


    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        print("Warning: Could not parse judge response:")
        print(text)

        return {
            "relevance": 0,
            "correctness": 0,
            "groundedness": 0,
            "completeness": 0,
            "tone": 0,
            "reason": "Judge output could not be parsed."
        }


# --------------------------------------------------
# Generate replies and evaluate
# --------------------------------------------------

# Import our generation function
import sys

sys.path.append(str(BASE_DIR / "src"))

from generate import generate_reply


results = []


# For the first version, evaluate 10 examples.
# We can expand to all 40 after confirming everything works.

evaluation_count = len(df)


for i in range(evaluation_count):

    row = df.iloc[i]

    email = row["email_body"]
    reference = row["email_reply"]

    print("\n====================================")
    print(f"Evaluating example {i + 1}/{evaluation_count}")
    print("====================================")

    print("Email:")
    print(email)

    # Generate response
    generated, retrieved = generate_reply(email)

    print("\nGenerated reply:")
    print(generated)

    # Judge
    score = evaluate_response(
        email,
        reference,
        generated
    )

    print("\nEvaluation:")
    print(score)


    # Calculate weighted score

    overall = (
        score["relevance"] * 0.25
        + score["correctness"] * 0.25
        + score["groundedness"] * 0.20
        + score["completeness"] * 0.20
        + score["tone"] * 0.10
    )


    result = {
        "id": int(row["id"]),
        "email": email,
        "reference_reply": reference,
        "generated_reply": generated,
        "scores": score,
        "overall_score": round(overall, 2)
    }

    results.append(result)

    print(f"\nOverall score: {overall:.2f}/10")


# --------------------------------------------------
# Overall result
# --------------------------------------------------

average_score = sum(
    r["overall_score"] for r in results
) / len(results)


output = {
    "num_evaluated": len(results),
    "average_score": round(average_score, 2),
    "results": results
}


# Save results

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n====================================")
print("FINAL EVALUATION")
print("====================================")

print(
    f"Examples evaluated: {len(results)}"
)

print(
    f"Average score: {average_score:.2f}/10"
)

print(
    f"\nSaved results to:\n{RESULT_FILE}"
)