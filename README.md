# AI Email Suggested-Response System

An AI-powered email suggested-response system built for the Hiver SDE internship coding challenge.

Given an incoming email, the system retrieves similar historical email/reply examples and uses a generative LLM to produce a suggested response. The project also includes an evaluation system that scores each generated response across multiple quality dimensions.

## 1. Problem

The goal is to generate a useful suggested reply for an incoming email while learning from previously seen email/reply pairs.

The system:
1. Builds a dataset of historical email/reply pairs.
2. Retrieves relevant historical examples.
3. Uses a generative AI model to produce a suggested reply.
4. Measures response quality using multiple evaluation dimensions.
5. Reports per-response and overall scores.

## 2. Dataset

The project uses a public dataset containing email messages and corresponding replies.

The original dataset contained 9,504 records. Many records represented repeated evaluation records for the same underlying email/reply pair.

During preprocessing:
- Raw records: 9,504
- Unique email/reply pairs: 132
- Clean pairs after removing missing/empty values: 132

The preprocessing code is `src/prepare_data.py`.

The cleaned dataset is stored in `data/pairs.csv`.

### Dataset representativeness

The dataset provides real examples of email conversations and responses, making it useful for demonstrating email-response generation.

However, it is a small dataset with only 132 unique pairs and is not claimed to represent Hiver's production customer-support email traffic. The examples contain general email conversations rather than a large, domain-specific customer-support corpus.

A larger customer-support-specific dataset would be required for production use.

## 3. Train/Retrieval and Test Split

The 132 unique pairs are split into:
- 92 retrieval examples
- 40 held-out test examples

The split is implemented in `src/split_data.py` using `random_state=42`.

The retrieval system uses only:

`data/retrieval_pairs.csv`

The held-out evaluation set is:

`data/test_pairs.csv`

This prevents the test examples from being included in the retrieval index during evaluation.

## 4. Response Generation

The system uses:
- Sentence Transformers
- FAISS similarity search
- Retrieval-augmented prompting
- OpenAI GPT-5-mini

Architecture:

```
Incoming Email
    |
    v
Sentence Transformer
    |
    v
FAISS Similarity Search
    |
    v
Top-3 Historical Email/Reply Examples
    |
    v
LLM Prompt + Retrieved Context
    |
    v
Generated Suggested Reply
    |
    v
Evaluation
```

## 5. Retrieval

The system uses `all-MiniLM-L6-v2` from Sentence Transformers to convert emails into vector embeddings.

The embeddings are normalized and stored in a FAISS `IndexFlatIP` index.

Because the embeddings are normalized, inner-product similarity corresponds to cosine similarity.

For each incoming email, the system retrieves the top 3 most similar historical email/reply examples.

## 6. Why RAG?

The challenge allows prompting, RAG/retrieval, few-shot examples, fine-tuning, or a combination.

RAG was chosen because the dataset is small.

Advantages:
- No model training is required.
- Works with a small dataset.
- Historical examples can be updated without retraining.
- Provides concrete examples to guide generation.
- Allows the LLM to generate a new response rather than simply selecting an existing reply.

Fine-tuning was not selected because 132 unique pairs provide limited training data, and fine-tuning would add unnecessary complexity for this prototype.

## 7. Generative Model

The response generator uses OpenAI `gpt-5-mini`.

Retrieved historical examples are included in the prompt.

The prompt instructs the model to:
- Produce a concise response.
- Address the incoming email.
- Be natural and polite.
- Use historical responses as guidance.
- Avoid blindly copying retrieved responses.
- Avoid inventing facts, policies, prices, dates, or actions.
- Output only the suggested reply.

The generator is implemented in `src/generate.py`.

## 8. What Does "Accurate" Mean?

Exact string matching is too strict for natural-language email responses.

For example:

"Thanks for the update!"

and

"Thank you for letting me know."

are different strings but can both be appropriate responses.

Therefore, response quality is evaluated using multiple dimensions.

## 9. Evaluation Metric

Each generated response is evaluated from 0 to 10 across five dimensions:

| Criterion | Weight |
|---|---:|
| Relevance | 25% |
| Correctness | 25% |
| Groundedness | 20% |
| Completeness | 20% |
| Tone | 10% |

The final score is:

```
Overall Score =
    0.25 × Relevance +
    0.25 × Correctness +
    0.20 × Groundedness +
    0.20 × Completeness +
    0.10 × Tone
```

### Why these metrics?

**Relevance (25%)**
Checks whether the response addresses the incoming email.

**Correctness (25%)**
Checks whether the response avoids incorrect claims and invented facts.

**Groundedness (20%)**
Checks whether the response is supported by the incoming email and retrieved context.

**Completeness (20%)**
Checks whether the response sufficiently addresses the main point.

**Tone (10%)**
Checks whether the response is natural, polite, and appropriate.

Correctness and relevance receive the highest weights because a fluent response that is incorrect or unrelated is not useful.

## 10. Evaluation Method

An LLM judge evaluates each generated response using:
- Incoming email
- Reference reply
- Generated reply

The reference reply is used as a comparison point, not as an exact-match target.

The generated response does not need to reproduce the reference word-for-word, because multiple valid responses can exist.

The evaluator produces per-response scores and a short explanation.

The evaluation implementation is in `src/evaluate.py`.

## 11. Results

The current evaluation contains 40 held-out test examples.

The previous evaluation run produced:

**Average LLM-judge quality score: 9.75 / 10**

This is an LLM-based quality score, not a conventional 97.5% accuracy measurement.

The complete per-response results are stored in:

`results/evaluation.json`

## 12. Metric Limitations

The evaluation metric is designed to capture multiple aspects of email quality rather than exact text similarity.

However, LLM judging has limitations:
- The judge may overestimate or underestimate quality.
- The reference reply is not necessarily the only valid answer.
- The test set is small.
- The underlying dataset is small.
- The dataset is not specifically representative of production customer-support traffic.

A stronger future evaluation would compare LLM-judge scores against human ratings on a sample of generated responses.

## 13. Failure Analysis

One observed failure involved a scheduling email:

**Incoming email:**
> "It would be good to have a meeting. Will you be around this afternoon?"

**Generated response:**
> "Yes — I'll be around this afternoon. What time works for you?"

The evaluation judge gave this response a lower score because the reference response indicated that the sender would not be available.

This demonstrates a key failure mode: the model can make an unsupported commitment when the incoming email does not provide enough information about the user's availability.

A production system should be more conservative about unsupported commitments involving availability, actions, dates, prices, or policies.

## 14. Project Structure

```
hiver-ai-email-response/
│
├── data/
│   ├── pairs.csv
│   ├── retrieval_pairs.csv
│   └── test_pairs.csv
│
├── results/
│   └── evaluation.json
│
├── src/
│   ├── prepare_data.py
│   ├── split_data.py
│   ├── retrieve.py
│   ├── generate.py
│   └── evaluate.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The original raw dataset is intentionally not committed to the public repository.

## 15. Installation

Clone the repository:

```bash
git clone https://github.com/sarveshm555/hiver-ai-email-response.git
cd hiver-ai-email-response
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

Never commit the `.env` file.

## 16. Running the System

Prepare the dataset:

```bash
python src/prepare_data.py
```

Create the retrieval/test split:

```bash
python src/split_data.py
```

Test retrieval:

```bash
python src/retrieve.py
```

Generate a suggested response:

```bash
python src/generate.py
```

Run the evaluation:

```bash
python src/evaluate.py
```

The evaluation results are written to:

`results/evaluation.json`

## 17. Technology Stack

- Python
- Pandas
- NumPy
- Sentence Transformers
- FAISS
- OpenAI API
- python-dotenv

## 18. Design Trade-offs

### RAG vs Fine-tuning

RAG was selected because the dataset contains only 132 unique email/reply pairs. It avoids the complexity of fine-tuning and allows historical examples to be updated without retraining the model.

### Top-3 Retrieval

The system retrieves three similar historical examples. This provides multiple examples for context while keeping the prompt reasonably small.

### LLM Generation

The LLM generates a new response instead of directly returning a retrieved reply. This provides flexibility, but generation can also introduce unsupported claims, which is why correctness and groundedness are explicitly evaluated.

### LLM-based Evaluation

An LLM judge allows multiple dimensions of response quality to be evaluated instead of relying on exact string matching. However, automated judging is not equivalent to human ground truth and should be validated further with human evaluation.

## 19. Limitations

The current prototype has several limitations:

- The dataset contains only 132 unique email/reply pairs.
- The dataset is not specifically representative of Hiver's production customer-support traffic.
- The held-out test set contains only 40 examples.
- The evaluation relies on an LLM judge.
- The reference reply is not necessarily the only valid response.
- Generated responses can occasionally make unsupported commitments.

Therefore, the current score should be treated as an evaluation signal for this prototype rather than a production accuracy guarantee.

## 20. Future Improvements

- Build or obtain a larger customer-support-specific email dataset.
- Add human evaluation of generated responses.
- Measure agreement between human and LLM judges.
- Add retrieval-quality evaluation.
- Add confidence thresholds for low-similarity emails.
- Detect unsupported commitments before returning a suggestion.
- Add privacy and PII filtering for real-world email data.
- Add production monitoring and user feedback.

## 21. AI Tools Used

AI coding assistants were used during development for code generation, debugging, explanation, and iteration.

The generated code was reviewed, tested, and modified during development.

## 22. Conclusion

This project demonstrates an end-to-end Gen-AI email suggested-response system using retrieval-augmented generation.

The system combines semantic retrieval with an LLM to generate responses from historical email/reply examples and evaluates response quality across relevance, correctness, groundedness, completeness, and tone.

The implementation is intentionally lightweight and reproducible while clearly documenting the limitations of the dataset and automated evaluation.
