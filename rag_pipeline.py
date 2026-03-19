import faiss
import pickle
import numpy as np
from openai import OpenAI

# ---------------- CONFIG ----------------
FAISS_INDEX_PATH = "Dataset/faiss_index.bin"
METADATA_PATH = "Dataset/faiss_metadata.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5
# ----------------------------------------

# Load the FAISS index and metadata first so the pipeline has everything ready to query.
def load_resources():
    """Load FAISS index and metadata."""
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"FAISS vectors loaded   : {index.ntotal}")
    print(f"Metadata records loaded: {len(metadata)}")

    return index, metadata


# Turn the user's question into an embedding using the same model we used during index creation.
client = OpenAI(api_key="insert_key")

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding).astype("float32")


# Search the FAISS index with the query embedding and get back the top matching incidents.
def retrieve_incidents(index, metadata, query_vector, k=TOP_K):
    """Search FAISS and retrieve top-k incident reports with similarity scores."""

    query_vector = query_vector.reshape(1, -1)

    # normalize for cosine similarity
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        record = metadata[idx].copy()
        record["similarity_score"] = round(float(score), 4)
        results.append(record)

    return results


# Put the retrieved incidents into a clean readable context block for the final prompt.
def build_context(results):
    """Build a readable context block from retrieved incidents."""
    context_parts = []

    for i, r in enumerate(results, 1):
        part = f"""
Incident {i}
ACN: {r.get('acn', 'Unknown')}
Date: {r.get('date', 'Unknown')}
Aircraft Type: {r.get('aircraft_type', 'Unknown')}
Flight Phase: {r.get('flight_phase', 'Unknown')}
Light Condition: {r.get('light_condition', 'Unknown')}
Flight Conditions: {r.get('flight_conditions', 'Unknown')}
Anomaly: {r.get('anomaly', 'Unknown')}
Contributing Factors: {r.get('contributing_factors', 'Unknown')}
Primary Problem: {r.get('primary_problem', 'Unknown')}
Human Factors: {r.get('human_factors', 'Unknown')}
Result: {r.get('result', 'Unknown')}

Synopsis:
{r.get('synopsis', '')}

Narrative:
{r.get('narrative', '')}
"""
        context_parts.append(part.strip())

    return "\n\n" + ("\n" + "=" * 80 + "\n\n").join(context_parts)


# Send the context and user query to the model so it can generate the final response.
def generate_answer(query, context):
    """Generate final RAG answer and return usage stats."""
    prompt = f"""
You are an aviation safety analysis assistant.

Use only the incident reports below to answer the question.

Return the answer in plain text using EXACTLY the following numbered structure.
Do NOT use markdown, hashtags (#), bullet points (*), or formatting symbols.

1. DIRECT ANSWER: Provide a concise answer to the question.

2. STATISTICAL PATTERN: Describe patterns or distributions seen in the retrieved incidents.

3. CAUSAL PATHWAY or COMPARISON: Explain relationships between contributing factors,
operational context, or phases of flight.

4. SPECIFIC EVIDENCE: Reference specific incidents using ACN numbers from the retrieved reports.

5. LIMITATIONS: Explain what cannot be concluded from the available evidence.

INCIDENT REPORTS:
{context}

USER QUESTION:
{query}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You analyze aviation safety incidents using retrieved report evidence only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    input_cost_per_1m = 0.15
    output_cost_per_1m = 0.60
    cost = ((prompt_tokens / 1_000_000) * input_cost_per_1m) + \
           ((output_tokens / 1_000_000) * output_cost_per_1m)

    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6)
    }

# Run the full RAG pipeline from query to final answer.
def run_rag_query(index, metadata, query):
    """Full RAG pipeline with evaluation-ready output."""
    import time

    start_time = time.time()

    query_vector = embed_query(query)
    results = retrieve_incidents(index, metadata, query_vector, k=TOP_K)
    context = build_context(results)
    generation = generate_answer(query, context)

    end_time = time.time()

    return {
        "query": query,
        "pipeline": "RAG",
        "answer": generation["answer"],
        "retrieved_acns": [r["acn"] for r in results],
        "similarity_scores": [r["similarity_score"] for r in results],
        "time_seconds": round(end_time - start_time, 2),
        "prompt_tokens": generation["prompt_tokens"],
        "output_tokens": generation["output_tokens"],
        "cost_usd": generation["cost_usd"],
        "retrieved_results": results,
        "context": context
    }


# Run the script interactively so we can type a question and see the retrieved incidents and answer.
if __name__ == "__main__":
    index, metadata = load_resources()

    query = input("Enter your aviation safety question: ").strip()
    if not query:
        query = "What are the common contributing factors in wake turbulence incidents during approach or takeoff?"

    try:
        output = run_rag_query(index, metadata, query)

        print("\nQUERY:\n")
        print(output["query"])

        print("\nTOP RETRIEVED INCIDENTS:\n")
        for r in output["retrieved_results"]:
            print("ACN:", r["acn"])
            print("Phase:", r["flight_phase"])
            print("Factors:", r["contributing_factors"])
            print("Synopsis:", r["synopsis"][:150])
            print("-" * 60)

        print("\nFINAL RAG ANSWER:\n")
        print(output["answer"])

    except Exception as e:
        print("\nPIPELINE RAN UNTIL ANSWER GENERATION, BUT API CALL FAILED.")
        print("Error:", e)
