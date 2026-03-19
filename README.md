# ✈️ RAG vs GraphRAG — Aviation Safety Intelligence System

> Most AI systems don’t fail because they hallucinate.  
> They fail because they **don’t see enough data to reason correctly.**

This project demonstrates that gap using **30,513 real-world aviation safety incidents** from NASA’s ASRS dataset.

---

## 🚀 Overview

We built and compared two AI systems:

- 🔍 **RAG (Retrieval-Augmented Generation)**  
  → Retrieves a few similar incidents and summarizes them

- 🧠 **GraphRAG (Knowledge Graph + LLM)**  
  → Builds relationships across the dataset to enable reasoning

👉 Same dataset. Same query. Completely different outcomes.

---

## ⚡ The Core Problem

Traditional RAG systems only analyze a **small retrieved subset** of data.

Example:

**Query:**  
> “How many incidents involved equipment failure?”

- RAG answer: *5 incidents* (based on retrieved docs)  
- Actual dataset: **13,000+ incidents**

💥 This is not hallucination — it is an **architectural limitation**

---

## 🏗️ System Design

### 🔍 RAG Pipeline

- Convert query → embedding
- Retrieve top-K incidents using FAISS
- Generate answer using LLM

👉 Works well for:
- Summarization
- Similarity search

👉 Fails at:
- Counting
- Global patterns
- Causal reasoning

---

### 🧠 GraphRAG Pipeline

- Extract entities (causes, factors, outcomes)
- Build knowledge graph
- Traverse graph based on query
- Combine graph insights + LLM reasoning

👉 Enables:
- Dataset-wide analysis
- Pattern discovery
- Causal reasoning

---

## 📊 Key Capabilities Comparison

| Capability            | RAG ❌ | GraphRAG ✅ |
|----------------------|--------|------------|
| Uses full dataset    | ❌     | ✅         |
| Accurate counting    | ❌     | ✅         |
| Pattern detection    | ❌     | ✅         |
| Causal reasoning     | ❌     | ✅         |

---

## 🔗 Example Insight

Instead of:

> “Here are 5 similar incidents…”

GraphRAG produces:

```
Weather → Pilot Delay → Runway Overshoot
```

👉 Not just *what happened* — but **why it happened**

---

## 🖥️ Features

- Side-by-side UI:
  - Left → RAG
  - Right → GraphRAG
- Structured outputs:
  - Direct Answer
  - Statistical Patterns
  - Causal Pathways
  - Evidence
- Knowledge graph visualization (D3.js)
- Query decomposition engine
- Token + cost tracking

---

## 🧪 Tech Stack

- LLM: OpenAI (GPT-4o-mini)
- Vector Search: FAISS
- Backend: Flask
- Frontend: HTML + D3.js
- Graph Processing: NetworkX
- Dataset: NASA ASRS (30K+ incidents)

---

## ▶️ Running the Project

```bash
git clone <your-repo-url>
cd <repo>

pip install -r requirements.txt
python graphrag_app.py
```

Then open:

http://localhost:5000

---

## ⚠️ Important Setup

Inside `graphrag_app.py`, replace:

```python
OPENAI_API_KEY = "your-api-key"
```

---

## 💡 Key Takeaways

- RAG is good for **retrieval + summarization**
- GraphRAG is required for:
  - **Aggregation**
  - **Patterns**
  - **Causality**

👉 Future AI systems will combine both.

---

## 🚀 Future Work

- Hybrid RAG + Graph reasoning
- Temporal trend analysis
- Real-time risk scoring
- Cross-domain applications (finance, healthcare)

---

