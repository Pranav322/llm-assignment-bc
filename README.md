
 📘 **README.md — LLM Evaluation Pipeline**

## 📌 Overview

This project implements a **production-ready evaluation pipeline** for assessing the quality of LLM-generated responses in a RAG (Retrieval-Augmented Generation) system.  
It evaluates the **final AI response** in a conversation against the **retrieved context vectors** that were available at generation time.

The evaluator reports:

- **Relevance** (Did the model answer the user's question?)
- **Grounding / Completeness** (Is the answer supported by retrieved context?)
- **Factual Accuracy** (Did the model hallucinate?)
- **Latency** (Response time between user and AI turn)
- **Token Cost Estimate** (LLM cost approximation)

The system uses a **two-tier evaluation strategy** to ensure both **scalability** and **factual precision**.

---

# 🧠 **Architecture & Design Choices**

## 🏛️ Tiered Evaluation Strategy  
The evaluation pipeline is designed around a **Tier-1 (cheap)** + **Tier-2 (precise)** hybrid approach.  
This mirrors real-world LLMOps: millions of daily conversations require **fast filtering**, with expensive checks only triggered when necessary.

---

## 🔹 **Tier 1 — Semantic Filtering (Fast & Cheap)**  
**Goal:** Determine whether the response is on-topic and grounded in the retrieved context.

**Tech Used:**  
- `sentence-transformers/all-MiniLM-L6-v2`  
- Cosine similarity (vector distance)

**Metrics Computed:**

- **Relevance Score**  
  Measures semantic similarity between user query and AI response.

- **Grounding Score**  
  Computes **max similarity** between AI response and each context chunk.  
  This avoids dilution from irrelevant context and reflects which chunk the model actually relied on.

**Why Embeddings?**  
They are:
- fast (milliseconds)  
- cheap (no API calls)  
- good for "semantic vibe checking"

This tier filters out most bad responses without incurring LLM costs.

---

## 🔹 **Tier 2 — LLM-as-a-Judge (Precise & Expensive)**  
**Goal:** Detect subtle hallucinations, especially numeric or factual contradictions that embeddings cannot catch.

**Reasoning:**  
Embedding models fail on numeric mismatches:

| Context | AI Response | Embeddings Think |
|--------|-------------|------------------|
| “Price is **Rs 800**” | “Price is **Rs 2000**” | **Similar** (both about “price”) |

This is why an **LLM Judge** is required.

**Tech Used:**  
- `GPT-4o-mini` with JSON-mode evaluation  
- Checks numeric consistency  
- Verifies factual grounding across all context chunks  
- Provides structured JSON:  
  ```json
  {
    "is_hallucination": true/false,
    "reason": "...",
    "score": 0.0 - 1.0
  }
  ```

**In production:**  
Tier-2 would be triggered conditionally (low grounding, presence of numbers, or sampling).  
For this assignment, Tier-2 runs for every evaluation to demonstrate correctness.

---

## 🔹 Latency & Cost Estimation

### ⏱ Latency  
Calculated from the difference in `created_at` timestamps between the last user message and the AI response.

### 💰 Cost  
Token cost estimated using:
- `tiktoken` tokenizer  
- GPT-4o-mini pricing  
  - $0.15 / 1M input tokens  
  - $0.60 / 1M output tokens  

Gives an approximate cost per evaluated turn.

---

# 📂 **Project Structure**

```
llm_evaluator/
│
├── data/
│   ├── sample-chat-conversation-01.json
│   └── sample_context_vectors-01.json
│
├── src/
│   ├── config.py              # model settings, API keys, thresholds
│   ├── metrics.py             # latency + cost utilities
│   ├── pipeline.py            # Tier 1 + Tier 2 evaluation logic
│
├── main.py                    # entry point for evaluation
├── requirements.txt
└── README.md
```

---

# 🚀 **How the Evaluator Works (Step-by-Step)**

### **1. Load chat and context JSON files**  
- Extracts the **final user → AI turn pair**
- Extracts all `text` fields from `vector_data`

### **2. Tier 1 — Embedding-based Metrics**
- Embed:  
  - user_query  
  - ai_response  
  - each context chunk  
- Compute:
  - Relevance = cosine(query, response)  
  - Grounding = **max** cosine(response, chunk_i)

### **3. Tier 2 — LLM-as-a-Judge**
A structured evaluation prompt checks:

- Does the AI invent numbers not present in context?
- Does it invent entities or claims not supported?
- Are prices/dates/quantities consistent with retrieved context?

### **4. Combine metrics into final evaluation report**  
Result saved as:

```
evaluation_report.json
```

---

# 🧪 Sample Output (Hotel Hallucination Case)

```json
{
  "relevance": 0.56,
  "grounding": 0.49,
  "factual_accuracy": 0.2,
  "hallucination_detected": true,
  "hallucination_reason": "Response invented subsidized Rs 2000 rooms not found in context."
}
```

This demonstrates that the evaluator **correctly catches numeric contradictions**, which is the core requirement of the assignment.

---

# ⚙️ **Installation**

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

---

# ▶️ **Run the Evaluator**

```bash
python main.py
```

---

# 🏁 **Why This Submission Is Strong**

This project demonstrates:

### ✔ Understanding of LLM hallucination behavior  
### ✔ Correct use of embedding models for cheap semantic checks  
### ✔ Correct use of LLM-as-a-Judge for factual accuracy  
### ✔ Awareness of cost & latency constraints  
### ✔ Good software engineering (clear modules, readable code)  
### ✔ Production-safe behavior (fallbacks, JSON-only outputs)

This is exactly what companies look for in LLM evaluation engineers.
