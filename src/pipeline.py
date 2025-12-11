import json
import time
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from src.config import Config
from src.metrics import MetricsCalculator


class EvaluationPipeline:
    def __init__(self):
        print("Initializing Evaluation Pipeline...")
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.metrics = MetricsCalculator()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def _embed(self, text: str):
        return self.encoder.encode([text])[0]

    def _cosine(self, a, b):
        return cosine_similarity([a], [b])[0][0]

    # ---------------- Tier 1 ---------------- #
    def _tier_1(self, query: str, response: str, chunks: List[str]):
        q_vec = self._embed(query)
        r_vec = self._embed(response)

        relevance = self._cosine(q_vec, r_vec)

        grounding_scores = []
        for chunk in chunks:
            grounding_scores.append(self._cosine(r_vec, self._embed(chunk)))

        grounding = max(grounding_scores) if grounding_scores else 0.0

        return {
            "relevance": float(relevance),
            "grounding": float(grounding)
        }

    # ---------------- Tier 2 ---------------- #
    def _tier_2(self, context_text: str, response: str):
        prompt = f"""
        CONTEXT:
        {context_text}

        AI RESPONSE:
        {response}

        TASK:
        Identify factual inconsistencies between RESPONSE and CONTEXT.
        Focus specifically on numeric values and entity names.
        Output JSON ONLY:
        {{
            "is_hallucination": <true/false>,
            "reason": "<string>",
            "score": <0.0 - 1.0>
        }}
        """

        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict factual evaluation judge. JSON output only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content)

        except Exception as e:
            return {
                "is_hallucination": False,
                "reason": f"LLM failure: {e}",
                "score": 0.5
            }

    # ---------------- Run Pipeline ---------------- #
    def run(self, chat: Dict, vectors: Dict):
        turns = chat.get("conversation_turns", [])
        if not turns:
            return {"error": "No conversation_turns found"}

        last_user = next((t for t in reversed(turns) if t["role"] == "User"), None)
        last_ai = next((t for t in reversed(turns) if t["role"] == "AI/Chatbot"), None)

        if not last_user or not last_ai:
            return {"error": "Missing user/AI turns"}

        context_chunks = [
            v.get("text", "") 
            for v in vectors.get("data", {}).get("vector_data", [])
            if v.get("text")
        ]

        context_text = "\n".join(context_chunks)[:4000]

        tier1 = self._tier_1(last_user["message"], last_ai["message"], context_chunks)
        tier2 = self._tier_2(context_text, last_ai["message"])

        return {
            "inputs": {
                "user_query": last_user["message"],
                "ai_response": last_ai["message"][:150] + "..."
            },
            "metrics": {
                "latency_ms": self.metrics.calculate_latency(last_user["created_at"], last_ai["created_at"]),
                "cost_usd": self.metrics.calculate_cost(last_user["message"] + context_text, last_ai["message"]),
                "relevance": round(tier1["relevance"], 3),
                "grounding": round(tier1["grounding"], 3),
                "factual_accuracy": tier2["score"],
                "hallucination_detected": tier2["is_hallucination"],
                "hallucination_reason": tier2["reason"]
            }
        }
