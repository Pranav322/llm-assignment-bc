import tiktoken
from datetime import datetime

class MetricsCalculator:
    def __init__(self):
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None

    def calculate_latency(self, t_user: str, t_ai: str) -> float:
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
            t1 = datetime.strptime(t_user.split('Z')[0], fmt)
            t2 = datetime.strptime(t_ai.split('Z')[0], fmt)
            return (t2 - t1).total_seconds() * 1000
        except:
            return 0.0

    def calculate_cost(self, prompt: str, response: str) -> float:
        if not self.tokenizer: return 0.0

        in_tokens = len(self.tokenizer.encode(prompt))
        out_tokens = len(self.tokenizer.encode(response))

        # GPT-4o-mini pricing
        cost = (in_tokens * 0.15 / 1_000_000) + (out_tokens * 0.60 / 1_000_000)
        return round(cost, 8)

