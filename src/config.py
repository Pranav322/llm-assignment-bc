import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Thresholds
    RELEVANCE_THRESHOLD = 0.5
    GROUNDING_THRESHOLD = 0.5
