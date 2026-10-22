import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Settings:
    cohere_api_key: str
    faiss_db_path: str = "vector_db/faiss_policy_index"
    embedding_model: str = "embed-v4.0"
    chat_model: str = "command-a-03-2025"
    temperature: float = 0.2
    top_k: int = 5
    fetch_k: int = 20


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found. Add it to your .env file.")
    return Settings(cohere_api_key=api_key)
