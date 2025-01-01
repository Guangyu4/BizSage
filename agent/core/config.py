"""Configuration dataclass — all secrets from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Global configuration for LLM and KG retrieval services."""

    # LLM (all from env vars, never hardcoded)
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 16384
    llm_timeout: int = 300

    # Pipeline control
    max_iterations: int = 3
    max_retrieval_retries: int = 3

    # Reviewer thresholds (4 dims × 5 max = 20)
    pass_total: int = 16
    pass_min: int = 3

    # KG retrieval
    kg_api_url: str = "http://localhost:8001/retrieve"
    kg_top_k: int = 300

    # Task-specific max writing sections
    max_sections_survey: int = 50
    max_sections_idea: int = 35
    max_sections_plan: int = 25
    min_useful_sections: int = 5

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_api_url=os.getenv("LLM_API_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            kg_api_url=os.getenv("KG_API_URL", "http://localhost:8001/retrieve"),
        )
