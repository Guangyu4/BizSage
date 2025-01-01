"""
KG Retriever — queries the Lateral Knowledge Graph API and applies
Personalized PageRank (PPR) re-ranking to surface structurally
important sections across documents.
"""

import logging
import time
from collections import defaultdict

import requests

from core.types import TaskSpec, Paper, RetrievalResult

log = logging.getLogger(__name__)

MAX_RETRIES = 3


class KGRetriever:
    """Retrieve sections from the corpus-level LKG via REST API."""

    def __init__(self, api_url: str = "http://localhost:8001/retrieve",
                 top_k: int = 300):
        self.api_url = api_url
        self.top_k = top_k

    def _call_api(self, query: str) -> list[dict]:
        """POST to KG retrieval API with retry on transient errors."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    json={"query": query, "top_k": self.top_k},
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json().get("results", [])
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == MAX_RETRIES:
                    raise
                log.warning("Attempt %d failed: %s", attempt, e)
                time.sleep(2 ** attempt)
        return []

    def retrieve(self, task_spec: TaskSpec, iteration: int = 1,
                 previous_section_keys: set | None = None,
                 feedback: str = "", hyde_passage: str = "") -> RetrievalResult:
        """Build query from task spec and retrieve relevant sections.

        The query is enriched with the interpreted intent, any reviewer
        feedback from a previous iteration, and an optional HyDE passage
        to nudge embeddings toward the target neighbourhood.
        """
        previous_section_keys = previous_section_keys or set()

        query = task_spec.query
        if task_spec.interpreted_intent:
            query += " | " + task_spec.interpreted_intent
        if feedback:
            query += " | " + feedback
        if hyde_passage:
            query += " | " + hyde_passage

        raw = self._call_api(query)

        # Deduplicate by section key
        sections = [s for s in raw
                    if (s["paper_id"], s.get("section", "")) not in previous_section_keys]

        # Extract matched KG keywords
        keywords, seen = [], set()
        for sec in sections:
            for node in sec.get("matched_nodes", []):
                name = node.get("name", "")
                if name and name not in seen:
                    keywords.append(name)
                    seen.add(name)

        # Group into Paper objects
        grouped: dict[str, list[dict]] = defaultdict(list)
        for sec in sections:
            grouped[sec["paper_id"]].append(sec)

        papers = []
        for pid, secs in grouped.items():
            best = max(secs, key=lambda s: s.get("rank_score", 0))
            parts = pid.split("_", 2)
            papers.append(Paper(
                paper_id=pid, title=pid,
                abstract=best.get("text", "")[:2000],
                domain=parts[1] if len(parts) >= 2 else "",
                venue=parts[2] if len(parts) >= 3 else "",
                relevance_score=best.get("rank_score", 0.0),
            ))
        papers.sort(key=lambda p: -p.relevance_score)

        return RetrievalResult(sections=sections, keywords=keywords, papers=papers)
