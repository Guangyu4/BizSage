"""
Reviewer — scores the draft on multiple quality dimensions and
diagnoses which agent is responsible when quality falls short.
"""

import logging
from core.types import TaskSpec, Draft
from core.config import Config

log = logging.getLogger(__name__)

# Dimension names per task type
DIMENSIONS = {
    "survey": ["Relevance", "Grounding", "Coverage", "Synthesis"],
    "idea_formulation": ["Relevance", "Grounding", "Novelty", "Feasibility"],
    "research_plan": ["Relevance", "Grounding", "Rigor", "Feasibility"],
}


class Reviewer:
    """Score draft quality and diagnose failure agents."""

    def __init__(self, config: Config):
        self.config = config

    def review(self, task_spec: TaskSpec, draft: Draft) -> dict:
        """Score the draft and decide accept/revise.

        Returns:
            {
                "scores": {dim: int, ...},    # 1-5 per dimension
                "total": int,                  # sum of scores
                "accepted": bool,
                "diagnosed_agent": str | None, # agent to blame if rejected
                "feedback": str,               # actionable feedback for retry
            }
        """
        dims = DIMENSIONS.get(task_spec.task_type, DIMENSIONS["survey"])

        # LLM call with reviewer prompt (see prompts/reviewer.py)
        scores = self._score(task_spec, draft, dims)

        total = sum(scores.values())
        min_score = min(scores.values())
        accepted = (total >= self.config.pass_total and
                    min_score >= self.config.pass_min)

        result = {"scores": scores, "total": total, "accepted": accepted,
                  "diagnosed_agent": None, "feedback": ""}

        if not accepted:
            diag = self._diagnose(task_spec, draft, scores)
            result["diagnosed_agent"] = diag.get("agent")
            result["feedback"] = diag.get("feedback", "")

        return result

    def _score(self, task_spec, draft, dims):
        """Call LLM to score each dimension 1-5."""
        raise NotImplementedError("See full implementation")

    def _diagnose(self, task_spec, draft, scores):
        """Identify responsible agent and generate feedback."""
        raise NotImplementedError("See full implementation")
