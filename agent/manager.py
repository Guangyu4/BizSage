"""
Manager — orchestrates the multi-agent pipeline with iterative refinement.

Pipeline per task:
  1. Planner: parse query → TaskSpec
  2. Retriever: KG-PPR retrieval → sections
  3. Assessor: classify sections → useful / uncertain / noise
  4. Analyst: Presenter → Discussant → Chair → Draft
  5. Reviewer: score draft → accept / diagnose + feedback
  6. If rejected: route feedback to diagnosed agent, retry (up to T rounds)
"""

import logging
from core.config import Config
from core.types import TaskSpec, Draft
from agents.kg_retriever import KGRetriever
from agents.analyst import Analyst
from agents.reviewer import Reviewer

log = logging.getLogger(__name__)


class Manager:
    """Self-evolving pipeline manager with iterative quality refinement."""

    def __init__(self, config: Config, retriever: KGRetriever,
                 results_dir: str = "results"):
        self.config = config
        self.retriever = retriever
        self.analyst = Analyst(config)
        self.reviewer = Reviewer(config)
        self.results_dir = results_dir
        self.quality_memory: list[dict] = []  # M_Q: accumulated strategies

    def run(self, query: str, task_type: str | None = None) -> dict:
        """Execute the full pipeline for a single query.

        Implements Eq. (5) from the paper:
            y^(t+1) = Phi(q | f_{a*}^(t)),  t = 1, ..., T
        """
        task_spec = self._plan(query, task_type)

        # Load quality strategies for each agent from M_Q
        self._apply_strategies(task_spec)

        draft = None
        for iteration in range(1, self.config.max_iterations + 1):
            log.info("=== Iteration %d/%d ===", iteration, self.config.max_iterations)

            # Stage 1: Retrieve
            feedback = "" if iteration == 1 else review["feedback"]
            retrieval = self.retriever.retrieve(task_spec, iteration=iteration,
                                                feedback=feedback)

            # Assess retrieved sections
            assessment = self._assess(task_spec, retrieval)

            # Stage 2: Generate (Presenter → Discussant → Chair)
            draft, trace = self.analyst.run(task_spec, retrieval, assessment,
                                            reviewer_feedback=feedback)

            # Review
            review = self.reviewer.review(task_spec, draft)
            log.info("Scores: %s (total=%d, accepted=%s)",
                     review["scores"], review["total"], review["accepted"])

            if review["accepted"]:
                break

            log.info("Rejected → routing to %s", review["diagnosed_agent"])

        return {"output": draft.content if draft else "",
                "task_spec": task_spec, "iterations": iteration}

    def _plan(self, query: str, task_type: str | None) -> TaskSpec:
        """Parse query into a structured task specification."""
        raise NotImplementedError("See full implementation")

    def _assess(self, task_spec, retrieval):
        """Classify sections as useful / uncertain / noise."""
        raise NotImplementedError("See full implementation")

    def _apply_strategies(self, task_spec: TaskSpec):
        """Inject accumulated quality strategies from M_Q into agent prompts."""
        for strategy in self.quality_memory:
            agent_name = strategy.get("agent")
            patch = strategy.get("patch", "")
            log.debug("Applying strategy for %s: %s", agent_name, patch[:80])
