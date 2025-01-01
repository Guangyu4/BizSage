"""
Analyst — implements the Academic Research Meeting (Presenter → Discussant → Chair).

The Presenter generates an initial draft from retrieved sections.
The Discussant critiques it on multiple quality dimensions.
The Chair integrates the draft and critique into a revised output.
"""

import logging
from core.types import TaskSpec, RetrievalResult, Assessment, Draft
from core.config import Config

log = logging.getLogger(__name__)


class Analyst:
    """Single-round academic round-table: Presenter → Discussant → Chair."""

    def __init__(self, config: Config):
        self.config = config

    def run(self, task_spec: TaskSpec, retrieval: RetrievalResult,
            assessment: Assessment, reviewer_feedback: str = "") -> tuple[Draft, dict]:
        """Execute the three-step writing pipeline.

        1. Presenter: generates draft y_0 from (task_spec, sections)
        2. Discussant: examines draft on quality dimensions, raises issues R
        3. Chair: integrates draft and critique into revised output y
        """
        sections_pool = list(assessment.useful)
        if len(sections_pool) < self.config.min_useful_sections:
            shortfall = self.config.min_useful_sections - len(sections_pool)
            sections_pool.extend(assessment.uncertain[:shortfall])

        # --- Step 1: Presenter ---
        presenter_output = self._call_presenter(task_spec, sections_pool,
                                                reviewer_feedback)

        # --- Step 2: Discussant ---
        critique = self._call_discussant(task_spec, presenter_output,
                                         sections_pool)

        # --- Step 3: Chair ---
        final_output = self._call_chair(task_spec, presenter_output,
                                        critique, sections_pool)

        draft = Draft(content=final_output, iteration=1)
        trace = {
            "presenter": presenter_output[:500],
            "critique": critique[:500],
            "final": final_output[:500],
        }
        return draft, trace

    def _call_presenter(self, task_spec, sections, feedback):
        """Generate initial draft with task-specific template."""
        # LLM call with presenter prompt (see prompts/analyst.py)
        raise NotImplementedError("See full implementation")

    def _call_discussant(self, task_spec, draft, sections):
        """Critique draft on quality dimensions."""
        raise NotImplementedError("See full implementation")

    def _call_chair(self, task_spec, draft, critique, sections):
        """Synthesize final output from draft + critique."""
        raise NotImplementedError("See full implementation")
