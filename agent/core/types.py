"""Core data types for the multi-agent pipeline."""

from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    """Parsed task specification from the Planner."""
    query: str
    task_type: str  # "survey" | "idea_formulation" | "research_plan"
    interpreted_intent: str = ""


@dataclass
class Paper:
    """A retrieved paper with metadata."""
    paper_id: str
    title: str
    abstract: str = ""
    domain: str = ""
    venue: str = ""
    relevance_score: float = 0.0


@dataclass
class RetrievalResult:
    """Result of a KG retrieval call."""
    sections: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    papers: list = field(default_factory=list)


@dataclass
class Assessment:
    """Assessor output: sections classified by usefulness."""
    useful: list = field(default_factory=list)
    uncertain: list = field(default_factory=list)
    noise: list = field(default_factory=list)
    sufficient: bool = False


@dataclass
class Draft:
    """A generated draft with metadata."""
    content: str = ""
    citations: list = field(default_factory=list)
    iteration: int = 1
