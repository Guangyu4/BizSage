# BizSage: A Self-Evolving Multi-Agent Framework for Business Research with Efficient Knowledge Retrieval

> **Anonymous submission to EMNLP 2025**

## Overview

BizSage is a self-evolving multi-agent framework for automated research assistance in economics and business. It combines:

1. **Lateral Knowledge Graph (LKG)** — a corpus-level graph unifying section-level KGs from 134K+ papers across economics, finance, operations management, and statistics, with **Personalized PageRank (PPR)** retrieval to surface structurally important sections.

2. **Multi-Agent Generation** — seven specialized agents (Planner, Retriever, Assessor, Presenter, Discussant, Chair, Reviewer) collaborating under a Manager to produce grounded research outputs for three progressive task types: *survey*, *idea formulation*, and *research plan*.

3. **Meta-Review Self-Evolution** — a mechanism that distills recurring failure patterns from evaluation traces into reusable quality strategies, enabling continuous adaptation to domain-specific standards.

## Repository Structure

```
BizSage-Anonymous/
├── kg/                           # LKG construction & retrieval
│   ├── extract_kg.py             # Section-level KG extraction via LLM
│   ├── retrieve.py               # Core retrieval: BERT + PPR (Eqs. 2–3)
│   ├── api_section_retrieval.py  # FastAPI serving endpoint
│   └── requirements.txt
│
├── agent/                        # Multi-agent system (key modules)
│   ├── agents/
│   │   ├── kg_retriever.py       # LKG+PPR retriever agent
│   │   ├── analyst.py            # Presenter → Discussant → Chair
│   │   └── reviewer.py           # Quality scoring & diagnosis
│   ├── core/                     # Config & data types
│   ├── prompts/                  # Prompt templates
│   ├── manager.py                # Orchestration + iterative refinement
│   ├── meta_review.py            # Self-evolution (Eqs. 6–7)
│   └── requirements.txt
│
├── benchmark/                    # 300 evaluation queries
│   ├── survey/                   # 100 survey queries
│   ├── idea_formulation/         # 100 idea formulation queries
│   └── research_plan/            # 100 research plan queries
│
├── eval_results/                 # Aggregated scores (3 CSV tables)
│
├── scripts/                      # Evaluation scripts
│   ├── evaluate_judge_v2.py      # LLM-as-judge evaluation
│   └── aggregate_scores.py       # Score aggregation
│
├── LICENSE                       # CC BY 4.0
└── README.md
```

## Quick Start

### 1. KG Retrieval API

```bash
cd kg
pip install -r requirements.txt

# Start section-level retrieval API (requires KG data in ./data)
uvicorn api_section_retrieval:app --host 0.0.0.0 --port 8000
```

### 2. Multi-Agent System

```bash
cd agent
pip install -r requirements.txt

# Set environment variables (never hardcode keys)
export LLM_API_KEY="your-api-key"
export LLM_API_URL="https://api.openai.com/v1/chat/completions"
export LLM_MODEL="gpt-4o"
export KG_API_URL="http://localhost:8000/retrieve"
```

### 3. Benchmark Evaluation

Each query in `benchmark/` is a JSON file with `paper_id`, `domain`, `journal`, `title`, and `query` fields. Evaluation uses GPT-4o as an LLM judge with task-specific rubrics (see paper Section 4.3).

## Key Results

- BizSage ranks **first on the majority of metrics** across three backbone LLMs (GPT-5.5, Gemini-3-Flash, DeepSeek-V4-Flash) and three task types.
- **Zero hallucinated citations** across 894 verified entries.
- Self-evolution concentrates corrections on the **weakest quality dimension** rather than lifting all dimensions uniformly.
- Retrieval modules and collaborative agents address **largely independent failure modes**, making both indispensable.

## Data & Corpus

The LKG is built from **134,234 papers** across **73 peer-reviewed journals** in four domains (economics, finance, operations management, statistics). The full KG data (nodes, edges, embeddings) and source corpus will be released upon acceptance.

## License

This project is licensed under **CC BY 4.0**. See [LICENSE](LICENSE) for details.
