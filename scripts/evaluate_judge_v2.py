"""
LLM-as-Judge evaluation — scores generated outputs using GPT-4o
on task-specific rubrics (4 dimensions × 5-point scale).

Usage:
    python evaluate_judge_v2.py --input-dir results/ --output-dir eval_out/
"""

import argparse
import json
import os
import sys

import requests

DIMENSIONS = {
    "survey": ["Relevance", "Grounding", "Coverage", "Synthesis"],
    "idea_formulation": ["Relevance", "Grounding", "Novelty", "Feasibility"],
    "research_plan": ["Relevance", "Grounding", "Rigor", "Feasibility"],
}

JUDGE_PROMPT = """\
You are an expert academic reviewer. Score the following {task_type} output \
on each dimension (1–5). Output JSON: {{"dim": score, ...}}

Dimensions:
{dim_desc}

Query: {query}
Output to evaluate:
{output}
"""


def judge_one(task_type: str, query: str, output: str,
              api_url: str, api_key: str, model: str) -> dict:
    """Score a single output via LLM judge."""
    dims = DIMENSIONS[task_type]
    dim_desc = "\n".join(f"- {d}: 1(worst)–5(best)" for d in dims)
    prompt = JUDGE_PROMPT.format(task_type=task_type, query=query,
                                 output=output[:8000], dim_desc=dim_desc)
    resp = requests.post(api_url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return json.loads(raw.strip().strip("`").removeprefix("json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--api-url", default=os.getenv("LLM_API_URL", ""))
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    p.add_argument("--model", default="gpt-4o")
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: Set LLM_API_KEY.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for task_type in DIMENSIONS:
        task_dir = os.path.join(args.input_dir, task_type)
        if not os.path.isdir(task_dir):
            continue
        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(task_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scores = judge_one(task_type, data["query"],
                               data.get("output", ""),
                               args.api_url, args.api_key, args.model)
            out = {**data, "scores": scores, "total": sum(scores.values())}
            out_path = os.path.join(args.output_dir, task_type, fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"{fname}: {scores} (total={sum(scores.values())})")


if __name__ == "__main__":
    main()
