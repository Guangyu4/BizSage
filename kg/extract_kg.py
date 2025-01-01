"""
Section-level Knowledge Graph extraction via LLM.

Given a parsed academic paper (split into sections), this script
calls an LLM API to extract structured KG triples from each section:

    (concept, relation, concept)

Triples are stored as JSON-Lines, one file per paper.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# --------------- LLM KG extraction prompt ---------------

EXTRACT_PROMPT = """\
Extract all knowledge graph triples from the following academic paper section.
Return a JSON array of objects with keys: "head", "relation", "tail".

Section label: {section_label}
---
{text}
---
Output ONLY valid JSON."""


def call_api(paper_id: str, section_label: str, text: str,
             api_url: str, api_key: str) -> list[dict]:
    """Call LLM API to extract KG triples from a single section."""
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("KG_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user",
                       "content": EXTRACT_PROMPT.format(
                           section_label=section_label, text=text[:12000])}],
        "temperature": 0.0,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    # Parse JSON from response (handle markdown fences)
    raw = re.sub(r"^```json?\s*|```\s*$", "", raw.strip())
    return json.loads(raw)


def process_paper(paper_path: str, out_dir: str,
                  api_url: str, api_key: str) -> int:
    """Extract KG from all sections of a single paper."""
    with open(paper_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    paper_id = paper.get("paper_id", os.path.basename(paper_path))
    sections = paper.get("sections", [])
    all_triples = []

    for sec in sections:
        label = sec.get("label", "unknown")
        text = sec.get("text", "")
        if len(text.strip()) < 50:
            continue
        triples = call_api(paper_id, label, text, api_url, api_key)
        for t in triples:
            t["paper_id"] = paper_id
            t["section"] = label
        all_triples.extend(triples)

    out_path = os.path.join(out_dir, f"{paper_id}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for t in all_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    return len(all_triples)


def main():
    parser = argparse.ArgumentParser(description="Extract section-level KG triples")
    parser.add_argument("--input-dir", required=True, help="Dir of parsed paper JSONs")
    parser.add_argument("--output-dir", required=True, help="Output dir for KG triples")
    parser.add_argument("--api-url", default=os.getenv("KG_API_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("KG_API_KEY", ""))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Set KG_API_KEY env var or use --api-key.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    papers = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
              if f.endswith(".json")]

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_paper, p, args.output_dir,
                            args.api_url, args.api_key): p for p in papers}
        for fut in as_completed(futs):
            total += fut.result()
    print(f"Extracted {total} triples from {len(papers)} papers.")


if __name__ == "__main__":
    main()
