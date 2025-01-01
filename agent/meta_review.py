"""
Meta-Review — system-level self-evolution mechanism.

After processing m tasks, the Manager analyzes accumulated traces to
identify recurring failure patterns and generate reusable quality
strategies. This implements Eqs. (6)–(7) from the paper:

    c_a* = argmax_k  ν(c_k, H_worst)      — select best strategy
    M_Q  ← M_Q ∪ {(a, c_a*)}              — commit to quality memory
"""

import json
import logging
import os

log = logging.getLogger(__name__)


def load_patches(patches_file: str) -> dict:
    """Load accumulated prompt patches (quality strategies) from disk."""
    if not os.path.exists(patches_file):
        return {}
    with open(patches_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_patches(patches_file: str, patches: dict):
    """Persist quality strategies to disk."""
    os.makedirs(os.path.dirname(patches_file), exist_ok=True)
    with open(patches_file, "w", encoding="utf-8") as f:
        json.dump(patches, f, ensure_ascii=False, indent=2)


def run_meta_review(results_dir: str, config, experiment_id: str | None = None):
    """Analyze traces from completed tasks and generate quality strategies.

    Steps:
    1. Collect evaluation traces {H_1, ..., H_m} from results_dir
    2. Identify agents that are repeatedly blamed (diagnosed_agent)
    3. For each failure pattern, generate candidate strategies {c_1,...,c_K}
    4. Select the strategy that most improves the worst historical case
    5. Commit validated strategies to M_Q (prompt_patches.json)
    """
    # Resolve patches file
    if experiment_id:
        patches_path = os.path.join(results_dir, "benchmark", experiment_id,
                                     "prompt_patches.json")
    else:
        patches_path = os.path.join(results_dir, "prompt_patches.json")

    existing = load_patches(patches_path)

    # Step 1: Collect traces
    traces = _collect_traces(results_dir, experiment_id)
    log.info("Collected %d task traces", len(traces))

    # Step 2: Identify failure patterns by agent
    blame_counts = _count_blame(traces)
    log.info("Blame distribution: %s", blame_counts)

    # Step 3-4: Generate and validate strategies per agent
    new_strategies = {}
    for agent_name, count in blame_counts.items():
        if count < 2:
            continue
        worst_trace = _find_worst_trace(traces, agent_name)
        candidates = _generate_candidates(agent_name, worst_trace, config)
        best = _select_best(candidates, worst_trace, config)
        if best:
            new_strategies[agent_name] = best
            log.info("New strategy for %s: %s", agent_name, best[:100])

    # Step 5: Commit
    for agent_name, strategy in new_strategies.items():
        patches = existing.get(agent_name, [])
        patches.append({"patch": strategy, "source": "meta_review"})
        existing[agent_name] = patches

    save_patches(patches_path, existing)
    log.info("Saved %d new strategies to %s", len(new_strategies), patches_path)


def _collect_traces(results_dir, experiment_id):
    """Scan result dirs for reviewer scores and diagnosis traces."""
    raise NotImplementedError("See full implementation")


def _count_blame(traces):
    """Count how often each agent is diagnosed as responsible."""
    raise NotImplementedError("See full implementation")


def _find_worst_trace(traces, agent_name):
    """Find the trace with lowest total score for a given agent."""
    raise NotImplementedError("See full implementation")


def _generate_candidates(agent_name, worst_trace, config):
    """Use LLM to generate K candidate strategy patches."""
    raise NotImplementedError("See full implementation")


def _select_best(candidates, worst_trace, config):
    """Replay worst case with each candidate; pick the one with max gain."""
    raise NotImplementedError("See full implementation")


# --- Utility functions for patch management ---

def clear_patches(patches_path: str):
    """Reset all patches (for baseline experiments)."""
    save_patches(patches_path, {})
    log.info("Cleared all patches at %s", patches_path)


def snapshot_patches(patches_path: str, name: str):
    """Save a named copy of current patches before meta-review."""
    patches = load_patches(patches_path)
    snap_dir = os.path.join(os.path.dirname(patches_path), "patch_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    snap_path = os.path.join(snap_dir, f"{name}.json")
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(patches, f, ensure_ascii=False, indent=2)
    log.info("Snapshot saved: %s", snap_path)


def restore_patches(patches_path: str, name: str) -> bool:
    """Restore patches from a named snapshot."""
    snap_path = os.path.join(os.path.dirname(patches_path),
                              "patch_snapshots", f"{name}.json")
    if not os.path.exists(snap_path):
        log.error("Snapshot not found: %s", snap_path)
        return False
    with open(snap_path, "r", encoding="utf-8") as f:
        patches = json.load(f)
    save_patches(patches_path, patches)
    log.info("Restored patches from %s", snap_path)
    return True
