"""
Core retrieval module — BERT similarity + Personalized PageRank (PPR).

This is the key retrieval component of the LKG. Given a user query:
  1. Encode query via Sentence-BERT
  2. Compute cosine similarity against pre-built section embeddings
  3. Build a section-level citation subgraph
  4. Run PPR with teleport biased toward high-similarity seed nodes
  5. Return top-K sections ranked by PPR score

The PPR formulation follows Eq. (2)-(3) in the paper.
"""

import json
import os

import numpy as np


class LKGRetriever:
    """Lateral Knowledge Graph retriever with PPR re-ranking."""

    def __init__(self, kg_dir: str, top_k: int = 300,
                 alpha: float = 0.15, max_iter: int = 50,
                 tol: float = 1e-6):
        """
        Args:
            kg_dir: Directory containing KG node/edge CSVs and embeddings.
            top_k: Number of sections to return.
            alpha: PPR teleport probability (Eq. 3).
            max_iter: Max PPR iterations.
            tol: Convergence tolerance for PPR.
        """
        self.kg_dir = kg_dir
        self.top_k = top_k
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        # Lazy-loaded data
        self._sections = None
        self._embeddings = None
        self._adj = None

    def load(self):
        """Load sections, embeddings, and adjacency from kg_dir."""
        sections_path = os.path.join(self.kg_dir, "sections.json")
        emb_path = os.path.join(self.kg_dir, "section_embeddings.npz")

        with open(sections_path, "r", encoding="utf-8") as f:
            self._sections = json.load(f)

        data = np.load(emb_path)
        self._embeddings = data["embeddings"]  # (N, D)
        self._embeddings /= np.linalg.norm(self._embeddings, axis=1, keepdims=True)

        self._adj = self._build_adjacency()

    def _build_adjacency(self) -> dict:
        """Build adjacency dict from KG edges (section → section links)."""
        edges_path = os.path.join(self.kg_dir, "edges.jsonl")
        adj = {}
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                src, dst = e["source"], e["target"]
                adj.setdefault(src, []).append(dst)
                adj.setdefault(dst, []).append(src)
        return adj

    def retrieve(self, query_embedding: np.ndarray) -> list[dict]:
        """Main retrieval: cosine similarity → PPR → top-K.

        Args:
            query_embedding: (D,) normalized query vector from Sentence-BERT.

        Returns:
            List of dicts with section metadata and rank_score.
        """
        # Step 1: Cosine similarity
        sims = self._embeddings @ query_embedding  # (N,)

        # Step 2: Seed nodes = top candidates by similarity
        seed_k = min(self.top_k * 2, len(sims))
        seed_idx = np.argsort(sims)[-seed_k:][::-1]

        # Step 3: Personalized PageRank
        teleport = np.zeros(len(sims))
        teleport[seed_idx] = sims[seed_idx]
        teleport /= teleport.sum() + 1e-12

        ppr = self._run_ppr(teleport)

        # Step 4: Combine similarity and PPR (Eq. 3)
        combined = 0.5 * sims + 0.5 * ppr
        top_idx = np.argsort(combined)[-self.top_k:][::-1]

        results = []
        for i in top_idx:
            sec = dict(self._sections[i])
            sec["rank_score"] = float(combined[i])
            sec["sim_score"] = float(sims[i])
            sec["ppr_score"] = float(ppr[i])
            results.append(sec)
        return results

    def _run_ppr(self, teleport: np.ndarray) -> np.ndarray:
        """Power iteration for Personalized PageRank.

        π^(t+1) = α · p + (1 - α) · A^T · π^(t)
        """
        n = len(teleport)
        pi = teleport.copy()

        for _ in range(self.max_iter):
            pi_new = self.alpha * teleport
            for src, neighbors in self._adj.items():
                src_i = int(src)
                if src_i >= n:
                    continue
                out_deg = len(neighbors)
                for dst in neighbors:
                    dst_i = int(dst)
                    if dst_i < n:
                        pi_new[dst_i] += (1 - self.alpha) * pi[src_i] / out_deg
            if np.abs(pi_new - pi).sum() < self.tol:
                break
            pi = pi_new

        return pi
