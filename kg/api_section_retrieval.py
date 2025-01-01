"""
FastAPI endpoint for section-level LKG retrieval.

Usage:
    uvicorn api_section_retrieval:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from retrieve import LKGRetriever

KG_DIR = os.getenv("KG_DATA_DIR", "./data")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

retriever: LKGRetriever | None = None
encoder: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, encoder
    retriever = LKGRetriever(KG_DIR)
    retriever.load()
    encoder = SentenceTransformer(EMBED_MODEL)
    yield


app = FastAPI(title="LKG Section Retrieval", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 300


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Retrieve top-K sections from the LKG via BERT + PPR."""
    q_emb = encoder.encode(req.query, normalize_embeddings=True)
    if req.top_k != retriever.top_k:
        retriever.top_k = req.top_k
    results = retriever.retrieve(np.array(q_emb, dtype=np.float32))
    return {"query": req.query, "count": len(results), "results": results}


@app.get("/health")
def health():
    return {"status": "ok",
            "sections_loaded": len(retriever._sections) if retriever else 0}
