import os
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from .database import engine, get_db, Base
from . import models, schemas
from typing import List, Optional
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from functools import lru_cache
import openai

# Initialize logging for auditing
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
handler = logging.FileHandler("audit.log")
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
audit_logger.addHandler(handler)

# Create DB tables
Base.metadata.create_all(bind=engine)

# Load embedding model (small local model)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Prometheus metrics
global_requests_total = Counter("kb_requests_total", "Total number of API requests", ["endpoint"])
search_latency = Histogram("kb_search_latency_seconds", "Latency of search endpoint", ["endpoint"])

app = FastAPI(title="Enterprise Knowledge Base Assistant")

def log_audit(user: str, role: Optional[str], query: str, result_count: int):
    audit_logger.info(f"User={user} Role={role} Query=\"{query}\" Results={result_count}")

def compute_embedding(text: str) -> List[float]:
    return embed_model.encode(text).tolist()

@lru_cache(maxsize=128)
def cached_search(query_embedding: tuple, department: Optional[str], role: Optional[str], tags: Optional[tuple]):
    # Simple linear scan for demo purposes
    db = Session(bind=engine)
    docs = db.query(models.Document).all()
    results = []
    q_vec = np.array(query_embedding)
    for doc in docs:
        # Role and department filtering
        if role and doc.role and doc.role != role:
            continue
        if department and doc.department and doc.department != department:
            continue
        if tags:
            doc_tags = set(doc.tags or [])
            if not set(tags).issubset(doc_tags):
                continue
        if doc.embedding is None:
            continue
        d_vec = np.array(doc.embedding)
        # Cosine similarity
        score = float(np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec)))
        results.append((score, doc))
    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    db.close()
    return results

@app.get("/search", response_model=List[schemas.SearchResult])
@search_latency.labels(endpoint="/search").time()
def search(q: str = Query(..., description="Search query"),
           department: Optional[str] = Query(None),
           role: Optional[str] = Query(None),
           tags: Optional[str] = Query(None),
           user: str = Query("demo_user")):
    global_requests_total.labels(endpoint="/search").inc()
    tag_list = [t.strip() for t in tags.split(',')] if tags else []
    query_emb = tuple(compute_embedding(q))
    cached_res = cached_search(query_emb, department, role, tuple(tag_list) if tag_list else None)
    # Build response (top 10)
    response = []
    for score, doc in cached_res[:10]:
        snippet = doc.content[:200] + ("..." if len(doc.content) > 200 else "")
        response.append(schemas.SearchResult(
            id=doc.id,
            title=doc.title,
            snippet=snippet,
            score=score,
            department=doc.department,
            role=doc.role,
            tags=doc.tags or []
        ))
    log_audit(user, role, q, len(response))
    return response

@app.post("/index", response_model=schemas.DocumentOut)
def index_document(doc: schemas.DocumentCreate, db: Session = Depends(get_db), user: str = Query("demo_user")):
    global_requests_total.labels(endpoint="/index").inc()
    # Compute embedding
    embedding = compute_embedding(doc.content)
    # Check if document with same title exists -> version bump
    existing = db.query(models.Document).filter(models.Document.title == doc.title).first()
    if existing:
        existing.content = doc.content
        existing.department = doc.department
        existing.role = doc.role
        existing.tags = doc.tags
        existing.version += 1
        existing.updated_at = datetime.utcnow()
        existing.embedding = embedding
        db.add(existing)
        db.commit()
        db.refresh(existing)
        result = existing
    else:
        new_doc = models.Document(
            title=doc.title,
            content=doc.content,
            department=doc.department,
            role=doc.role,
            tags=doc.tags,
            version=doc.version or 1,
            embedding=embedding,
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        result = new_doc
    # Invalidate cache (simple approach: clear whole cache)
    cached_search.cache_clear()
    audit_logger.info(f"User={user} Indexed document id={result.id} title=\"{result.title}\"")
    return schemas.DocumentOut(
        id=result.id,
        title=result.title,
        department=result.department,
        role=result.role,
        tags=result.tags or [],
        version=result.version,
        created_at=result.created_at.isoformat(),
        updated_at=result.updated_at.isoformat() if result.updated_at else None,
    )

@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
