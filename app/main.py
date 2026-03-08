from fastapi import FastAPI
from app.models import QueryRequest
from app.embeddings import get_embedding_model
from app.vector_store import get_vector_store
from app.clustering import get_gmm
import pandas as pd
import numpy as np

model = get_embedding_model()
index = get_vector_store()
gmm = get_gmm()
corpus = pd.read_pickle("data/corpus.pkl")

cache_entries = []
CACHE_THRESHOLD = 0.85

app = FastAPI(
    title="Semantic Search",
    description="By Archita Bajpai"
)

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    global cache_entries

    q_emb = model.encode([req.query])[0]
    
    cache_hit = None
    best_score = 0
    
    for entry in cache_entries:
        score = np.dot(q_emb / np.linalg.norm(q_emb), 
                      entry["embedding"] / np.linalg.norm(entry["embedding"]))
        if score > best_score and score > CACHE_THRESHOLD:
            best_score = score
            cache_hit = entry
    
    if cache_hit:
        return {
            "query": req.query,
            "cache_hit": True,
            "matched_query": cache_hit["query"],
            "similarity_score": round(float(best_score), 3),
            "result": cache_hit["result"],
            "dominant_cluster": cache_hit["cluster"]
        }
    scores, indices = index.search(q_emb.reshape(1, -1).astype("float32"), 5)
    top_docs = []
    
    for i, score in zip(indices[0], scores[0]):
        if score > 0.5 and i < len(corpus):
            top_docs.append(corpus.iloc[i]["text"][:200] + "...")
    
    result = "Top matches:\n" + "\n\n".join(top_docs) if top_docs else "No relevant matches"
    
    probs = gmm.predict_proba(q_emb.reshape(1, -1))[0]
    cluster_id = int(np.argmax(probs))
    
    cache_entries.append({
        "query": req.query,
        "embedding": q_emb.copy(),
        "result": result,
        "cluster": cluster_id
    })
    
    return {
        "query": req.query,
        "cache_hit": False,
        "result": result,
        "dominant_cluster": cluster_id
    }

@app.get("/cache/stats")
async def stats():
    total_queries = len(cache_entries)
    hits = sum(1 for e in cache_entries if True)
    return {
        "total_entries": len(cache_entries),
        "hit_count": 0,  
        "miss_count": total_queries,
        "hit_rate": 0.0
    }

@app.delete("/cache")
async def clear_cache():
    global cache_entries
    cache_entries = []
    return {"message": "Cache cleared"}
