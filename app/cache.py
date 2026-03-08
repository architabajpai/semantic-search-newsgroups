import numpy as np
from typing import Dict, Optional, List
from sklearn.metrics.pairwise import cosine_similarity
from app.clustering import get_gmm

class SemanticCache:
    def __init__(self, threshold: float = 0.85):
        self.entries: List[Dict] = []
        self.hit_count = self.miss_count = 0
        self.threshold = threshold
        self.gmm = get_gmm()
        self.cluster_entries: Dict[int, List[int]] = {}

    def lookup(self, query_emb: np.ndarray, query_text: str) -> Optional[Dict]:
        try:
            probs = self.gmm.predict_proba(query_emb.reshape(1, -1))[0]
            dom_cluster = np.argmax(probs)
            
            candidates = self.cluster_entries.get(dom_cluster, list(range(len(self.entries))))
            best_sim = 0.0
            best_entry = None
            
            query_emb_2d = query_emb.reshape(1, -1)
            for idx in candidates:
                entry = self.entries[idx]
                sim = cosine_similarity(query_emb_2d, [entry["embedding"]])[0][0]
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
            
            if best_sim > self.threshold:
                self.hit_count += 1
                return {"entry": best_entry, "sim": best_sim}
            self.miss_count += 1
            return None
        except Exception:
            self.miss_count += 1
            return None

    def store(self, query_text: str, query_emb: np.ndarray, result: str, cluster_id: int):
        entry = {
            "query": query_text, 
            "embedding": query_emb.copy(), 
            "result": result, 
            "cluster": cluster_id
        }
        self.entries.append(entry)
        if cluster_id not in self.cluster_entries:
            self.cluster_entries[cluster_id] = []
        self.cluster_entries[cluster_id].append(len(self.entries) - 1)

    def stats(self) -> Dict:
        total = self.hit_count + self.miss_count
        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total, 3) if total else 0
        }

    def clear(self):
        self.entries = []
        self.cluster_entries = {}
        self.hit_count = self.miss_count = 0
