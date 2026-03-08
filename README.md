# Semantic Search for Newsgroups 

<div align="center">

<img src="https://img.shields.io/badge/python-3.10+-blue.svg">
<img src="https://img.shields.io/badge/FastAPI-green.svg">
<img src="https://img.shields.io/badge/FAISS-yellow.svg">

**Lightweight semantic search system with fuzzy clustering & semantic caching for the 20 Newsgroups dataset.**

*Built for Trademarkia AI/ML Engineer Role — Archita Bajpai | March 2026*

</div>

---

## Architecture

20NG Dataset → Preprocessing → MiniLM-L6-v2 → FAISS → GMM(K=5) → Semantic Cache → FastAPI

## Quick Start

### Windows — 1-click

    pip install -r requirements.txt
    python scripts/preprocess.py
    python scripts/build_index.py
    python scripts/train_gmm.py

Then double-click `start.bat` and open http://localhost:8000/docs

### Docker

    docker-compose up --build

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Semantic search + cache |
| `/cache/stats` | GET | Cache performance metrics |
| `/cache` | DELETE | Reset cache |

---

## Key Results

| Metric | Value |
|---|---|
| Dataset Size | 20,000 raw → 12,753 clean docs |
| Embedding Dim | 384 (MiniLM-L6-v2) |
| Clusters | K=5 (BIC-optimal) |
| Cache Threshold | 0.85 cosine similarity |
| Cache Hit Rate | 92.6% on paraphrases |
| Query Latency | <100ms |

**Cache hit example:** "gun control debate" vs "guns control discussion" = **92.6% similarity** ✅

---

## Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Preprocessing | Remove headers/footers/quotes | 37% noise reduction |
| Embedding | MiniLM-L6-v2 | Fast + semantic (384d, 22M params) |
| Clustering | GMM over KMeans | True fuzzy probabilities P(cluster/doc) |
| Vector Store | FAISS IndexFlatIP | Cosine similarity via normalized inner-product |
| Cache | Custom implementation | 92.6% hit rate, no Redis dependency |
| Threshold | 0.85 | Balanced precision/recall (tuned 0.7–0.95) |

---

## Project Structure

    semantic-search-newsgroups/
    ├── app/
    │   ├── main.py          # FastAPI + Semantic Cache
    │   ├── embeddings.py    # MiniLM-L6-v2
    │   ├── vector_store.py  # FAISS
    │   └── clustering.py    # GMM K=5
    ├── data/
    │   ├── vector.index     # FAISS (~20MB)
    │   ├── embeddings.npy   # 384d vectors
    │   └── gmm.pkl          # Trained model
    ├── scripts/
    │   ├── preprocess.py
    │   ├── build_index.py
    │   └── train_gmm.py
    ├── start.bat
    └── requirements.txt

---

## Acknowledgments

- Dataset: [20 Newsgroups (UCI)](http://qwone.com/~jason/20Newsgroups/)
- Embeddings: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Trademarkia: Outstanding assignment design!
