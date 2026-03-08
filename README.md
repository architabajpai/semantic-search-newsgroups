# Semantic Search for newsgroups: By Archita Bajpai

## Live Demo
Double-click `start.bat` → Open `http://localhost:8000/docs`

## Architecture
20NG Dataset → Preprocessing → MiniLM-L6-v2 → FAISS → GMM(K=5) → Semantic Cache → FastAPI

## Key Results
- **Cache Hit**: "gun control debate" vs "guns control discussion" = **92.6% similarity**
- **Fuzzy Clustering**: BIC-optimal K=5 clusters (not arbitrary 20 categories)
- **Vector Search**: FAISS returns relevant gun control debates instantly

## API Endpoints
- `POST /query {"query": "space nasa"}` → Semantic search + cache
- `GET /cache/stats` → Cache performance  
- `DELETE /cache` → Reset cache

## Design Decisions
- Removed headers/footers/quotes → 12,753 clean documents
- MiniLM-L6-v2: Fast + semantic (384d)
- GMM > KMeans: True fuzzy probabilities P(cluster|doc)
- Cache threshold 0.85: Balanced precision/recall
