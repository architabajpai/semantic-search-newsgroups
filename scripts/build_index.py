import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('app'))

import pandas as pd
import numpy as np
import faiss
from app.embeddings import get_embedding_model

print("Loading corpus...")
df = pd.read_pickle("data/processed_corpus.pkl")
model = get_embedding_model()
print("Generating embeddings...")
embeddings = model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)

# Normalize for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype("float32"))

faiss.write_index(index, "data/vector.index")
np.save("data/embeddings.npy", embeddings)
df.to_pickle("data/corpus.pkl")
print(f"Indexed {len(embeddings)} docs (dim={dim})")
