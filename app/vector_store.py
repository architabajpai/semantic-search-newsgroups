import faiss
import numpy as np

_index = None

def get_vector_store():
    global _index
    if _index is None:
        _index = faiss.read_index("data/vector.index")
    return _index
 
