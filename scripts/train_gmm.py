import sys
import os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('app'))

import numpy as np
from sklearn.mixture import GaussianMixture
import pickle

embeddings = np.load("data/embeddings.npy")
print("Finding optimal clusters...")
bic_scores = {}
for k in range(5, 25):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
    gmm.fit(embeddings[:1000]) 
    log_likelihood = gmm.score(embeddings[:1000])
    n_samples, n_features = embeddings[:1000].shape
    n_params = k * (n_features * (n_features + 3) / 2 + 1)
    bic = -2 * log_likelihood * n_samples + np.log(n_samples) * n_params
    bic_scores[k] = bic
    print(f"K={k}: BIC={bic:.0f}")

best_k = min(bic_scores, key=bic_scores.get)
print(f"Best K: {best_k}")

gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
gmm.fit(embeddings)
cluster_probs = gmm.predict_proba(embeddings)

with open("data/gmm.pkl", "wb") as f:
    pickle.dump(gmm, f)
np.save("data/cluster_probs.npy", cluster_probs)
print("GMM trained")
