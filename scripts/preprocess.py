from sklearn.datasets import fetch_20newsgroups
import pandas as pd
print("Loading 20 Newsgroups...")
dataset = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),  # Remove email noise
    random_state=42
)
df = pd.DataFrame({
    "text": dataset.data,
    "target": dataset.target,
    "target_names": [dataset.target_names[t] for t in dataset.target]
})
df["word_count"] = df["text"].str.split().str.len()
df = df[df["word_count"] > 50].reset_index(drop=True)  # Filter short docs
df.to_pickle("data/processed_corpus.pkl")
print(f"Processed {len(df)} documents")

