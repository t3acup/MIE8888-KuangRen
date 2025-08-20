import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Load embeddings
file_name = "embeddings\java_InCoder_embeddings.pkl"
df = pd.read_pickle(file_name)

# Convert embeddings to NumPy
embeddings = np.vstack(df["embedding"].values)  # each is a 768‑D vector
labels = df["cluster"].tolist()

# Run t-SNE
reducer = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = reducer.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1],
                hue=labels, palette="tab10", s=50, alpha=0.8)

plt.title("CodeBERT Embeddings via t‑SNE")
plt.xlabel("t‑SNE Dim 1")
plt.ylabel("t‑SNE Dim 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
