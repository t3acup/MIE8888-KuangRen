import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# === Load Data ===
df = pd.read_pickle("embeddings\java_InCoder_embeddings.pkl")
embeddings = np.vstack(df["embedding"].values)

# === Encode ground-truth labels if needed ===
if not np.issubdtype(df["cluster"].dtype, np.integer):
    le = LabelEncoder()
    df["cluster"] = le.fit_transform(df["cluster"])

y_true = df["cluster"].values

# === KMeans Clustering ===
k = 10
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(embeddings)
df["kmeans_cluster"] = y_pred

# === Map predicted clusters to true labels ===
def map_clusters(y_true, y_pred):
    labels = np.unique(y_pred)
    new_pred = np.zeros_like(y_pred)
    for label in labels:
        mask = y_pred == label
        if np.any(mask):
            new_pred[mask] = mode(y_true[mask])[0]
    return new_pred

y_pred_mapped = map_clusters(y_true, y_pred)

# === Compute Metrics ===
acc = accuracy_score(y_true, y_pred_mapped)
f1 = f1_score(y_true, y_pred_mapped, average='weighted')
prec = precision_score(y_true, y_pred_mapped, average='weighted')
rec = recall_score(y_true, y_pred_mapped, average='weighted')

# === Print Results ===
metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score", "Precision", "Recall"],
    "Score": [acc, f1, prec, rec]
})

print("\n📊 KMeans Clustering Evaluation Metrics:")
print(metrics_table.to_markdown(index=False))

# === Confusion Matrix Heatmap ===
cm = confusion_matrix(y_true, y_pred_mapped)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=True,
            xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel("Predicted Cluster")
plt.ylabel("True Cluster")
plt.title("GraphCodeBERT Confusion Matrix Heatmap (KMeans Clustering)")
plt.tight_layout()
plt.show()

