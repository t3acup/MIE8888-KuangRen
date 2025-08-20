import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# === CONFIGURATION ===
BATCH_SIZE = 16
INPUT_DIR = "separated_by_language/cpp"  # <-- Change if needed
OUTPUT_PATH = f"cpp_GraphCodeBERT_embeddings.pkl"

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === LOAD TOKENIZER + MODEL ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
model.to(device)
model.eval()

# === LOAD JAVA FILES ===
def load_files(folder):
    records = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".cpp") or file.endswith(".c"):           # cpp or java
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        code = f.read()
                        cluster = os.path.basename(os.path.dirname(full_path))
                        records.append({
                            "path": full_path,
                            "cluster": cluster,
                            "code": code
                        })
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    return pd.DataFrame(records)

df = load_files(INPUT_DIR)
print(f"Loaded {len(df)} files")

# === DATASET + DATALOADER ===
class CodeDataset(Dataset):
    def __init__(self, codes):
        self.codes = codes

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx]

def collate_fn(batch):
    return tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)

dataset = CodeDataset(df["code"].tolist())
loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# === EMBEDDING EXTRACTION ===
all_embeddings = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Extracting embeddings"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        all_embeddings.extend(cls_embeddings.cpu().numpy())

df["embedding"] = all_embeddings

# === SAVE OUTPUT ===
df.to_pickle(OUTPUT_PATH)
print(f"Saved embeddings to {OUTPUT_PATH}")