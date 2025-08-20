import os
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_DIR = "separated_by_language/cpp"   # your folder with Java files organized by cluster
OUTPUT_PATH = "embeddings/cpp_CodeBERT_embeddings.pkl"

# === SETUP MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)
model.eval()

# === FILE LOADING ===
def load_files(folder):
    file_records = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.endswith(".cpp") or fname.endswith(".c"):
                full_path = os.path.join(root, fname)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        code = f.read()
                        cluster = os.path.basename(os.path.dirname(full_path))  # assumes: .../cluster_x/filename.java
                        file_records.append({
                            "path": full_path,
                            "cluster": cluster,
                            "code": code
                        })
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    return pd.DataFrame(file_records)

# === EMBEDDING FUNCTION ===
def get_embedding(code):
    try:
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_embedding.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None


# === PROCESSING ===
df = load_files(INPUT_DIR)
print(f"Loaded {len(df)} files")

tqdm.pandas()
df["embedding"] = df["code"].progress_apply(get_embedding)

# === SAVE ===
df.to_pickle(OUTPUT_PATH)
print(f"Saved embeddings to {OUTPUT_PATH}")
