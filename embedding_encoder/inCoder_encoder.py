import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_DIR = "separated_by_language/cpp"
OUTPUT_PATH = "embeddings/cpp_InCoder_embeddings.pkl"
MODEL_NAME = "facebook/incoder-1B"
BATCH_SIZE = 4  # Reduce if OOM occurs
MAX_LENGTH = 512

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer with correct settings
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tokenizer.pad_token = "<pad>"

# Load model with causal LM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
)
model.to(device)
model.eval()
print(f"Model loaded on {device}")


# === FILE LOADING ===
def load_files(folder):
    file_records = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.endswith(".c") or fname.endswith(".cpp"):
                full_path = os.path.join(root, fname)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        code = f.read()
                        cluster = os.path.basename(os.path.dirname(full_path))
                        file_records.append({
                            "path": full_path,
                            "cluster": cluster,
                            "code": code
                        })
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    return pd.DataFrame(file_records)


# === BATCH EMBEDDING ===
def get_batch_embeddings(codes):
    try:
        inputs = tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_token_type_ids=False,  # CRITICAL FIX: Disable token type IDs
            pad_to_multiple_of=8
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            # Mask padding tokens
            mask = inputs.attention_mask.unsqueeze(-1)
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        return embeddings.cpu().numpy()

    except Exception as e:
        print(f"Embedding failed: {e}")
        return [None] * len(codes)


# === PROCESSING ===
df = load_files(INPUT_DIR)
print(f"Loaded {len(df)} Java files")

# Batch processing
embeddings = []
for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_codes = df.iloc[i:i + BATCH_SIZE]["code"].tolist()
    batch_embeddings = get_batch_embeddings(batch_codes)
    embeddings.extend(batch_embeddings)

df["embedding"] = embeddings

# === SAVE ===
df.to_pickle(OUTPUT_PATH)
print(f"Saved embeddings to {OUTPUT_PATH}")