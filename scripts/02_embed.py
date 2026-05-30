import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64


def main():
    print("[INFO] Loading dataset...")
    df = pd.read_parquet(INPUT_FILE)

    required_cols = {"title", "abstract"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"[INFO] Loaded rows: {len(df)}")
    texts = (
        df["title"].fillna("").astype(str).str.replace("\n", " ").str.strip()
        + " [SEP] "
        + df["abstract"].fillna("").astype(str).str.replace("\n", " ").str.strip()
    ).tolist()

    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] Encoding texts...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)

    first_norm = float(np.linalg.norm(embeddings[0])) if len(embeddings) else 0.0
    print(f"[INFO] Processed texts: {len(texts)}")
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] First embedding norm: {first_norm:.4f}")
    print(f"[INFO] Saved embeddings to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
