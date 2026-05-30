import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200
CLOUD = "aws"
REGION = "us-east-1"


def wait_until_ready(pc: Pinecone, index_name: str, timeout_sec: int = 180):
    print(f"[INFO] Waiting for index '{index_name}' to become ready...")
    start = time.time()
    while time.time() - start < timeout_sec:
        desc = pc.describe_index(index_name)
        if getattr(desc.status, "ready", False):
            print("[INFO] Index is ready")
            return
        time.sleep(5)
    raise TimeoutError(f"Index '{index_name}' was not ready after {timeout_sec} seconds")


def get_existing_index_names(pc: Pinecone):
    indexes = pc.list_indexes()
    return [idx.name for idx in indexes]


def safe_text(value, max_len: int):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()[:max_len]


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    print("[INFO] Loading prepared dataset and embeddings...")
    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    embeddings = np.load(INPUT_EMBEDDINGS).astype(np.float32)

    if len(df) != len(embeddings):
        raise ValueError(f"Rows count mismatch: df={len(df)}, embeddings={len(embeddings)}")
    if embeddings.shape[1] != VECTOR_DIM:
        raise ValueError(f"Wrong vector dimension: expected {VECTOR_DIM}, got {embeddings.shape[1]}")

    pc = Pinecone(api_key=api_key)
    existing_indexes = get_existing_index_names(pc)

    if INDEX_NAME not in existing_indexes:
        print(f"[INFO] Creating index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        wait_until_ready(pc, INDEX_NAME)
    else:
        print(f"[INFO] Index already exists: {INDEX_NAME}")

    index = pc.Index(INDEX_NAME)

    print("[INFO] Uploading vectors to Pinecone...")
    total = 0
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Upserting batches"):
        end = min(start + BATCH_SIZE, len(df))
        batch_df = df.iloc[start:end]
        batch_embeddings = embeddings[start:end]

        vectors = []
        for local_pos, (_, row) in enumerate(batch_df.iterrows()):
            row_idx = start + local_pos
            metadata = {
                "arxiv_id": safe_text(row.get("id", ""), 100),
                "title": safe_text(row.get("title", ""), 500),
                "abstract": safe_text(row.get("abstract", ""), 500),
                "authors": safe_text(row.get("authors", ""), 200),
                "year": int(row.get("year", 0)) if not pd.isna(row.get("year", 0)) else 0,
                "category": safe_text(row.get("category", "unknown"), 50),
            }
            vectors.append({
                "id": f"paper_{row_idx}",
                "values": batch_embeddings[local_pos].tolist(),
                "metadata": metadata,
            })

        index.upsert(vectors=vectors)
        total += len(vectors)

    stats = index.describe_index_stats()
    print(f"[INFO] Uploaded vectors: {total}")
    print(f"[INFO] Pinecone total vector count: {stats.get('total_vector_count')}")


if __name__ == "__main__":
    main()
