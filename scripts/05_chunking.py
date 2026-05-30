import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
INPUT_PARQUET = "data/arxiv_subset.parquet"
FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"
CLOUD = "aws"
REGION = "us-east-1"
BATCH_SIZE = 100
MAX_WORDS = 120
OVERLAP_WORDS = 25
TOP_K = 5
TEST_QUERIES = [
    "neural networks for image recognition",
    "reinforcement learning for decision making",
    "language models for text generation",
]


def split_sentences(text: str):
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def fixed_size_chunks(text: str, max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS):
    words = str(text).split()
    if not words:
        return []

    chunks = []
    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than max_words")

    for start in range(0, len(words), step):
        chunk_words = words[start:start + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def sentence_preserving_chunks(text: str, max_words: int = MAX_WORDS):
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)

        if current and current_len + sentence_len > max_words:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def get_existing_index_names(pc: Pinecone):
    return [idx.name for idx in pc.list_indexes()]


def wait_until_ready(pc: Pinecone, index_name: str, timeout_sec: int = 180):
    start = time.time()
    while time.time() - start < timeout_sec:
        desc = pc.describe_index(index_name)
        if getattr(desc.status, "ready", False):
            return
        time.sleep(5)
    raise TimeoutError(f"Index '{index_name}' was not ready after {timeout_sec} seconds")


def ensure_index(pc: Pinecone, index_name: str):
    if index_name not in get_existing_index_names(pc):
        print(f"[INFO] Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        wait_until_ready(pc, index_name)
    else:
        print(f"[INFO] Index already exists: {index_name}")


def build_chunk_records(df: pd.DataFrame, strategy: str):
    records = []
    longest = df.sort_values("abstract", key=lambda s: s.astype(str).str.len(), ascending=False).head(30)

    for row_idx, row in longest.reset_index(drop=True).iterrows():
        abstract = str(row.get("abstract", ""))
        chunks = fixed_size_chunks(abstract) if strategy == "fixed" else sentence_preserving_chunks(abstract)

        for chunk_id, chunk_text in enumerate(chunks):
            records.append({
                "id": f"{strategy}_{row_idx}_{chunk_id}",
                "text": chunk_text,
                "metadata": {
                    "arxiv_id": str(row.get("id", ""))[:100],
                    "title": str(row.get("title", "")).replace("\n", " ")[:500],
                    "text": chunk_text[:1000],
                    "chunk_number": int(chunk_id),
                    "year": int(row.get("year", 0)) if not pd.isna(row.get("year", 0)) else 0,
                    "category": str(row.get("category", "unknown"))[:50],
                    "strategy": strategy,
                },
            })
    return records


def upsert_chunks(pc: Pinecone, model: SentenceTransformer, index_name: str, records):
    index = pc.Index(index_name)
    print(f"[INFO] Uploading {len(records)} chunks to {index_name}...")

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Upserting {index_name}"):
        batch = records[start:start + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

        vectors = []
        for record, embedding in zip(batch, embeddings):
            vectors.append({
                "id": record["id"],
                "values": embedding.astype(np.float32).tolist(),
                "metadata": record["metadata"],
            })
        index.upsert(vectors=vectors)


def search_chunks(pc: Pinecone, model: SentenceTransformer, index_name: str, query: str):
    index = pc.Index(index_name)
    query_emb = model.encode(query, normalize_embeddings=True).astype(np.float32)
    return index.query(vector=query_emb.tolist(), top_k=TOP_K, include_metadata=True).matches


def print_chunk_results(index_name: str, query: str, matches):
    print(f"\n========== {index_name} | query: {query} ==========")
    for rank, match in enumerate(matches, start=1):
        md = match.metadata
        print(f"\n#{rank} | score={match.score:.4f}")
        print(f"Title: {md.get('title', '')}")
        print(f"Chunk: {md.get('chunk_number')} | Year: {md.get('year')} | Category: {md.get('category')}")
        print(f"Text: {md.get('text', '')[:400]}...")


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    print("[INFO] Loading dataset and model...")
    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    model = SentenceTransformer(MODEL_NAME)
    pc = Pinecone(api_key=api_key)

    ensure_index(pc, FIXED_INDEX)
    ensure_index(pc, SEMANTIC_INDEX)

    fixed_records = build_chunk_records(df, strategy="fixed")
    semantic_records = build_chunk_records(df, strategy="semantic")

    print(f"[INFO] Fixed-size chunks: {len(fixed_records)}")
    print(f"[INFO] Sentence-preserving chunks: {len(semantic_records)}")

    upsert_chunks(pc, model, FIXED_INDEX, fixed_records)
    upsert_chunks(pc, model, SEMANTIC_INDEX, semantic_records)

    for query in TEST_QUERIES:
        fixed_matches = search_chunks(pc, model, FIXED_INDEX, query)
        semantic_matches = search_chunks(pc, model, SEMANTIC_INDEX, query)
        print_chunk_results(FIXED_INDEX, query, fixed_matches)
        print_chunk_results(SEMANTIC_INDEX, query, semantic_matches)


if __name__ == "__main__":
    main()
