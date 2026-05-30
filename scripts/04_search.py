
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
QUERY = "teaching machines to recognize objects in pictures"
INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"


def print_results(title, matches):
    print(f"\n========== {title} ==========")
    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {}) if isinstance(match, dict) else match.metadata
        score = match.get("score") if isinstance(match, dict) else match.score
        print(f"\n#{rank} | score={score:.4f}")
        print(f"Title: {metadata.get('title', '')}")
        print(f"Category: {metadata.get('category', '')} | Year: {metadata.get('year', '')}")
        print(f"Abstract: {metadata.get('abstract', '')[:350]}...")


def local_top_k(query_emb, embeddings, df, metric: str, top_k: int):
    if metric == "cosine":
        scores = embeddings @ query_emb
        order = np.argsort(-scores)[:top_k]
        values = scores[order]
    elif metric == "dot":
        scores = embeddings @ query_emb
        order = np.argsort(-scores)[:top_k]
        values = scores[order]
    elif metric == "l2":
        distances = np.linalg.norm(embeddings - query_emb, axis=1)
        order = np.argsort(distances)[:top_k]
        values = distances[order]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    results = []
    for idx, value in zip(order, values):
        row = df.iloc[int(idx)]
        results.append({
            "score": float(value),
            "metadata": {
                "title": row.get("title", ""),
                "category": row.get("category", ""),
                "year": int(row.get("year", 0)),
                "abstract": row.get("abstract", ""),
            },
        })
    return results


def search_pinecone(index, query_emb, filter_dict=None):
    return index.query(
        vector=query_emb.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_dict,
    ).matches


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    print("[INFO] Loading model, Pinecone index, local dataset and embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    embeddings = np.load(INPUT_EMBEDDINGS).astype(np.float32)

    query_emb = model.encode(QUERY, normalize_embeddings=True).astype(np.float32)
    print(f"[INFO] Query: {QUERY}")
    print(f"[INFO] Query norm: {np.linalg.norm(query_emb):.4f}")

    pure_results = search_pinecone(index, query_emb)
    print_results("Pinecone semantic search", pure_results)

    current_year = int(df["year"].max())
    filtered_recent_rl = search_pinecone(
        index,
        query_emb,
        filter_dict={"year": {"$gte": current_year - 5}, "category": {"$eq": "cs.LG"}},
    )
    print_results("Pinecone filtered search: recent cs.LG", filtered_recent_rl)

    filtered_old = search_pinecone(
        index,
        query_emb,
        filter_dict={"year": {"$lte": 2014}},
    )
    print_results("Pinecone filtered search: year <= 2014", filtered_old)

    cosine_results = local_top_k(query_emb, embeddings, df, "cosine", TOP_K)
    dot_results = local_top_k(query_emb, embeddings, df, "dot", TOP_K)
    l2_results = local_top_k(query_emb, embeddings, df, "l2", TOP_K)

    print_results("Local cosine similarity", cosine_results)
    print_results("Local dot product", dot_results)
    print_results("Local L2 distance - lower is better", l2_results)

    cosine_titles = [r["metadata"]["title"] for r in cosine_results]
    dot_titles = [r["metadata"]["title"] for r in dot_results]
    l2_titles = [r["metadata"]["title"] for r in l2_results]

    print("\n========== Metric comparison ==========")
    print(f"Cosine and dot top-5 are identical: {cosine_titles == dot_titles}")
    print(f"Cosine and L2 top-5 are identical: {cosine_titles == l2_titles}")


if __name__ == "__main__":
    main()
