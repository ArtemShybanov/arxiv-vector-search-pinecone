import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10
RRF_K = 60
INPUT_PARQUET = "data/arxiv_subset.parquet"
DEMO_QUERIES = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text",
]


def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9_]+", str(text).lower())


def build_bm25(df: pd.DataFrame):
    corpus = (
        df["title"].fillna("").astype(str)
        + " "
        + df["abstract"].fillna("").astype(str)
    ).tolist()
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def bm25_search(df: pd.DataFrame, bm25: BM25Okapi, query: str, top_k: int = TOP_K):
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores)[:top_k]
    return [
        {
            "id": f"paper_{int(idx)}",
            "rank": rank,
            "score": float(scores[idx]),
            "title": df.iloc[int(idx)].get("title", ""),
            "year": int(df.iloc[int(idx)].get("year", 0)),
            "category": df.iloc[int(idx)].get("category", ""),
            "abstract": df.iloc[int(idx)].get("abstract", ""),
            "source": "bm25",
        }
        for rank, idx in enumerate(order, start=1)
    ]


def vector_search(index, model: SentenceTransformer, query: str, top_k: int = TOP_K):
    query_emb = model.encode(query, normalize_embeddings=True).astype(np.float32)
    matches = index.query(vector=query_emb.tolist(), top_k=top_k, include_metadata=True).matches
    results = []
    for rank, match in enumerate(matches, start=1):
        md = match.metadata
        results.append({
            "id": match.id,
            "rank": rank,
            "score": float(match.score),
            "title": md.get("title", ""),
            "year": md.get("year", ""),
            "category": md.get("category", ""),
            "abstract": md.get("abstract", ""),
            "source": "vector",
        })
    return results


def reciprocal_rank_fusion(result_lists, k: int = RRF_K, top_k: int = TOP_K):
    scores = {}
    docs = {}

    for results in result_lists:
        for item in results:
            doc_id = item["id"]
            rank = item["rank"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            docs[doc_id] = item

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    fused = []
    for rank, doc_id in enumerate(ranked_ids, start=1):
        item = docs[doc_id].copy()
        item["rank"] = rank
        item["rrf_score"] = scores[doc_id]
        item["source"] = "hybrid_rrf"
        fused.append(item)
    return fused


def print_results(title: str, results, score_field: str = "score"):
    print(f"\n========== {title} ==========")
    for item in results[:5]:
        score = item.get(score_field, item.get("score", 0.0))
        print(f"\n#{item['rank']} | {score_field}={score:.6f}")
        print(f"Title: {item['title']}")
        print(f"Year: {item['year']} | Category: {item['category']}")
        print(f"Abstract: {str(item['abstract'])[:350]}...")


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    print("[INFO] Loading dataset, BM25 index, embedding model and Pinecone index...")
    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    bm25 = build_bm25(df)
    model = SentenceTransformer(MODEL_NAME)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    comparison_rows = []

    for query in DEMO_QUERIES:
        print(f"\n\n################ QUERY: {query} ################")
        bm25_results = bm25_search(df, bm25, query, TOP_K)
        vector_results = vector_search(index, model, query, TOP_K)
        hybrid_results = reciprocal_rank_fusion([bm25_results, vector_results], k=RRF_K, top_k=TOP_K)

        print_results("BM25 top-5", bm25_results, "score")
        print_results("Vector top-5", vector_results, "score")
        print_results("Hybrid RRF top-5", hybrid_results, "rrf_score")

        for method, results in [
            ("BM25", bm25_results),
            ("Vector", vector_results),
            ("Hybrid RRF", hybrid_results),
        ]:
            for item in results[:5]:
                comparison_rows.append({
                    "query": query,
                    "method": method,
                    "rank": item["rank"],
                    "title": item["title"],
                    "year": item["year"],
                    "category": item["category"],
                })

    comparison = pd.DataFrame(comparison_rows)
    print("\n========== Comparison table ==========")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
