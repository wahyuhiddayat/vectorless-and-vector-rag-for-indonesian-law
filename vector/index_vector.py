"""
Vector indexing pipeline for Indonesian legal documents.

Reads the same index JSON files used by vectorless-rag, chunks by Pasal (leaf node),
embeds each chunk using Google gemini-embedding-001, and stores in Qdrant.

Core logic (same as lexin-baseline):
- Embedding model: gemini-embedding-001 (successor to text-embedding-004)
- Each Pasal = 1 chunk (same granularity as tree leaf nodes for fair comparison)
- Metadata stored as Qdrant payload (doc_id, title, navigation_path, text)

Usage:
    python -m vector.index_vector
    python -m vector.index_vector --source data/index_pasal
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
BATCH_SIZE = 50  # Google API batch limit
DEFAULT_COLLECTION = "law-pasal"
QDRANT_URL = "http://localhost:6333"

# Default: read from shared data/index_pasal/ at project root
DEFAULT_SOURCE = Path("data/index_pasal")


def get_embed_client():
    """Initialize Google GenAI client."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def collect_leaf_nodes(nodes: list[dict], doc_id: str, doc_title: str) -> list[dict]:
    """Recursively collect all leaf nodes (Pasal) from a tree structure."""
    chunks = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            chunks.extend(collect_leaf_nodes(node["nodes"], doc_id, doc_title))
        elif node.get("text"):
            chunk_text = node["text"]
            penjelasan = node.get("penjelasan", "")
            if penjelasan and penjelasan != "Cukup jelas.":
                chunk_text += f"\n\nPenjelasan Resmi:\n{penjelasan}"

            chunks.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "navigation_path": node.get("navigation_path", ""),
                "text": chunk_text,
            })
    return chunks


def embed_texts(client, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using gemini-embedding-001. Handles batching."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                break
            except Exception as e:
                if ("rate" in str(e).lower() or "429" in str(e)) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        for emb in result.embeddings:
            all_embeddings.append([float(x) for x in emb.values])

        if len(texts) > BATCH_SIZE:
            print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    return all_embeddings


def build_index(source_dir: Path, collection_name: str = DEFAULT_COLLECTION):
    """Build Qdrant collection from all index JSON files."""
    catalog_path = source_dir / "catalog.json"
    if not catalog_path.exists():
        print(f"ERROR: catalog.json not found at {catalog_path}")
        sys.exit(1)

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Found {len(catalog)} documents in catalog")

    # Collect all chunks from all documents
    all_chunks = []
    for doc_meta in catalog:
        doc_id = doc_meta["doc_id"]
        category = doc_id.split("-")[0].upper()
        doc_path = source_dir / category / f"{doc_id}.json"
        if not doc_path.exists():
            print(f"  SKIP: {doc_id} — index file not found")
            continue

        with open(doc_path, encoding="utf-8") as f:
            doc = json.load(f)

        chunks = collect_leaf_nodes(doc.get("structure", []), doc_id, doc_meta["judul"])
        all_chunks.extend(chunks)
        print(f"  {doc_id}: {len(chunks)} chunks (Pasal)")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("ERROR: No chunks found")
        sys.exit(1)

    # Embed all chunks
    print(f"\nEmbedding with {EMBEDDING_MODEL}...")
    embed_client = get_embed_client()
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(embed_client, texts)

    # Create Qdrant collection
    print(f"\nUploading to Qdrant ({QDRANT_URL})...")
    qdrant = QdrantClient(url=QDRANT_URL)

    # Recreate collection (drop if exists)
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    # Upload in batches
    points = []
    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "doc_id": chunk["doc_id"],
                "doc_title": chunk["doc_title"],
                "node_id": chunk["node_id"],
                "title": chunk["title"],
                "navigation_path": chunk["navigation_path"],
                "text": chunk["text"],
            },
        ))

    # Upsert in batches of 100
    UPSERT_BATCH = 100
    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i:i + UPSERT_BATCH]
        qdrant.upsert(collection_name=collection_name, points=batch)
        if len(points) > UPSERT_BATCH:
            print(f"  Uploaded {min(i + UPSERT_BATCH, len(points))}/{len(points)} points")

    # Verify
    info = qdrant.get_collection(collection_name)
    print(f"\nDone!")
    print(f"  Collection: {collection_name}")
    print(f"  Points: {info.points_count}")
    print(f"  Vectors: {EMBEDDING_DIM}d, distance=Cosine")
    print(f"  Status: {info.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant vector index from legal document index JSONs")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Path to index JSON directory (default: data/index_pasal)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})")
    args = parser.parse_args()
    build_index(args.source, collection_name=args.collection)
