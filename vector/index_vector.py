"""
Vector indexing pipeline for Indonesian legal documents.

Reads the same index JSON files used by vectorless-rag, collects all leaf nodes,
embeds each chunk using the chosen embedding model, and stores in Qdrant.

Supported embedding models (all local via SentenceTransformer):
    bge-m3                          (1024d, BAAI/bge-m3, MIRACL SOTA, 8K context)
    multilingual-e5-large-instruct  (1024d, intfloat/..., MMTEB best public, 512-token)
    all-indobert-base-v4            (768d,  LazarusNLP/..., Indonesian-specific, 128-token)

Qdrant storage:
    --qdrant-path ./qdrant_local    local file-based mode (no server needed)
    (omit)                          server mode via QDRANT_URL env var

Collection name is auto-derived as  law-{granularity}-{model_short}
unless --collection is supplied explicitly.

Usage:
    python -m vector.index_vector --source data/index_pasal --qdrant-path ./qdrant_local
    python -m vector.index_vector --source data/index_pasal --model multilingual-e5-large-instruct --qdrant-path ./qdrant_local
    python -m vector.index_vector --source data/index_ayat  --model all-indobert-base-v4 --qdrant-path ./qdrant_local
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

BATCH_SIZE_ST = 64      # SentenceTransformer encode batch size
UPSERT_BATCH = 100

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

DEFAULT_SOURCE = Path("data/index_pasal")

# ============================================================
# EMBEDDING MODEL REGISTRY
# ============================================================

_EMBEDDING_MODEL_MAP: dict[str, dict] = {
    # MIRACL SOTA, 8K context window, handles long pasal
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "short": "bgem3",
    },
    # Indonesian-specific, 128-token limit
    "all-indobert-base-v4": {
        "model_id": "LazarusNLP/all-indobert-base-v4",
        "dim": 768,
        "short": "indobert",
    },
    # MMTEB best public multilingual, 512-token context
    "multilingual-e5-large-instruct": {
        "model_id": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
        "short": "e5",
    },
}

_SOURCE_TO_GRAN = {
    "index_pasal": "pasal",
    "index_ayat": "ayat",
    "index_full_split": "full_split",
}


def _source_to_gran(source_dir: Path) -> str:
    """Infer granularity label from source directory name."""
    return _SOURCE_TO_GRAN.get(source_dir.name, source_dir.name)


# ============================================================
# CHUNK COLLECTION
# ============================================================

def collect_leaf_nodes(nodes: list[dict], doc_id: str, doc_title: str) -> list[dict]:
    """Recursively collect all leaf nodes from a document tree structure."""
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


# ============================================================
# EMBEDDING
# ============================================================

def embed_texts_st(texts: list[str], model_id: str) -> list[list[float]]:
    """Embed texts using SentenceTransformer (local, no API calls).

    Passages are embedded without instruction prefix for all three models:
    - BGE-M3: no prefix needed
    - all-indobert-base-v4: no prefix needed
    - multilingual-e5-large-instruct: instruction prefix is query-only; passages use raw text
    """
    from sentence_transformers import SentenceTransformer
    print(f"  Loading SentenceTransformer: {model_id}")
    st = SentenceTransformer(model_id)
    vecs = st.encode(
        texts,
        batch_size=BATCH_SIZE_ST,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vecs]


# ============================================================
# INDEX BUILD
# ============================================================

def build_index(
    source_dir: Path,
    collection_name: str | None = None,
    model: str = "bge-m3",
    qdrant_path: str | None = None,
    catalog_filename: str = "catalog.json",
):
    """Build Qdrant collection from all index JSON files in source_dir.

    Args:
        source_dir:       Path to index JSON directory (e.g. data/index_pasal)
        collection_name:  Qdrant collection name; auto-derived if None
        model:            Embedding model key from _EMBEDDING_MODEL_MAP
        qdrant_path:      Path to local Qdrant storage (None = use server URL)
        catalog_filename: Catalog JSON filename inside source_dir (default: catalog.json).
                          Use catalog_gt.json to embed only GT-verified documents.
    """
    model_cfg = _EMBEDDING_MODEL_MAP.get(model)
    if not model_cfg:
        print(f"ERROR: Unknown model {model!r}. Choose from: {list(_EMBEDDING_MODEL_MAP)}")
        sys.exit(1)

    if collection_name is None:
        gran = _source_to_gran(source_dir)
        collection_name = f"law-{gran}-{model_cfg['short']}"

    embedding_dim = model_cfg["dim"]

    catalog_path = source_dir / catalog_filename
    if not catalog_path.exists():
        print(f"ERROR: {catalog_filename} not found at {catalog_path}")
        sys.exit(1)

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Found {len(catalog)} documents in catalog")
    print(f"Model:      {model} (SentenceTransformer, {embedding_dim}d)")
    print(f"Collection: {collection_name}")
    if qdrant_path:
        print(f"Qdrant:     local path {qdrant_path}")
    else:
        print(f"Qdrant:     server {QDRANT_URL}")

    # Collect all chunks
    all_chunks: list[dict] = []
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
        print(f"  {doc_id}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("ERROR: No chunks found")
        sys.exit(1)

    # Embed
    texts = [c["text"] for c in all_chunks]
    print(f"\nEmbedding {len(texts)} chunks with {model}...")
    embeddings = embed_texts_st(texts, model_cfg["model_id"])

    # Connect to Qdrant
    if qdrant_path:
        qdrant = QdrantClient(path=qdrant_path)
    else:
        qdrant = QdrantClient(url=QDRANT_URL)
        print(f"\nUploading to Qdrant ({QDRANT_URL})...")

    # Recreate collection (drops existing data)
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )

    # Build and upsert points
    points = [
        PointStruct(
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
        )
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings))
    ]

    for i in range(0, len(points), UPSERT_BATCH):
        batch = points[i:i + UPSERT_BATCH]
        qdrant.upsert(collection_name=collection_name, points=batch)
        if len(points) > UPSERT_BATCH:
            print(f"  Uploaded {min(i + UPSERT_BATCH, len(points))}/{len(points)} points")

    # Verify
    info = qdrant.get_collection(collection_name)
    print(f"\nDone!")
    print(f"  Collection: {collection_name}")
    print(f"  Points:     {info.points_count}")
    print(f"  Vectors:    {embedding_dim}d, distance=Cosine")
    print(f"  Status:     {info.status}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Qdrant vector index from legal document index JSONs"
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help="Path to index JSON directory (default: data/index_pasal)",
    )
    parser.add_argument(
        "--model", default="bge-m3",
        choices=list(_EMBEDDING_MODEL_MAP),
        help="Embedding model to use (default: bge-m3)",
    )
    parser.add_argument(
        "--collection", default=None,
        help="Qdrant collection name (auto-derived from --source and --model if omitted)",
    )
    parser.add_argument(
        "--qdrant-path", default=None,
        help="Path to local Qdrant storage directory (uses server URL if omitted)",
    )
    parser.add_argument(
        "--catalog", default="catalog.json",
        help="Catalog filename inside --source dir (default: catalog.json). "
             "Use catalog_gt.json to embed only GT-verified documents.",
    )
    args = parser.parse_args()
    build_index(
        source_dir=args.source,
        collection_name=args.collection,
        model=args.model,
        qdrant_path=args.qdrant_path,
        catalog_filename=args.catalog,
    )
