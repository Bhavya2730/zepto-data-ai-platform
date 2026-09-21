from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Missing ChromaDB directory: {CHROMA_DIR}. "
            "Run 03_embed_and_index.py first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise ValueError("ChromaDB collection is empty.")

    return collection


def retrieve(query: str, top_k: int = 3):
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    collection = load_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for index, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][index]

        retrieved.append(
            {
                "chunk_id": chunk_id,
                "document_id": results["metadatas"][0][index]["document_id"],
                "source_file": results["metadatas"][0][index]["source_file"],
                "chunk_index": results["metadatas"][0][index]["chunk_index"],
                "distance": distance,
                "similarity": 1.0 - distance,
                "text": results["documents"][0][index],
            }
        )

    return retrieved