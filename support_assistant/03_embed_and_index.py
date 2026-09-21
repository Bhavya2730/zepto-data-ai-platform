from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
CHUNKS_FILE = BASE_DIR / "chunks.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CHUNKS_FILE}. Run 02_chunk_documents.py first."
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("chunks.jsonl contains no chunks.")

    return chunks


def main():
    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Loading local embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Embedding count: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "document_id": chunk["document_id"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    stored_count = collection.count()

    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Stored vectors: {stored_count}")
    print(f"ChromaDB path: {CHROMA_DIR}")

    test_chunk = chunks[0]

    test_embedding = model.encode(
        [test_chunk["text"]],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=test_embedding.tolist(),
        n_results=1,
    )

    returned_id = results["ids"][0][0]

    print("\nSmoke test:")
    print(f"Query chunk:    {test_chunk['chunk_id']}")
    print(f"Returned chunk: {returned_id}")

    if returned_id != test_chunk["chunk_id"]:
        raise AssertionError(
            "ChromaDB smoke test failed: expected the same chunk "
            "to be the nearest result."
        )

    print("Stage 3.3 embedding + ChromaDB verification passed.")


if __name__ == "__main__":
    main()