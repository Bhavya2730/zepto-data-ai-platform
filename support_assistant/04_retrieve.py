from pathlib import Path
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


def load_collection():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Missing ChromaDB directory: {CHROMA_DIR}. "
            "Run 03_embed_and_index.py first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' was not found. "
            "Run 03_embed_and_index.py first."
        ) from exc

    if collection.count() == 0:
        raise ValueError("ChromaDB collection is empty.")

    return collection


def retrieve(query: str, top_k: int = TOP_K):
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
        similarity = 1.0 - distance

        retrieved.append(
            {
                "chunk_id": chunk_id,
                "document_id": results["metadatas"][0][index]["document_id"],
                "source_file": results["metadatas"][0][index]["source_file"],
                "chunk_index": results["metadatas"][0][index]["chunk_index"],
                "distance": distance,
                "similarity": similarity,
                "text": results["documents"][0][index],
            }
        )

    return retrieved


def print_results(query: str, results):
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult #{rank}")
        print(f"Chunk ID:       {result['chunk_id']}")
        print(f"Document:       {result['document_id']}")
        print(f"Source file:    {result['source_file']}")
        print(f"Chunk index:    {result['chunk_index']}")
        print(f"Cosine distance:{result['distance']:.6f}")
        print(f"Similarity:     {result['similarity']:.6f}")
        print(f"Text:           {result['text']}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve the top 3 Zepto policy chunks from ChromaDB."
    )

    parser.add_argument(
        "query",
        nargs="?",
        default="What is the delivery fee?",
        help="Question to retrieve policy context for.",
    )

    args = parser.parse_args()

    results = retrieve(args.query, TOP_K)

    if len(results) != TOP_K:
        raise AssertionError(
            f"Expected {TOP_K} results, received {len(results)}."
        )

    print_results(args.query, results)


if __name__ == "__main__":
    main()