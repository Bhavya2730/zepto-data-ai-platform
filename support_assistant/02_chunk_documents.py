from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHUNKS_FILE = BASE_DIR / "chunks.jsonl"

CHUNK_SIZE = 250
OVERLAP = 40


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP
) -> list[str]:
    """Normalize whitespace and split text into overlapping character chunks."""

    text = " ".join(text.split())

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def main():
    records = []

    doc_paths = sorted(DOCS_DIR.glob("doc_*.txt"))

    if len(doc_paths) != 8:
        raise ValueError(
            f"Expected 8 corpus documents, found {len(doc_paths)}"
        )

    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        pieces = chunk_text(text)

        for index, piece in enumerate(pieces):
            records.append(
                {
                    "chunk_id": f"{doc_path.stem}_chunk_{index:02d}",
                    "document_id": doc_path.stem,
                    "source_file": doc_path.name,
                    "chunk_index": index,
                    "text": piece,
                }
            )

    with CHUNKS_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Documents loaded: {len(doc_paths)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {OVERLAP} characters")
    print(f"Total chunks: {len(records)}")

    for doc_path in doc_paths:
        count = sum(
            1
            for record in records
            if record["document_id"] == doc_path.stem
        )
        print(f"{doc_path.name}: {count} chunk(s)")

    print(f"Saved chunks to: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()