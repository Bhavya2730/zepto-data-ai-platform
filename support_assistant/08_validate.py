import json
from pathlib import Path

from fastapi.testclient import TestClient

from support_assistant.main import app

def check_file(path):
    assert Path(path).exists(), f"Missing file: {path}"
    print(f"[PASS] Found: {path}")


def main():
    print("=" * 80)
    print("STAGE 3.8 - FINAL END-TO-END VALIDATION")
    print("=" * 80)

    # 1. Check knowledge-base documents
    docs_dir = Path("support_assistant/docs")
    documents = sorted(docs_dir.glob("doc_*.txt"))

    print(f"\nKnowledge-base documents found: {len(documents)}")
    assert len(documents) == 8, "Expected exactly 8 policy documents."
    print("[PASS] 8 policy documents found.")

    # 2. Check chunks
    chunks_path = Path("support_assistant/chunks.jsonl")
    check_file(chunks_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"[PASS] Chunks found: {len(chunks)}")
    assert len(chunks) == 22, "Expected exactly 22 chunks."

    # 3. Check ChromaDB
    chroma_dir = Path("support_assistant/chroma_db")
    assert chroma_dir.exists(), "ChromaDB directory not found."
    print("[PASS] ChromaDB directory found.")


    # 5. Test FastAPI
    print("\nTesting FastAPI /ask endpoint...")

    client = TestClient(app)

    # Policy query
    policy_response = client.post(
        "/ask",
        json={"query": "What is the delivery fee?"}
    )

    assert policy_response.status_code == 200
    policy_json = policy_response.json()

    assert "answer" in policy_json
    assert "sources" in policy_json
    assert "confidence" in policy_json
    assert len(policy_json["sources"]) > 0

    print("\nPolicy API response:")
    print(json.dumps(policy_json, indent=2))

    # General query
    general_response = client.post(
        "/ask",
        json={"query": "What is the capital of France?"}
    )

    assert general_response.status_code == 200
    general_json = general_response.json()

    assert "answer" in general_json
    assert "sources" in general_json
    assert "confidence" in general_json
    assert general_json["sources"] == []

    print("\nGeneral API response:")
    print(json.dumps(general_json, indent=2))

    print("\n" + "=" * 80)
    print("ALL STAGE 3.8 CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()