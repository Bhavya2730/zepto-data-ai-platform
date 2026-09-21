from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"

expected = [f"doc_{i:02d}.txt" for i in range(1, 9)]
actual = sorted(p.name for p in DOCS_DIR.glob("doc_*.txt"))

print(f"Expected documents: {len(expected)}")
print(f"Found documents:    {len(actual)}")
print("Files:", actual)

assert actual == expected, f"Corpus files do not match expected set: {actual}"

for filename in expected:
    path = DOCS_DIR / filename
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{filename} is empty"
    print(f"{filename}: {len(text)} characters")

print("\nStage 3.1 corpus verification passed.")
