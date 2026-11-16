from rag.loader import load_docs
from rag.chunking import chunk_fixed

docs = load_docs()

for doc in docs:
    if doc.id == "of-450h-merhav-hybrid":
        chunks = chunk_fixed(doc)

        for chunk in chunks:
            print(f"\n--- {chunk.id} ---")
            print(chunk.text)