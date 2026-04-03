
import sys
sys.path.append('.')

from app.core.logging_config import setup_logging
from app.services.pdf_service import process_pdf
from app.services.embedding_service import (
    embed_chunks,
    search_similar_chunks,
    get_collection_count
)

setup_logging()


count = get_collection_count()
print(f"\nChunks already in Chroma: {count}")

if count == 0:
    print("No chunks found — processing and embedding PDF...")


    chunks = process_pdf('documents/WorldEnergyOutlook2023.pdf')
    print(f"Chunks created: {len(chunks)}")


    embed_chunks(chunks)
    print(f"Embedding complete!")
else:
    print(f"Using existing embeddings — skipping PDF processing")


print(f"\n{'='*50}")
print("SEMANTIC SEARCH TESTS")
print(f"{'='*50}\n")

test_queries = [
    "What are the renewable energy targets for 2030?",
    "How is solar power affecting electricity prices?",
    "What is the role of natural gas in the energy transition?"
]

for query in test_queries:
    print(f"Query: {query}")
    print(f"{'-'*40}")

    results = search_similar_chunks(query, n_results=3)

    for i, chunk in enumerate(results):
        print(f"Result {i+1} (similarity: {chunk['similarity']:.3f}):")
        print(f"  Source: {chunk['source']}")
        print(f"  Preview: {chunk['text'][:150]}...")
        print()

    print()