import logging
import chromadb
from chromadb.utils import embedding_functions
from app.core.config import CHROMA_DB_PATH, COLLECTION_NAME

logger = logging.getLogger('app.services.embedding')

embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Initialize Chroma client — saves to disk
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)


def embed_chunks(chunks: list[dict]) -> None:
    """Embed all chunks and store in Chroma DB."""
    logger.info(f"Embedding {len(chunks)} chunks...")

    texts = [chunk['text'] for chunk in chunks]
    ids = [f"{chunk['source']}_{chunk['chunk_index']}" for chunk in chunks]
    metadatas = [
        {
            "source": chunk['source'],
            "chunk_index": chunk['chunk_index'],
            "token_count": chunk['token_count']
        }
        for chunk in chunks
    ]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
        logger.info(
            f"Stored batch {i//batch_size + 1} | "
            f"chunks {i} to {min(i + batch_size, len(chunks))}"
        )

    logger.info(f"Embedding complete | total: {len(chunks)}")


def search_similar_chunks(query: str, n_results: int = 5) -> list[dict]:
    """Search for chunks similar to the query."""
    logger.info(f"Searching for: '{query[:50]}...'")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'chunk_index': results['metadatas'][0][i]['chunk_index'],
            'similarity': 1 - results['distances'][0][i]
        })

    logger.info(
        f"Search complete | "
        f"found {len(chunks)} | "
        f"top similarity: {chunks[0]['similarity']:.3f}"
    )

    return chunks


def get_collection_count() -> int:
    """Return how many chunks are stored in Chroma."""
    return collection.count()
