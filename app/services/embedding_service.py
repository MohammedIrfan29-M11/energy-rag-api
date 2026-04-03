import chromadb
import logging
from sentence_transformers import SentenceTransformer
from app.core.config import CHROMA_DB_PATH, COLLECTION_NAME

logger=logging.getLogger('app.services.embedding')

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection=client.get_or_create_collection(name=COLLECTION_NAME,metadata={"hnsw:space": "cosine"})

def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    embedding=model.encode(text)
    return embedding.tolist()

def embed_chunks(chunks: list[dict])-> None:
    """Embed all chunks and store in chroma db
    This is called once when a PDF is uploaded
    We batch the embeddings for efficiency —
    embedding 853 texts one by one would be slow.
    Batching sends them all at once."""
    logger.info(f"Embedding chunks {len(chunks)} chunks ....")

    texts = [chunks['text'] for chunks in chunks]

    embeddings = model.encode(texts,show_progress_bar=True)

    ids = [f"{chunk['source']}_{chunk['chunk_index']}" for chunk in chunks]
    documents = texts
    metadatas = [{"source": chunk['source'], "chunk_index": chunk['chunk_index'],"token_count": chunk['token_count']} for chunk in chunks]
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
        logger.info(
            f"Stored batch {i//batch_size + 1} | "
            f"chunks {i} to {min(i+batch_size, len(chunks))}"
        )
    logger.info(f"Embedding complete | total stored: {len(chunks)}")

def search_similar_chunks(query: str, n_results: int = 5) -> list[dict]:
    """
    Search for chunks similar to the query.
    This is the retrieval step of RAG.

    1. Embed the query into a vector
    2. Find the n_results most similar vectors in Chroma
    3. Return the original text chunks with their similarity scores
    """
    logger.info(f"Searching for: '{query[:50]}...'")

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
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
        f"found {len(chunks)} chunks | "
        f"top similarity: {chunks[0]['similarity']:.3f}"
    )

    return chunks


def get_collection_count() -> int:
    """Return how many chunks are stored in Chroma."""
    return collection.count()
    

