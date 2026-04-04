import logging
from anthropic import Anthropic
from app.core.config import ANTHROPIC_API_KEY, MAX_CHUNKS_PER_QUERY
from app.services.embedding_service import search_similar_chunks

logger=logging.getLogger('app.services.rag')
client=Anthropic(api_key=ANTHROPIC_API_KEY)

SIMILARITY_THRESHOLD=0.65

RAG_SYSTEM_PROMPT="""You are an expert energy sector analyst assistant.
You answer questions about energy policy, regulations, and market trends
based EXCLUSIVELY on the document sections provided to you.

Your rules:
1. ONLY use information from the provided context sections
2. NEVER use your training knowledge to answer factual questions
3. If the answer is not in the context — say exactly:
   "I cannot find this information in the provided documents."
4. Always cite which section(s) you used
5. Be precise — use exact numbers and figures from the documents
6. If context sections partially answer the question — say what you
   found and what is missing"""

def build_context(chunks: list[dict]) -> str:
    """Build a context string from the most relevant chunks.
    We include the chunk index and source for traceability."""
    if not chunks:
        return "No relevant information found in the documents."
    
    context_parts=[]
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Section {i+1}] Source: {chunk['source']} "
            f"(similarity: {chunk['similarity']:.2f})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)

def build_rag_prompt(question: str,context: str)->str:
    """Build the full RAG prompt combining context and question.
    Structure matters — context first, then question, then instructions."""
    return f"""Here are relevant inform from energy documents.
    {context}
    ---
    Question:{question}
    Instructions:Answer only using the context of the energy documents.
    Cite the section numbers you used (e.g., "According to Section 1...").
    If the information is not in the sections above, say so clearly."""

def query_rag(question:str,history:list[dict]=[],n_chunks:int=MAX_CHUNKS_PER_QUERY)->str:
    """Main RAG query function.
    1. Search for relevant chunks based on the question
    2. Build a context string from those chunks
    3. Construct the full prompt with instructions
    4. Send to Anthropic and return the answer"""

    logger.info(f"RAG query | question: '{question[:60]}...'")
    chunks=search_similar_chunks(question,n_results=n_chunks)

    
    relevant_chunks=[c for c in chunks if c['similarity'] >= SIMILARITY_THRESHOLD]
    logger.info(f"RAG query | found {len(relevant_chunks)} relevant chunks  | total retrieved: {len(chunks)}")
    if not relevant_chunks:
        logger.warning(
            f"No chunks above threshold {SIMILARITY_THRESHOLD} | "
            f"top similarity: {chunks[0]['similarity']:.3f}"
        )
        return {
            "answer": "I cannot find relevant information about this "
                     "in the provided documents. The most similar content "
                     f"had a similarity score of {chunks[0]['similarity']:.2f} "
                     f"which is below the minimum threshold of {SIMILARITY_THRESHOLD}.",
            "citations": [],
            "retrieved_chunks": chunks,
            "history": history
        }
    context=build_context(relevant_chunks)
    rag_prompt=build_rag_prompt(question,context)
    messages=history+[{"role":"user","content":rag_prompt}]
    logger.info("RAG query | built prompt with context | sending to Anthropic...")

    response=client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=RAG_SYSTEM_PROMPT,
        messages=messages
    )

    answer=response.content[0].text
    citations=[
        {
            "source": chunk['source'],
            "chunk_index": chunk['chunk_index'],
            "similarity": chunk['similarity']
        }
        for chunk in relevant_chunks
    ]
    updated_history=history+[
        {"role":"user","content":rag_prompt},
        {"role":"assistant","content":answer}
    ]
    logger.info(f"RAG complete | "
        f"answer_length: {len(answer)} | "
        f"citations: {len(citations)}")

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": relevant_chunks,
        "history": updated_history
    }


