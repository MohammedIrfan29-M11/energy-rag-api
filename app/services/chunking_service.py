import logging
import re
import tiktoken
from app.core.config import CHUNK_OVERLAP,CHUNK_SIZE

logger = logging.getLogger('app.services.chunking')
encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    """Counts the number of tokens in a given text using tiktoken."""
    return len(encoder.encode(text))

def clean_text(text: str) -> str:
    
    text = re.sub(r'-\n', '', text)

    
    text = re.sub(r'\n+', ' ', text)

   
    text = re.sub(r' {2,}', ' ', text)

    text = text.strip()
    return text


def split_into_paragraphs(text: str) -> list[str]:
    """
    For PDFs with no paragraph markers we split by sentences
    and group them into paragraph-sized chunks of ~5 sentences.
    This gives us natural semantic units to work with.
    """
    
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    paragraphs = []
    group_size = 5
    for i in range(0, len(sentences), group_size):
        group = sentences[i:i + group_size]
        paragraph = ' '.join(group)
        if len(paragraph.strip()) > 50:
            paragraphs.append(paragraph)

    return paragraphs

def create_chunks(paragraphs: list[str], source: str) -> list[dict]:
    """Convert paragraphs into chunks with overlap.

    Why overlap? Imagine a paragraph boundary falls in the middle
    of an important sentence. Without overlap, that sentence gets
    split — part in chunk N, part in chunk N+1. Neither chunk
    contains the complete thought. Overlap ensures important
    information at boundaries appears in both adjacent chunks.

    Each chunk is a dict with:
    - text: the actual content
    - source: filename it came from
    - chunk_index: position in document
    - token_count: how many tokens it contains"""

    chunks = []
    current_chunk_text = ""
    chunk_index = 0
    current_token_count = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        if paragraph_tokens > CHUNK_SIZE:
            if current_chunk_text:
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "source": source,
                    "chunk_index": chunk_index,
                    "token_count": current_token_count
                })
                chunk_index += 1
                overlap_text = get_overlap_text(current_chunk_text, CHUNK_OVERLAP)
                current_chunk_text = overlap_text
                current_token_count = count_tokens(overlap_text)

            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                if current_token_count + sentence_tokens > CHUNK_SIZE:
                    chunks.append({
                        "text": current_chunk_text.strip(),
                        "source": source,
                        "chunk_index": chunk_index,
                        "token_count": current_token_count
                    })
                    chunk_index += 1
                    overlap_text = get_overlap_text(current_chunk_text, CHUNK_OVERLAP)
                    current_chunk_text = overlap_text
                    current_token_count = count_tokens(overlap_text)
        
        else:
            if current_token_count + paragraph_tokens > CHUNK_SIZE:
                if current_chunk_text:
                    chunks.append({
                        "text": current_chunk_text.strip(),
                        "source": source,
                        "chunk_index": chunk_index,
                        "token_count": current_token_count
                    })
                    chunk_index += 1
                    overlap_text = get_overlap_text(current_chunk_text, CHUNK_OVERLAP)
                    current_chunk_text = overlap_text
                    current_token_count = count_tokens(overlap_text)
            
            current_chunk_text += " " + paragraph
            current_token_count += paragraph_tokens

    if current_chunk_text.strip():
        chunks.append({
            "text": current_chunk_text.strip(),
            "source": source,
            "chunk_index": chunk_index,
            "token_count": current_token_count
        })
    logger.info("Chunking complete | "
                f"source: {source} | "
                f"paragraphs: {len(paragraphs)} | "
                f"chunks: {len(chunks)}"
                )
    return chunks

def get_overlap_text(text, overlap_tokens):
    """Returns the last N tokens of the text as a string for overlap."""
    words = text.split()
    overlap_words=int(overlap_tokens / 1.3)
    if len(words) <= overlap_words:
        return text
    else:
        return " ".join(words[-overlap_words:])
    
def split_into_sentences(text):
    """Splits a paragraph into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]