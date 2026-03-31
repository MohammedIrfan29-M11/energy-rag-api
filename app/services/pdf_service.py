import logging
import pathlib
from pypdf import PdfReader 
from app.services.chunking_service import clean_text, split_into_paragraphs, create_chunks

logger = logging.getLogger('app.services.pdf')

def extract_text_from_pdf(file_path):
    """Extract raw text from a PDF file."""
    logger.info(f"Extracting text from PDF: {file_path}")
    reader = PdfReader(file_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    logger.info(
        f"Extraction complete | "
        f"pages: {len(reader.pages)} | "
        f"characters: {len(full_text)}"
    )
    return full_text

def process_pdf(file_path):
    """Process a PDF file and return cleaned text chunks."""
    source = pathlib.Path(file_path).name
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text.strip():
        raise ValueError(f"No text extracted from {source}. Please check the file.")
    clean=clean_text(raw_text)  
    paragraphs = split_into_paragraphs(clean)
    chunks = create_chunks(paragraphs, source)
    logger.info(
        f"PDF processed | "
        f"source: {source} | "
        f"chunks: {len(chunks)}"
    )
    return chunks

