import sys
sys.path.append('.')

from app.core.logging_config import setup_logging
from app.services.pdf_service import process_pdf, extract_text_from_pdf
from app.services.chunking_service import clean_text, split_into_paragraphs

setup_logging()


raw = extract_text_from_pdf('documents/WorldEnergyOutlook2023.pdf')
print(f"\nRAW TEXT SAMPLE:")
print(repr(raw[1000:1200])) 

cleaned = clean_text(raw)
print(f"\nCLEANED TEXT SAMPLE:")
print(repr(cleaned[1000:1200]))

paragraphs = split_into_paragraphs(cleaned)
print(f"\nParagraph count: {len(paragraphs)}")
print(f"First paragraph preview: {paragraphs[0][:200] if paragraphs else 'EMPTY'}")


chunks = process_pdf('documents/WorldEnergyOutlook2023.pdf')

print(f"\n{'='*50}")
print(f"Total chunks created: {len(chunks)}")
print(f"{'='*50}\n")

for i, chunk in enumerate(chunks[:3]):
    print(f"CHUNK {i+1}:")
    print(f"Source: {chunk['source']}")
    print(f"Index: {chunk['chunk_index']}")
    print(f"Tokens: {chunk['token_count']}")
    print(f"Text preview: {chunk['text'][:200]}...")
    print(f"{'-'*50}\n")

token_counts = [c['token_count'] for c in chunks]
print(f"Token stats:")
print(f"  Min: {min(token_counts)}")
print(f"  Max: {max(token_counts)}")
print(f"  Avg: {sum(token_counts)//len(token_counts)}")