# EnergyRAG — Energy Policy Document Intelligence

> Query EU energy regulations and IEA policy documents using Claude AI. Get grounded answers with citations showing exactly which document sections were used — zero hallucination.

🔴 **Live API:** https://energy-rag-api-production.up.railway.app  
📖 **Interactive Docs:** https://energy-rag-api-production.up.railway.app/docs  
💻 **GitHub:** https://github.com/MohammedIrfan29-M11/energy-rag-api

---

## What it does

Upload any energy policy PDF → ask questions in plain English → get grounded answers with citations.

```bash
# Upload a document
curl -X POST https://energy-rag-api-production.up.railway.app/documents/upload \
  -F "file=@EU_Renewable_Energy_Directive_2023.pdf"

# Ask a question
curl -X POST https://energy-rag-api-production.up.railway.app/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the EU renewable energy targets for 2030?", "history": []}'
```

```json
{
  "answer": "According to Section 4, the European Union has set a target of 42.5% renewables share of gross final consumption by 2030 within the Fit for 55 package. Germany has committed to 30 GW of offshore wind and the Netherlands to 21 GW by 2030.",
  "citations": [
    {
      "source": "WorldEnergyOutlook2023.pdf",
      "chunk_index": 736,
      "similarity": 0.727
    }
  ],
  "chunks_used": 5
}
```

---

## Why I built this from scratch

Most RAG tutorials use LangChain which abstracts everything away. I built every component manually so I can explain every decision:

- **Why sentence-based chunking?** The IEA PDF has no paragraph markers when extracted — only single newlines. Sentence boundaries were the most reliable semantic unit available in the raw text.
- **Why 0.65 similarity threshold?** Below that, retrieved chunks are only topically related — not actually answering the question. Sending weak chunks to Claude causes hallucination. A "cannot find" response is safer than a confidently wrong answer.
- **Why store history as original question, not RAG prompt?** The RAG prompt contains thousands of tokens of context. Storing it in history would overflow the context window within 2-3 turns.

---

## Architecture

```
INDEXING PHASE (once per document):
PDF Upload → pypdf extraction → text cleaning → sentence chunking
→ ChromaDB embedding → persistent vector storage

RETRIEVAL PHASE (every query):
User question → embed query → cosine similarity search (853 chunks)
→ filter by threshold (≥0.65) → build grounded prompt
→ Claude Sonnet API → grounded answer + citations
```

### Key design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding | ChromaDB DefaultEmbeddingFunction | No PyTorch dependency — lightweight deployment |
| Vector DB | ChromaDB | Local persistence, no managed service needed for portfolio |
| Chunking | Sentence-based, 300 tokens, 50-token overlap | IEA PDFs have no paragraph markers |
| LLM | Claude Sonnet 4.5 | Best grounding behavior, reliable citation following |
| Similarity threshold | 0.65 | Prevents hallucination on weak retrievals |
| Framework | FastAPI (no LangChain) | Full understanding of every component |

---

## Tech Stack

- **LLM:** Claude Sonnet 4.5 (Anthropic)
- **Backend:** FastAPI + Python 3.12
- **Vector Database:** ChromaDB
- **PDF Processing:** pypdf + tiktoken
- **Containerization:** Docker + Docker Compose
- **Deployment:** Railway
- **Validation:** Pydantic v2

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System status + chunks indexed |
| POST | `/documents/upload` | Upload and index a PDF |
| GET | `/documents` | List all indexed documents |
| POST | `/rag/query` | Query documents with RAG |

Full interactive documentation at `/docs`.

---

## Quick Start

### Local development

```bash
git clone https://github.com/MohammedIrfan29-M11/energy-rag-api
cd energy-rag-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```

Run:
```bash
uvicorn app.main:app --reload
```

### Docker

```bash
docker compose up --build
```

Upload a PDF and start querying:
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@your_document.pdf"
```

---

## Project Structure

```
app/
├── api/routes.py              # REST endpoints
├── services/
│   ├── pdf_service.py         # PDF extraction pipeline
│   ├── chunking_service.py    # Sentence-based chunking with overlap
│   ├── embedding_service.py   # ChromaDB vector storage and search
│   └── rag_service.py         # RAG orchestration + grounded prompts
├── core/
│   ├── config.py              # Centralised configuration
│   └── logging_config.py      # Structured logging
└── main.py                    # FastAPI application
```

---

## Known Limitations and Production Improvements

| Current | Production |
|---------|-----------|
| ChromaDB local storage (ephemeral on Railway) | Pinecone or Qdrant managed vector DB |
| Single PDF format supported | Multi-format: Word, HTML, markdown |
| No authentication | JWT or API key authentication |
| In-memory rate limiting | Redis-backed distributed rate limiting |
| Basic sentence chunking | Recursive chunking with topic detection |
| No query rewriting | Claude Haiku rewrites follow-up questions before embedding |

---


This project is part of a structured self-learning plan to transition into an AI Engineer role in Europe. Built without LangChain or LlamaIndex — every component implemented from scratch to ensure deep understanding of the RAG pipeline.

**Related project:** [Finn — FinTech AI Assistant](https://github.com/MohammedIrfan29-M11/Claudeai-chatbot-api)
