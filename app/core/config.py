import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set in .env file")


CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_QUERY = 5

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "energy_documents"

MAX_CONTEXT_TOKENS = 4000
MAX_HISTORY_LENGTH = 10

UPLOAD_DIR = "Documents"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx'}