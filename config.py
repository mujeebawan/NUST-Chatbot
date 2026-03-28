"""Configuration for NUST Admissions Chatbot."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PDF_DATA_DIR = os.path.join(DATA_DIR, "pdfs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Embedding model (small, fast, runs on CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings
LLM_MODEL_PATH = os.path.join(MODEL_DIR, "model.gguf")
LLM_CONTEXT_LENGTH = 2048
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.1
LLM_N_THREADS = 4  # Conservative for i5 13th gen

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4

# UI settings
APP_TITLE = "NUST Admissions Guide"
APP_DESCRIPTION = "Your offline NUST admissions assistant - accurate, fast, and transparent."

# Create directories
for d in [DATA_DIR, RAW_DATA_DIR, PDF_DATA_DIR, VECTOR_STORE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
