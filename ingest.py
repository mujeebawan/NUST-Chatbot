"""Data ingestion: load documents, chunk them, and build FAISS vector store."""
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from config import (
    RAW_DATA_DIR, PDF_DATA_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP
)


def load_text_files(directory: str) -> list[Document]:
    """Load all .txt and .md files from directory."""
    docs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
    return docs


def load_json_files(directory: str) -> list[Document]:
    """Load structured JSON data files."""
    docs = []
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    content = "\n".join(f"{k}: {v}" for k, v in item.items())
                else:
                    content = str(item)
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
        elif isinstance(data, dict):
            content = "\n".join(f"{k}: {v}" for k, v in data.items())
            docs.append(Document(
                page_content=content,
                metadata={"source": filename}
            ))
    return docs


def load_pdf_files(directory: str) -> list[Document]:
    """Load PDF files."""
    from pypdf import PdfReader
    docs = []
    for filename in os.listdir(directory):
        if not filename.endswith(".pdf"):
            continue
        filepath = os.path.join(directory, filename)
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i + 1}
                ))
    return docs


def build_vector_store():
    """Load all data, chunk it, and create FAISS index."""
    print("Loading documents...")
    docs = []
    docs.extend(load_text_files(RAW_DATA_DIR))
    docs.extend(load_json_files(RAW_DATA_DIR))
    docs.extend(load_pdf_files(RAW_DATA_DIR))
    if os.path.exists(PDF_DATA_DIR):
        docs.extend(load_pdf_files(PDF_DATA_DIR))

    if not docs:
        print(f"No documents found in {RAW_DATA_DIR}")
        print("Please add .txt, .md, .json, or .pdf files with NUST admissions data.")
        return

    print(f"Loaded {len(docs)} documents")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\nQ:", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print(f"Building embeddings with {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"Vector store saved to {VECTOR_STORE_DIR}")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    build_vector_store()
