"""RAG pipeline: retrieve context and generate answers using local LLM."""
import os
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_PATH,
    LLM_CONTEXT_LENGTH, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_N_THREADS, TOP_K_RESULTS
)


SYSTEM_PROMPT = """You are the NUST Admissions Guide, an accurate and helpful assistant for students seeking information about admission to the National University of Sciences and Technology (NUST), Islamabad, Pakistan.

Rules:
1. ONLY answer questions related to NUST admissions using the provided context.
2. If the context does not contain enough information to answer confidently, say: "I don't have enough information to answer this accurately. Please check the official NUST website or contact the admissions office."
3. Never make up facts, numbers, dates, or fees. Only state what is supported by the context.
4. Be concise, clear, and helpful.
5. When citing specific numbers (fees, dates, percentages), mention that these may be subject to change and students should verify from official sources.
6. If a question is not about NUST admissions, politely redirect: "I'm designed to help with NUST admissions queries. Could you ask me something about NUST admissions?"
"""


class NUSTAdmissionsBot:
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.embeddings = None

    def load(self):
        """Load all components."""
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("Loading vector store...")
        if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
            raise FileNotFoundError(
                "Vector store not found. Run 'python ingest.py' first."
            )
        self.vector_store = FAISS.load_local(
            VECTOR_STORE_DIR, self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("Loading LLM...")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {LLM_MODEL_PATH}. "
                "Please download a GGUF model and place it there."
            )
        self.llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_CONTEXT_LENGTH,
            n_threads=LLM_N_THREADS,
            verbose=False,
        )
        print("All components loaded successfully!")

    def retrieve(self, query: str) -> str:
        """Retrieve relevant context from vector store."""
        results = self.vector_store.similarity_search(query, k=TOP_K_RESULTS)
        context_parts = []
        sources = set()
        for doc in results:
            context_parts.append(doc.page_content)
            sources.add(doc.metadata.get("source", "unknown"))
        context = "\n\n".join(context_parts)
        return context, list(sources)

    def generate(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context from NUST admissions documents:
---
{context}
---

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        response = self.llm(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()

    def ask(self, query: str) -> dict:
        """Full RAG pipeline: retrieve + generate."""
        context, sources = self.retrieve(query)
        answer = self.generate(query, context)
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context[:500] + "..." if len(context) > 500 else context
        }
