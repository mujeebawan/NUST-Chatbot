"""RAG pipeline: retrieve context and generate answers using local LLM."""
import os
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_PATH,
    LLM_CONTEXT_LENGTH, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_N_THREADS, LLM_N_BATCH, TOP_K_RESULTS
)


SYSTEM_PROMPT = """You are the NUST Admissions Guide — an accurate assistant for students applying to the National University of Sciences & Technology (NUST), Islamabad.

STRICT RULES:
1. Answer ONLY from the provided context. Do not add outside knowledge.
2. If the context does not cover the question, say: "I don't have this information in my database. Please visit nust.edu.pk or contact ugadmissions@nust.edu.pk for the latest details."
3. NEVER invent facts, fees, dates, percentages, or cutoff scores.
4. Keep answers concise and direct — use bullet points for lists.
5. For fees/dates/numbers, add: "Please verify from official sources as these may change."
6. For non-NUST questions, say: "I only handle NUST admissions queries."
7. If multiple context chunks give conflicting info, state what each source says rather than picking one.
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
            n_batch=LLM_N_BATCH,
            use_mmap=True,
            use_mlock=False,
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
        # Trim context to fit within token budget (leave room for prompt + response)
        max_context_chars = 2500
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context:
{context}

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
