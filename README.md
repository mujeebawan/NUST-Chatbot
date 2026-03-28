# NUST Admissions Guide - Offline Chatbot

An offline, RAG-powered admissions chatbot for NUST (National University of Sciences and Technology) that runs entirely on student hardware — no internet, no GPU, no cloud APIs.

Built for the **NUST Islamabad Local Chatbot Competition 2026**.

## Architecture

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│  Embedding  │────▶│  FAISS Vector DB │
│  (MiniLM)   │     │  (Local Index)   │
└─────────────┘     └──────┬───────────┘
                           │ Top-K chunks
                           ▼
                    ┌──────────────────┐
                    │   Qwen2.5-3B   │
                    │  (Q4_K_M GGUF)   │
                    │   CPU Inference  │
                    └──────┬───────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │  Gradio Chat UI  │
                    └──────────────────┘
```

## Design Tradeoffs

| Decision | Choice | Why |
|----------|--------|-----|
| LLM | Qwen2.5-3B Q4_K_M | Best quality-to-size ratio for CPU; 2.0GB fits in 8GB RAM with room for embeddings + OS |
| Embeddings | all-MiniLM-L6-v2 | Only 80MB, fast on CPU, excellent semantic search quality |
| Vector Store | FAISS | Zero-dependency, fast similarity search, no server needed |
| Quantization | Q4_K_M | Sweet spot between quality and speed; negligible quality loss vs FP16 |
| Context | 2048 tokens | Enough for RAG context + answer; keeps inference fast |
| Retrieval | Top-4 chunks | Balances context richness vs. inference speed |
| UI | Gradio | Clean, responsive, zero-config, works offline |

## Hardware Requirements

- **RAM:** 4-6 GB (model ~2.0GB + embeddings ~80MB + OS overhead)
- **CPU:** Any modern CPU (optimized for i5 13th gen)
- **GPU:** Not required
- **Storage:** ~3 GB for model + data
- **Internet:** Only needed for initial setup (downloading model)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the LLM Model
```bash
python download_model.py
```
This downloads Qwen2.5-3B (Q4_K_M, ~2.0GB) from HuggingFace.

### 3. Build the Knowledge Base
```bash
python ingest.py
```
This processes the admissions data and builds the FAISS vector index.

### 4. Launch the Chatbot
```bash
python app.py
```
Opens at http://127.0.0.1:7860

### 5. Run Benchmarks (Optional)
```bash
python benchmark.py
```

## Adding More Data

Place `.txt`, `.md`, `.json`, or `.pdf` files in `data/raw/` and re-run:
```bash
python ingest.py
```

## Project Structure

```
ChatBot/
├── app.py              # Gradio UI
├── rag.py              # RAG pipeline (retrieval + generation)
├── ingest.py           # Data ingestion and vector store builder
├── config.py           # All configuration in one place
├── download_model.py   # Model downloader
├── benchmark.py        # Performance benchmarking
├── requirements.txt    # Python dependencies
├── data/
│   ├── raw/            # Source documents (txt, md, json, pdf)
│   └── vector_store/   # FAISS index (auto-generated)
└── models/
    └── model.gguf      # LLM model (auto-downloaded)
```

## Key Features

- **100% Offline:** After initial setup, no internet needed
- **Transparent:** Shows source documents and response time for every answer
- **Honest:** Explicitly states when it doesn't have enough information
- **Fast:** Optimized for CPU inference with quantized model
- **Extensible:** Add new data by dropping files in `data/raw/`

## Benchmarks

Run `python benchmark.py` to generate benchmarks on your hardware. Typical results on i5 13th gen:

| Metric | Value |
|--------|-------|
| Model load time | ~5-10s |
| Average response time | ~5-15s |
| RAM usage | ~3-5 GB |
| Knowledge base size | ~50 chunks |

---

*Built for the NUST Islamabad Local Chatbot Competition 2026*
*Runs entirely offline on student hardware (8GB RAM, Core i5, no GPU)*
