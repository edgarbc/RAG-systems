# RAG-systems

[![CI](https://github.com/edgarbc/RAG-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/edgarbc/RAG-systems/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edgarbc/RAG-systems/blob/main/RAGAs/RAGAS_demo.ipynb)

> **TL;DR**: A hands-on portfolio project demonstrating Retrieval-Augmented Generation (RAG) systems—from building retrievers to evaluating generation quality.

![RAG Architecture](https://via.placeholder.com/800x400?text=RAG+Architecture+Diagram)
*Screenshot placeholder: Add your own architecture diagram here*

---

## 🎯 Problem Description

Large Language Models (LLMs) are powerful but suffer from:
- **Hallucinations**: Generating plausible but incorrect information
- **Stale knowledge**: Training data has a cutoff date
- **Lack of domain specificity**: General models lack specialized knowledge

**Retrieval-Augmented Generation (RAG)** addresses these issues by:
1. Retrieving relevant documents from a knowledge base
2. Providing this context to the LLM at inference time
3. Grounding generated responses in verifiable sources

This project demonstrates how to build, evaluate, and deploy RAG systems.

---

## 📦 What's Included

| Component | Description |
|-----------|-------------|
| `demo/run_rag_demo.py` | Minimal end-to-end RAG demo script |
| `demo/sample_corpus.json` | Small corpus (5 documents) for testing |
| `RAGAs/RAGAS_demo.ipynb` | Jupyter notebook for RAG evaluation with RAGAS |
| `docs/DESIGN.md` | Architecture notes and design decisions |
| `.github/workflows/ci.yml` | CI smoke tests |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/edgarbc/RAG-systems.git
cd RAG-systems

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Demo

```bash
python demo/run_rag_demo.py
```

### Expected Output

```
============================================================
RAG Demo - Retrieval-Augmented Generation
============================================================

Loading corpus from: demo/sample_corpus.json
Loaded 5 documents

Using BM25-based retrieval (rank_bm25)
Tip: Set USE_EMBEDDINGS_RETRIEVER=1 to use sentence-transformers
Building BM25 index...

------------------------------------------------------------
Query: What is RAG and how does it work?
------------------------------------------------------------

Top 2 Retrieved Documents:
  1. [1.6549] Introduction to RAG
     Retrieval-Augmented Generation (RAG) is a technique that combines...
  2. [0.9651] RAG vs Fine-tuning
     Unlike fine-tuning which modifies model weights, RAG keeps...

[Response]: [PLACEHOLDER] Would generate answer using context from: Introduction to RAG, RAG vs Fine-tuning

============================================================
Demo completed successfully!
============================================================
```

The demo:
1. ✅ Loads a sample corpus from JSON
2. ✅ Builds a BM25 index (or embedding index with `USE_EMBEDDINGS_RETRIEVER=1`)
3. ✅ Retrieves top-k relevant passages
4. ✅ Formats a RAG prompt (placeholder for LLM generation)

> **Note**: The demo uses a placeholder generator to avoid external API calls. See `demo/run_rag_demo.py` for where to plug in a real LLM.

---

## 📓 Notebooks

### RAGAS Evaluation Demo
Evaluate RAG systems using the RAGAS framework with metrics like:
- **Context Precision**: Relevance of retrieved documents
- **Context Recall**: Coverage of necessary information
- **Faithfulness**: How well answers are grounded in context
- **Answer Correctness**: Accuracy compared to ground truth

👉 [Open in Colab](https://colab.research.google.com/github/edgarbc/RAG-systems/blob/main/RAGAs/RAGAS_demo.ipynb)

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Retriever  │────▶│  Generator  │────▶ Response
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Document   │
                    │   Index     │
                    └─────────────┘
```

For detailed architecture notes, see [docs/DESIGN.md](docs/DESIGN.md).

---

## 📚 Learn More

- [RAGAS Documentation](https://docs.ragas.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built with ❤️ by [Edgar Bermudez](https://github.com/edgarbc)*
