#!/usr/bin/env python3
"""
RAG Demo Script
===============

A minimal end-to-end Retrieval-Augmented Generation demo that:
1. Loads a sample corpus from JSON
2. Builds an index (using BM25 or sentence-transformers embeddings)
3. Retrieves top-k relevant passages for a query
4. Formats a prompt with retrieved context (placeholder for LLM generation)

This script is designed to run in CI without external API calls.
By default, it uses BM25 (lightweight, no network required).
Set USE_EMBEDDINGS_RETRIEVER=1 to use sentence-transformers instead.
"""

import json
import os
import sys
from pathlib import Path

# Import numpy (needed for both retrievers)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import BM25 (preferred for CI - no network required)
try:
    from rank_bm25 import BM25Okapi
    USE_BM25 = True
except ImportError:
    USE_BM25 = False

# Try to import sentence-transformers for embedding-based retrieval
# Only used if explicitly requested via environment variable
try:
    from sentence_transformers import SentenceTransformer
    USE_EMBEDDINGS = True
except ImportError:
    USE_EMBEDDINGS = False


def load_corpus(corpus_path: str) -> list:
    """Load documents from a JSON file."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


class EmbeddingRetriever:
    """Retriever using sentence-transformers embeddings."""
    
    def __init__(self, documents: list, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding retriever.
        
        Args:
            documents: List of document dicts with 'content' field
            model_name: Sentence-transformer model name
        """
        self.documents = documents
        self.contents = [doc["content"] for doc in documents]
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print("Encoding documents...")
        self.doc_embeddings = self.model.encode(self.contents, show_progress_bar=False)
    
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Search query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        
        # Compute cosine similarities
        similarities = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results


class BM25Retriever:
    """Retriever using BM25 algorithm (sparse retrieval)."""
    
    def __init__(self, documents: list):
        """
        Initialize the BM25 retriever.
        
        Args:
            documents: List of document dicts with 'content' field
        """
        self.documents = documents
        self.contents = [doc["content"] for doc in documents]
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_corpus = [content.lower().split() for content in self.contents]
        
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Search query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(scores[idx])))
        
        return results


def format_rag_prompt(query: str, retrieved_docs: list) -> str:
    """
    Format a RAG prompt with retrieved context.
    
    This is a placeholder for where you would call an LLM API.
    In production, replace this with actual LLM generation.
    
    Args:
        query: User's question
        retrieved_docs: List of (document, score) tuples
        
    Returns:
        Formatted prompt string
    """
    context_parts = []
    for i, (doc, score) in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc['title']}: {doc['content']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    return prompt


def generate_placeholder_response(query: str, retrieved_docs: list) -> str:
    """
    Generate a placeholder response (no actual LLM call).
    
    In production, replace this function with actual LLM API calls.
    Example integrations:
    - OpenAI: openai.ChatCompletion.create(...)
    - Anthropic: anthropic.messages.create(...)
    - Local models: Use transformers pipeline or llama.cpp
    
    Args:
        query: User's question
        retrieved_docs: List of (document, score) tuples
        
    Returns:
        Placeholder response string
    """
    # For CI/demo purposes, we just acknowledge the retrieved context
    doc_titles = [doc["title"] for doc, _ in retrieved_docs]
    return f"[PLACEHOLDER] Would generate answer using context from: {', '.join(doc_titles)}"


def main():
    """Main demo function."""
    print("=" * 60)
    print("RAG Demo - Retrieval-Augmented Generation")
    print("=" * 60)
    print()
    
    # Check if numpy is available (required for both retrievers)
    if not HAS_NUMPY:
        print("ERROR: numpy is required but not installed!")
        print("Please run: pip install numpy")
        sys.exit(1)
    
    # Determine corpus path (relative to this script)
    script_dir = Path(__file__).parent
    corpus_path = script_dir / "sample_corpus.json"
    
    if not corpus_path.exists():
        print(f"ERROR: Corpus file not found at {corpus_path}")
        sys.exit(1)
    
    # Load corpus
    print(f"Loading corpus from: {corpus_path}")
    documents = load_corpus(str(corpus_path))
    print(f"Loaded {len(documents)} documents")
    print()
    
    # Check if user explicitly wants embeddings
    use_embeddings_retriever = os.environ.get("USE_EMBEDDINGS_RETRIEVER", "0") == "1"
    
    # Initialize retriever
    # Default: BM25 (lightweight, works offline)
    # Optional: Embeddings (requires network to download model)
    if use_embeddings_retriever and USE_EMBEDDINGS:
        print("Using embedding-based retrieval (sentence-transformers)")
        print("Note: This requires downloading a model from HuggingFace")
        try:
            retriever = EmbeddingRetriever(documents)
        except Exception as e:
            print(f"WARNING: Failed to initialize embedding retriever: {e}")
            print("Falling back to BM25...")
            if USE_BM25:
                retriever = BM25Retriever(documents)
            else:
                print("ERROR: No retrieval backend available!")
                sys.exit(1)
    elif USE_BM25:
        print("Using BM25-based retrieval (rank_bm25)")
        print("Tip: Set USE_EMBEDDINGS_RETRIEVER=1 to use sentence-transformers")
        retriever = BM25Retriever(documents)
    elif USE_EMBEDDINGS:
        print("Using embedding-based retrieval (sentence-transformers)")
        try:
            retriever = EmbeddingRetriever(documents)
        except Exception as e:
            print(f"ERROR: Failed to initialize retriever: {e}")
            sys.exit(1)
    else:
        print("ERROR: No retrieval backend available!")
        print("Please install rank_bm25 or sentence-transformers")
        sys.exit(1)
    
    print()
    
    # Demo queries
    queries = [
        "What is RAG and how does it work?",
        "How do you evaluate RAG systems?",
        "What are the main components of a RAG architecture?",
    ]
    
    top_k = 2
    
    for query in queries:
        print("-" * 60)
        print(f"Query: {query}")
        print("-" * 60)
        
        # Retrieve relevant documents
        results = retriever.retrieve(query, top_k=top_k)
        
        print(f"\nTop {top_k} Retrieved Documents:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.4f}] {doc['title']}")
            print(f"     {doc['content'][:100]}...")
        
        # Format prompt (shows what would be sent to LLM)
        prompt = format_rag_prompt(query, results)
        print(f"\n[Formatted RAG Prompt Preview]")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        # Generate placeholder response
        response = generate_placeholder_response(query, results)
        print(f"\n[Response]: {response}")
        print()
    
    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    
    # Exit with success code for CI
    sys.exit(0)


if __name__ == "__main__":
    main()
