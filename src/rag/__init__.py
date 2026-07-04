# RAG Pipeline Module
#
# Core components:
#   - embedder: document loading, chunking, and local embeddings
#   - llm_provider: Groq/Ollama abstraction (same LangChain interface)
#   - retriever: vector store search (coming Day 3)
#   - generator: LLM answer generation (coming Day 4)
#   - graph: LangGraph wiring (coming Day 5)
