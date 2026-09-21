# Zepto Support Assistant

A small GenAI support assistant that answers questions about Zepto policies using a local knowledge base, embeddings, ChromaDB retrieval, LangGraph routing, structured Pydantic output, and a FastAPI API.

The project runs in deterministic offline mock mode by default, so no API key or network-based LLM call is required.

---

## Architecture

```text
8 Zepto policy documents
        |
        v
Document loading
        |
        v
Chunking
(250 characters, 40 overlap)
        |
        v
Sentence Transformers
(all-MiniLM-L6-v2)
        |
        v
ChromaDB
22 stored vectors
        |
        v
User query
        |
        v
LangGraph
        |
        +----------------------+
        |                      |
        v                      v
classify_intent          general_question
        |                      |
        v                      v
policy_question          direct_answer
        |
        v
retrieve_and_answer
        |
        +-----------+
                    |
                    v
          Pydantic structured response
                    |
                    v
              FastAPI /ask
                    |
                    v
                  JSON