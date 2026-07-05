"""Backward-compatible entrypoint for RAG vector indexing."""

from src.sdg16_pipeline.phase6_rag_llm_policy.index_documents import *  # noqa: F401,F403
from src.sdg16_pipeline.phase6_rag_llm_policy.index_documents import main


if __name__ == "__main__":
    main()

