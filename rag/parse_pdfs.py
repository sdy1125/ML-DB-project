"""Backward-compatible entrypoint for PDF ingestion.

The implementation lives in `src.sdg16_pipeline.data_ingestion.pdf_ingestion`.
"""

from src.sdg16_pipeline.data_ingestion.pdf_ingestion import *  # noqa: F401,F403
from src.sdg16_pipeline.data_ingestion.pdf_ingestion import main


if __name__ == "__main__":
    main()

