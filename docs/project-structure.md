# SDG16 project structure

Project được chia theo workflow 6 phase trong sơ đồ, nhưng vẫn giữ các entrypoint cũ để không phá lệnh Docker/test hiện tại.

```text
src/sdg16_pipeline/
├── data_ingestion/                 # CSV/PDF/knowledge ingestion
│   └── pdf_ingestion.py             # scan PDF 5 nhóm -> chunks/text exports
├── phase1_panel_ols/                # Spark prepare + linear baseline artifacts
│   ├── config.py
│   ├── run_pipeline.py
│   └── spark_jobs/
│       ├── prepare_data.py
│       └── train_linear.py
├── phase2_xgboost_ablation/         # reserved: XGBoost, tuning, ablation
├── phase3_gru_forecasting/          # reserved: GRU 2025-2030 scenarios
├── phase4_shap_explanation/         # XGBoost tree-contribution explanation
│   └── run_shap_sdg16_vietnam.py
├── phase5_subnational_drilldown/    # reserved: PAPI/PCI province drill-down
├── phase6_rag_llm_policy/           # RAG indexing + embeddings
│   ├── embeddings.py
│   └── index_documents.py
└── final_output/                    # reserved: final report/table composition
```

Runtime/delivery layer vẫn ở vị trí riêng:

```text
apps/
├── api/                             # FastAPI ML/RAG/final-insight API
├── backend/                         # Spring Boot gateway/API shell
└── frontend/                        # Vue web UI

data/                                # raw/clean/knowledge data
artifacts/                           # model metadata + SHAP outputs
infra/                               # Dockerfiles, nginx
configs/                             # project.yaml and env-style config
tests/                               # regression tests
```

Compatibility wrappers vẫn còn:

```text
scripts/run_shap_sdg16_vietnam.py    # calls phase4 implementation
rag/parse_pdfs.py                    # calls data_ingestion.pdf_ingestion
rag/index_documents.py               # calls phase6 implementation
rag/embeddings.py                    # re-exports phase6 embedding client
pipelines/run_pipeline.py            # calls phase1 implementation
pipelines/config.py                  # re-exports phase1 config
pipelines/spark_jobs/*.py            # re-export phase1 Spark jobs
```

Vì vậy các lệnh cũ vẫn dùng được:

```powershell
python scripts\run_shap_sdg16_vietnam.py
python -m rag.parse_pdfs
python -m rag.index_documents
python -m pipelines.run_pipeline all --config configs\project.yaml
```
