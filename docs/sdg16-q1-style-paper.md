# Explainable Composite-Score Reconstruction and RAG-Based Policy Intelligence for SDG16: Evidence from Vietnam

## Abstract

Improving institutional quality, justice accessibility, transparency, and public accountability remains central to Sustainable Development Goal 16 (SDG16). This study proposes an end-to-end SDG16 intelligence framework that combines Panel OLS with fixed effects, XGBoost-based composite-score reconstruction, XGBoost tree-contribution analysis, GRU forecasting, subnational drill-down, and retrieval-augmented generation (RAG) for policy recommendation. Using the SDR2024 dataset covering 183 countries and 17 normalized SDG16 component indicators, the main XGBoost runner achieves RMSE = 1.8249, MAE = 1.4019, and R2 = 0.9859 on the test set. Because `goal16` is constructed from the same component indicators, this performance is interpreted as high-fidelity composite-score reconstruction, not as independent causal prediction. Panel OLS with country and year fixed effects provides the econometric baseline, while GRU forecasting provides Vietnam's modeled baseline trajectory from 63.6456 in 2024 to 65.1743 in 2030. For Vietnam, the raw diagnostic output flags RSF/accountability and expropriation as large negative contributions, but both have zero-valued Vietnam inputs with nonzero benchmarks and are excluded from headline policy priorities. The headline policy priorities are access to justice, child labor, and administrative transparency. The RAG + LLM layer then translates model outputs and retrieved documentary evidence into policy recommendations.

**Keywords:** SDG16, XGBoost, explainable AI, Panel OLS, GRU, RAG, Vietnam, governance.

## 1. Introduction

SDG16 measures peace, justice, transparency, anti-corruption, institutional effectiveness, and inclusive governance. Aggregate SDG16 scores are useful for benchmarking, but they do not directly explain which component indicators pull a country's score down, how the trajectory may evolve, or how quantitative findings can be connected to real policy documents.

This study develops a practical SDG16 intelligence pipeline for Vietnam. The goal is not to prove causal effects among governance variables. Instead, the system reconstructs the SDG16 composite score from its component indicators, decomposes the reconstructed score into indicator-level contributions, forecasts the baseline trajectory, maps national bottlenecks to provincial indicators, and generates evidence-grounded policy recommendations through RAG + LLM.

The main contributions are:

1. A six-phase SDG16 pipeline aligned with the project implementation: Panel OLS, XGBoost, contribution analysis, subnational drill-down, GRU forecast, and RAG + LLM.
2. A clear distinction between composite-score reconstruction and causal/independent prediction.
3. Vietnam-specific diagnostic outputs identifying the component indicators that pull the reconstructed SDG16 score downward.
4. A policy recommendation layer that combines model outputs with retrieved documents.

## 2. Data and Methodology

### 2.1 Data

The main country-level dataset is `data/clean/sdg16_spark.csv`, derived from SDR2024. It contains 4,392 observations, 183 countries, annual records, the target `goal16`, and 17 normalized component indicators with prefix `n_sdg16_*`.

The subnational dataset is `data/subnational/sdg16_provinces.csv`, containing province-level PAPI/PCI/GRDP proxy indicators used for drill-down analysis.

The RAG layer uses parsed policy documents from `data/knowledge/text/` and processed chunks from `data/knowledge/processed/`.

### 2.2 Six-phase pipeline

The implemented workflow is:

```text
sdg16 country data
  -> Phase 1: Panel OLS + Fixed Effects
  -> Phase 2: XGBoost composite-score reconstruction
  -> Phase 3: XGBoost tree-contribution analysis
  -> Phase 5: GRU forecast 2024-2030
  -> Phase 6: RAG + LLM policy recommendation

sdg16 provincial data
  -> Phase 4: Subnational drill-down / province Panel FE
  -> Phase 6: RAG + LLM policy recommendation
```

Phase 1 uses `linearmodels.PanelOLS` with entity and time fixed effects as the official econometric baseline. Phase 2 trains the optional XGBoost pipeline. Phase 3 runs the main XGBoost reconstruction runner and exports tree contributions using XGBoost `pred_contribs=True`. Phase 4 maps national weak indicators to province-level PAPI/PCI dimensions. Phase 5 trains a GRU forecaster using 5-year country histories. Phase 6 generates recommendations from model outputs and retrieved policy evidence.

### 2.3 Methodological caveat: circular composite target

The dependent variable `goal16` is a composite score constructed from SDG16 component indicators. Therefore, models trained on `n_sdg16_*` indicators are learning to reconstruct/decompose the composite score. The high R2 of XGBoost is useful for diagnostic reconstruction and prioritization, but it must not be presented as causal evidence or as an independent forecast of governance quality from external predictors.

## 3. Results

### 3.1 Model comparison

| Model | Split | RMSE | MAE | R2 | Interpretation |
|---|---:|---:|---:|---:|---|
| XGBoost SHAP Reconstruction Runner | Test | 1.8249 | 1.4019 | 0.9859 | Best composite-score reconstruction |
| Panel OLS + Fixed Effects | Full panel | 2.1276 | 1.6099 | 0.6263 | Official econometric baseline |
| GRU Sequence Forecaster | Test | 2.4411 | 1.9119 | 0.9747 | Temporal forecasting layer |
| Optional XGBoost Tuned Pipeline | Overall | 7.7773 | 4.6659 | 0.6974 | Robustness/alternative runner |
| Spark Linear Regression | Test | 12.2139 | 9.2173 | 0.3509 | Legacy online fallback |

XGBoost performs best because it reconstructs the composite `goal16` score from normalized SDG16 components. This result is useful for decomposing a score into indicator contributions, but not for claiming that the model causally predicts governance outcomes.

### 3.2 Vietnam contribution analysis

Raw diagnostic contribution table:

| Indicator | Interpretation | Contribution | Data-quality status |
|---|---|---:|---|
| `n_sdg16_rsf` | Press freedom / accountability | -2.3477 | Flagged zero-value input |
| `n_sdg16_justice` | Access to justice | -1.9228 | Eligible |
| `n_sdg16_exprop` | Protection against expropriation / property rights | -1.3962 | Flagged zero-value input |
| `n_sdg16_clabor` | Child labor | -0.5346 | Eligible |
| `n_sdg16_admin` | Transparent administration | -0.3568 | Eligible |

Headline policy-priority table after excluding flagged zero-valued Vietnam indicators:

| Indicator | Interpretation | Contribution |
|---|---|---:|
| `n_sdg16_justice` | Access to justice | -1.9228 |
| `n_sdg16_clabor` | Child labor | -0.5346 |
| `n_sdg16_admin` | Transparent administration | -0.3568 |

### 3.3 GRU forecast for Vietnam

| Year | Forecasted Goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.8785 |
| 2026 | 64.1630 |
| 2027 | 64.4077 |
| 2028 | 64.6922 |
| 2029 | 64.9520 |
| 2030 | 65.1743 |

The current artifact shows a gradual upward baseline, not a flat trajectory. This forecast should be treated as a model baseline rather than an official SDSN projection.

### 3.4 Subnational drill-down

The subnational layer uses lagged province-level predictors, including `papi_score_lag1`, `pci_transparency_lag1`, and `grdp_index_lag1`, to reduce same-year leakage. Current artifacts indicate that province-level results are suitable for drill-down and demonstration, but should not be overinterpreted as final official provincial rankings unless the PAPI/PCI dataset is validated.

## 4. Discussion

The framework is strongest as a diagnostic policy intelligence system. Panel OLS provides a transparent baseline and residual diagnostics. XGBoost reconstructs the composite SDG16 score accurately and exposes which component indicators matter for the model. GRU contributes the temporal baseline, while the subnational module maps national weaknesses to province-level proxies. RAG + LLM then converts these structured findings into policy language grounded in documents.

The key correction is methodological framing. The system should not claim that XGBoost independently predicts SDG16 from external causes. Its best use is to reverse-engineer and decompose a composite score so that policymakers can see which measured indicators create the largest penalty.

## 5. Policy implications for Vietnam

Based on the current model outputs, Vietnam's SDG16 policy priorities should focus on:

1. Improving access to justice and legal aid.
2. Reducing child labor through enforcement, education access, and social protection.
3. Continuing administrative reform and digital public-service modernization.
4. Rechecking flagged RSF/accountability and expropriation inputs before turning them into headline policy claims.
5. Keeping transparency/accountability as a document-retrieval theme, not as a confirmed model headline until the flagged inputs are validated.

These recommendations should be read together with retrieved policy evidence, not as direct causal conclusions from the model alone.

## 6. Limitations

First, `goal16` is constructed from the same SDG16 indicators used as model inputs, so XGBoost accuracy is reconstruction accuracy. Second, tree-contribution outputs explain model behavior, not causal effects. Third, missing or imputed country values can distort the importance of individual indicators. Fourth, the GRU forecast depends on historical trajectories and can change when new data arrive. Fifth, the RAG + LLM recommendation layer is a prototype and has not yet undergone expert validation.

## 7. Conclusion

This study presents a six-phase SDG16 intelligence framework aligned with the current project implementation. The XGBoost reconstruction runner achieves the best composite-score reconstruction performance, Panel OLS provides the econometric baseline, GRU forecasts Vietnam's 2024-2030 trajectory, and RAG + LLM translates quantitative diagnostics into policy recommendations. The corrected interpretation is that the pipeline supports diagnostic decomposition and policy prioritization, not causal proof.

## Reproducibility

```powershell
python scripts\optimize_and_compare_models.py
```

Main artifacts:

- `artifacts/panel_ols/panel_ols_results.json`
- `artifacts/shap/shap_summary.json`
- `artifacts/gru/vietnam_forecast_2024_2030.csv`
- `artifacts/subnational/subnational_results.json`
- `artifacts/model_comparison/model_comparison.csv`
