# Explainable Machine Learning and Retrieval-Augmented Policy Recommendation for SDG16 Performance: Evidence from Cross-Country Panel Data and Vietnam

## Abstract

Improving institutional quality, justice accessibility, transparency, and public accountability remains a central challenge for countries pursuing Sustainable Development Goal 16 (SDG16). Although global SDG datasets provide rich cross-country indicators, existing approaches often stop at descriptive ranking and provide limited support for explainable, evidence-based policy recommendation. This study proposes an integrated machine learning and retrieval-augmented generation framework for predicting SDG16 performance, identifying country-specific institutional bottlenecks, forecasting future trajectories, and generating policy recommendations grounded in documentary evidence. Using the SDR2024 dataset covering 183 countries and 17 normalized SDG16 indicators, we compare four modeling strategies: Spark Linear Regression, an optimized XGBoost model with SHAP-style contribution analysis, an optional tuned XGBoost pipeline, and a GRU-based sequence forecaster. The optimized XGBoost model achieves the best predictive performance on the test set, with RMSE of 1.8249, MAE of 1.4019, and R² of 0.9859. The GRU model also performs strongly for temporal forecasting, achieving RMSE of 2.4411 and R² of 0.9747. For Vietnam, model explanations indicate that press freedom/accountability, access to justice, protection against expropriation, child labor, and administrative transparency are the most important negative contributors to SDG16 performance. The proposed framework further connects these quantitative findings with a retrieval-augmented policy recommendation engine using policy documents, legal texts, and international reports. The results demonstrate that combining predictive modeling, explainable AI, temporal forecasting, and evidence retrieval can move SDG monitoring from static benchmarking toward actionable policy intelligence.

**Keywords:** SDG16, explainable AI, XGBoost, SHAP, GRU, policy recommendation, RAG, Vietnam, institutional quality, sustainable development.

## 1. Introduction

Sustainable Development Goal 16 emphasizes peace, justice, effective institutions, transparency, anti-corruption, access to justice, and inclusive governance. These dimensions are difficult to evaluate because they involve both measurable institutional outcomes and complex political-administrative processes. Countries may receive an aggregate SDG16 score, but policymakers still need to know which indicators are dragging performance down, how the score may evolve in future years, and what concrete interventions are supported by evidence.

Traditional SDG monitoring systems usually rely on descriptive statistics, ranking tables, or dashboard-based comparison. While useful, these methods have three limitations. First, they do not always provide strong predictive capacity. Second, they offer limited explanation of how individual indicators contribute to the final score. Third, they rarely connect quantitative findings with policy documents in a way that can directly support recommendation generation.

This study addresses these limitations by developing a complete analytical pipeline for SDG16 performance analysis. The framework combines cross-country machine learning, explainable contribution analysis, temporal forecasting, subnational drill-down, and retrieval-augmented policy recommendation. Vietnam is used as the main country case because it represents a relevant policy context where improvements in transparency, justice, anti-corruption, and public administration are closely linked to national competitiveness and institutional reform.

The main contributions of this study are as follows:

1. It develops a comparative modeling framework for SDG16 score prediction using linear regression, XGBoost, and GRU sequence forecasting.
2. It identifies the best-performing model through RMSE, MAE, and R² comparison.
3. It applies SHAP-style contribution analysis to identify the indicators that negatively affect Vietnam's SDG16 score.
4. It uses GRU forecasting to estimate Vietnam's SDG16 trajectory for 2024–2030.
5. It integrates model outputs with retrieval-augmented generation to support evidence-based policy recommendation.

## 2. Related Work

Machine learning has increasingly been applied to sustainable development, governance measurement, and economic policy analysis. Tree-based ensemble methods such as XGBoost are widely used because they can model nonlinear relationships and interactions among indicators. However, black-box models are difficult to use in policy environments unless their predictions can be explained.

Explainable AI methods, especially SHAP-based feature attribution, have become important tools for interpreting machine learning predictions. In institutional and governance analysis, explainability is particularly important because policy recommendations should not be generated only from opaque scores. Policymakers need to understand why a country performs poorly and which factors should be prioritized.

Time-series and sequence models such as GRU and LSTM are also relevant for SDG forecasting. They can capture temporal dependence in country-level trajectories and support scenario planning. Nevertheless, forecasting alone is insufficient for policy recommendation unless it is linked to causal interpretation, domain evidence, and context-specific documents.

Recent advances in retrieval-augmented generation provide a way to combine structured model outputs with unstructured policy evidence. Instead of relying only on the language model's internal knowledge, RAG retrieves relevant documents and uses them as context for generating recommendations. This is suitable for SDG16 because policy solutions often depend on legal frameworks, national strategies, institutional reform programs, and international reports.

## 3. Data and Methodology

### 3.1 Data

The main dataset used in this study is the SDR2024 SDG dataset. After filtering the relevant observations, the modeling dataset contains:

| Item | Description |
|---|---|
| Source | SDR2024 |
| Number of observations | 4,392 |
| Number of countries | 183 |
| Input indicators | 17 normalized SDG16 indicators |
| Target variable | `goal16` |

The input variables are normalized indicators with the prefix `n_sdg16_*`. These variables represent different aspects of SDG16, including anti-corruption, safety, justice access, administrative quality, press freedom/accountability, birth registration, and protection against institutional risks.

### 3.2 Modeling Strategy

Four models are compared:

1. **Spark Linear Regression:** used as a transparent baseline and as a simple reverse-engineering approach for linear indicator weights.
2. **XGBoost SHAP Runner:** optimized gradient boosting model with contribution-based explanation using XGBoost `pred_contribs`.
3. **Optional XGBoost Tuned Pipeline:** an additional XGBoost training pipeline with optional Optuna-based hyperparameter tuning.
4. **GRU Sequence Forecaster:** a recurrent neural network model designed for temporal forecasting using five-year country histories.

For the main XGBoost model, data are split chronologically:

- Training set: observations up to 2018.
- Validation set: observations from 2019 to 2021.
- Test set: observations from 2022 onward.

The GRU model uses a sequence length of five years. Each input sample contains five consecutive years of SDG16 indicators, and the output is the next-year `Goal16` score.

### 3.3 Evaluation Metrics

The models are evaluated using three standard regression metrics:

```text
RMSE = root mean squared error
MAE  = mean absolute error
R²   = coefficient of determination
```

Lower RMSE and MAE indicate better predictive accuracy, while higher R² indicates stronger explanatory power.

### 3.4 Explainability and Policy Recommendation

The best-performing predictive model is connected to an explainability layer. For each prediction, the system computes the contribution of each indicator to the predicted SDG16 score. For Vietnam, negative contributions are interpreted as indicators that pull the predicted SDG16 score downward.

The final policy recommendation layer uses a retrieval-augmented generation pipeline. The retrieved documents include policy reports, legal texts, country reports, and academic evidence. The LLM receives four types of information:

1. Current predicted SDG16 performance.
2. Weakest indicators from the explainability layer.
3. Forecasted trajectory from the GRU model.
4. Retrieved documentary evidence from the RAG knowledge base.

## 4. Experimental Results

### 4.1 Model Comparison

The comparative results are shown in Table 1.

**Table 1. Model performance comparison**

| Model | Evaluation split | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| XGBoost SHAP Runner | Test | 1.8249 | 1.4019 | 0.9859 |
| GRU Sequence Forecaster | Test | 2.4411 | 1.9119 | 0.9747 |
| Optional XGBoost Tuned Pipeline | Overall | 7.7773 | 4.6659 | 0.6974 |
| Spark Linear Regression | Test | 12.2139 | 9.2173 | 0.3509 |

The optimized XGBoost SHAP Runner achieves the best overall performance, with the lowest RMSE and MAE and the highest R². This indicates that nonlinear tree-based modeling is highly effective for predicting SDG16 scores from normalized SDG16 indicators.

The GRU Sequence Forecaster also performs well, especially considering that it is designed for temporal forecasting rather than only static prediction. Its R² of 0.9747 suggests strong temporal signal in the SDG16 indicator sequence.

The Spark Linear Regression baseline performs substantially worse. This suggests that the relationship between SDG16 indicators and the aggregate `Goal16` score is not purely linear or that nonlinear interactions among indicators are important.

### 4.2 XGBoost Optimization

The best XGBoost configuration is:

```text
max_depth = 4
learning_rate = 0.09
n_estimators = 850
subsample = 0.82
colsample_bytree = 0.82
reg_alpha = 0.5
reg_lambda = 1.0
tree_method = hist
objective = reg:squarederror
```

**Table 2. Optimized XGBoost performance**

| Split | RMSE | MAE | R² |
|---|---:|---:|---:|
| Validation | 1.1886 | 0.8502 | 0.9937 |
| Test | 1.8249 | 1.4019 | 0.9859 |

The small gap between validation and test performance suggests that the model generalizes well and does not rely only on the validation period.

### 4.3 Explainability Results for Vietnam

For Vietnam, the model identifies several indicators with strong negative contributions to the predicted SDG16 score.

**Table 3. Main negative SHAP-style contributions for Vietnam**

| Indicator | Interpretation | Contribution |
|---|---|---:|
| `n_sdg16_rsf` | Press freedom / accountability | -2.3477 |
| `n_sdg16_justice` | Access to justice | -1.9228 |
| `n_sdg16_exprop` | Protection against expropriation / property rights | -1.3962 |
| `n_sdg16_clabor` | Child labor | -0.5346 |
| `n_sdg16_admin` | Transparent administration | -0.3568 |

These results imply that Vietnam's SDG16 improvement strategy should prioritize institutional transparency, justice accessibility, accountability, property-right protection, and administrative reform.

The global importance ranking also highlights the broad drivers of SDG16 prediction.

**Table 4. Global indicator importance**

| Indicator | Interpretation | Mean absolute contribution |
|---|---|---:|
| `n_sdg16_cpi` | Anti-corruption / CPI | 5.6254 |
| `n_sdg16_u5reg` | Birth registration | 3.1668 |
| `n_sdg16_detain` | Pre-trial detention | 2.1033 |
| `n_sdg16_homicides` | Violence-related deaths | 1.9804 |
| `n_sdg16_clabor` | Child labor | 1.9643 |

This shows that the model relies on a combination of governance, legal, safety, and human-rights-related indicators, rather than a single institutional variable.

### 4.4 GRU Forecasting for Vietnam

The best GRU configuration is:

```text
hidden_size = 32
num_layers = 1
dropout = 0.0
learning_rate = 0.01
sequence_length = 5
```

**Table 5. GRU forecasting performance**

| Split | RMSE | MAE | R² |
|---|---:|---:|---:|
| Validation | 1.6999 | 1.2350 | 0.9871 |
| Test | 2.4411 | 1.9119 | 0.9747 |

The GRU-based forecast for Vietnam is shown in Table 6.

**Table 6. Forecasted Goal16 score for Vietnam, 2024–2030**

| Year | Forecasted Goal16 |
|---:|---:|
| 2024 | 63.6456 |
| 2025 | 63.6389 |
| 2026 | 63.6521 |
| 2027 | 63.6432 |
| 2028 | 63.6419 |
| 2029 | 63.6419 |
| 2030 | 63.6419 |

The forecast suggests a stable trajectory around 63.64 under the current indicator structure. This implies that without targeted institutional reforms, Vietnam's SDG16 score may not improve substantially by 2030.

## 5. Discussion

The experimental results provide three important insights.

First, nonlinear machine learning models are more suitable than linear regression for SDG16 prediction. The large performance gap between XGBoost and Spark Linear Regression indicates that the relationship between institutional indicators and aggregate SDG16 performance is complex and nonlinear.

Second, prediction alone is insufficient for policy use. The strongest advantage of the selected XGBoost model is not only its accuracy but also its ability to produce indicator-level contributions. For Vietnam, the model identifies concrete bottlenecks rather than simply returning an aggregate score.

Third, the GRU forecast indicates that Vietnam's score may remain stable if no major changes occur in the underlying indicators. This creates a clear policy implication: improvement requires active intervention in the weakest institutional areas identified by the explainability layer.

The proposed RAG + LLM layer helps translate these findings into policy language. Instead of generating generic recommendations, the system can retrieve documents related to anti-corruption, administrative reform, justice access, transparency, and institutional modernization. This makes the final recommendations more grounded and auditable.

## 6. Policy Implications for Vietnam

Based on the model explanation and forecast results, Vietnam should prioritize the following policy directions:

1. **Strengthen transparency and accountability mechanisms.**  
   The negative contribution of `n_sdg16_rsf` and the high global importance of anti-corruption indicators suggest that transparency and accountability are central to SDG16 improvement.

2. **Improve access to justice.**  
   The negative contribution of `n_sdg16_justice` indicates that justice accessibility remains a major bottleneck. Policy efforts should focus on legal aid, procedural simplification, digital justice services, and equal access to legal protection.

3. **Enhance property-right protection and investor confidence.**  
   The negative contribution of `n_sdg16_exprop` suggests that protection against expropriation and institutional predictability matter for governance quality and investment climate.

4. **Reduce child labor and strengthen social protection.**  
   The contribution of `n_sdg16_clabor` shows the importance of linking SDG16 with labor regulation, social protection, education access, and vulnerable-group protection.

5. **Accelerate administrative reform and digital governance.**  
   The negative contribution of `n_sdg16_admin` indicates that transparent administration is still a relevant constraint. Digital public services, open data, and simplified procedures can improve institutional effectiveness.

## 7. System Architecture Contribution

The proposed framework is not limited to model training. It is designed as an end-to-end decision-support system:

```text
Cross-country SDG16 data
  -> Model training and evaluation
  -> XGBoost prediction and contribution analysis
  -> GRU forecasting for 2024–2030
  -> Subnational drill-down using provincial data
  -> RAG retrieval from policy documents
  -> LLM-based policy recommendation
```

This architecture allows the system to answer not only "what is Vietnam's score?" but also:

- Which indicators are weakening Vietnam's SDG16 performance?
- How may the score evolve by 2030?
- Which provinces or subnational units require attention?
- Which policy documents support the proposed interventions?
- What recommendations can be generated from both data and evidence?

## 8. Limitations

This study has several limitations. First, the analysis depends on the availability and quality of SDG16 indicators in the SDR2024 dataset. Missing data, measurement differences, or normalization choices may affect model outputs. Second, SHAP-style contributions explain model behavior rather than proving causal effects. Therefore, policy interpretation should be combined with domain knowledge and institutional evidence. Third, the GRU forecast assumes that historical patterns contain useful information for future years, but unexpected institutional reforms or political changes may alter the trajectory. Finally, the RAG-based recommendation layer depends on the completeness and quality of the document knowledge base.

## 9. Conclusion

This study proposes an explainable machine learning and retrieval-augmented policy recommendation framework for SDG16 performance analysis. Using cross-country SDR2024 data, the optimized XGBoost SHAP Runner achieves the best predictive performance among all tested models, with RMSE of 1.8249 and R² of 0.9859 on the test set. The GRU sequence forecaster provides complementary temporal forecasting, suggesting that Vietnam's SDG16 score may remain stable around 63.64 during 2024–2030 without substantial policy intervention.

For Vietnam, the most important negative contributors are press freedom/accountability, access to justice, protection against expropriation, child labor, and administrative transparency. These results provide a clear analytical basis for targeted policy recommendation. By integrating machine learning, explainability, forecasting, subnational analysis, and RAG-based evidence retrieval, the proposed system moves SDG16 monitoring from passive ranking toward actionable policy intelligence.

## Reproducibility

The main experimental pipeline can be reproduced with:

```powershell
python scripts\optimize_and_compare_models.py
```

Main artifacts:

- `artifacts/model_comparison/model_comparison.csv`
- `artifacts/model_comparison/best_model.json`
- `artifacts/shap/shap_summary.json`
- `artifacts/gru/gru_summary.json`
- `artifacts/gru/vietnam_forecast_2024_2030.csv`
