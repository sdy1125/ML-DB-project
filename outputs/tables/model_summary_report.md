# Modeling Summary Report

## Data Used
- Source file: `C:\Users\ACER\Documents\ML_BD_Project_CK\ML-DB-project\data\clean\sdg16_spark.csv`
- Raw shape: 10150 rows x 21 columns
- Prepared shape: 10150 rows x 35 columns
- Target column: `goal16`
- Country column: `country`
- Year column: `year`
- Project structure detected: {'data': True, 'data/clean': True, 'notebooks': False, 'src': False, 'scripts': False, 'models': False, 'outputs': True, 'pipelines': True, 'pipelines/modeling': True}

## Model Roles
- Model 1, Elastic Net Regression: explain hidden weights of input indicators.
- Model 2, XGBoost Regression: improve prediction accuracy with nonlinear patterns.
- Model 3, ARIMAX/ARIMA: forecast Vietnam score for 2025-2030 under three scenarios.

## Model 1 - Elastic Net Regression
Elastic Net estimates regularized coefficients on standardized SDG indicators. Target-history features are excluded before training so the coefficients support indicator interpretation instead of autoregressive prediction.

Metrics:
- RMSE: 7.9003
- MAE: 5.5864
- R2: 0.7188
- MAPE: 9.5368

Excluded target-history features:
`goal16_lag_1`, `goal16_lag_2`, `goal16_rolling_mean_3`

Top hidden weights from Model 1 feature set:
- n_sdg16_homicides: normalized_weight=0.1102, direction=positive
- n_sdg16_detain: normalized_weight=0.1001, direction=positive
- n_sdg16_cpi: normalized_weight=0.0954, direction=positive
- n_sdg16_cpi_lag_1: normalized_weight=0.0952, direction=positive
- n_sdg16_u5reg: normalized_weight=0.0856, direction=positive
- n_sdg16_justice: normalized_weight=0.0656, direction=negative
- n_sdg16_power: normalized_weight=0.0631, direction=positive
- n_sdg16_exprop: normalized_weight=0.0548, direction=positive
- n_sdg16_weaponsexp: normalized_weight=0.0459, direction=positive
- n_sdg16_crime: normalized_weight=0.0379, direction=positive

Hidden weights file: `C:\Users\ACER\Documents\ML_BD_Project_CK\ML-DB-project\outputs\tables\model1_hidden_weights.csv`

## Model 2 - XGBoost Regression
XGBoost is used as the accuracy boost model for nonlinear relationships.

Metrics:
- RMSE: 2.5489
- MAE: 1.6510
- R2: 0.9707
- MAPE: 2.7629

Feature importance: `C:\Users\ACER\Documents\ML_BD_Project_CK\ML-DB-project\outputs\tables\model2_feature_importance.csv`


## Model 3 - ARIMAX/ARIMA Forecast
Model used: `ARIMAX` with order `(1, 0, 1)`.

Exogenous features used: `['n_sdg16_homicides', 'n_sdg16_detain', 'n_sdg16_cpi']`

Forecast file: `C:\Users\ACER\Documents\ML_BD_Project_CK\ML-DB-project\outputs\predictions\model3_vietnam_forecast_2025_2030.csv`

Policy impact file (+1% per exogenous indicator): `C:\Users\ACER\Documents\ML_BD_Project_CK\ML-DB-project\outputs\tables\model3_policy_impact_plus1pct.csv`

Top average +1% policy impacts:
- optimistic, n_sdg16_homicides: average score_delta=0.8466
- base, n_sdg16_homicides: average score_delta=0.8063
- pessimistic, n_sdg16_homicides: average score_delta=0.7660
- optimistic, n_sdg16_cpi: average score_delta=0.0038
- base, n_sdg16_cpi: average score_delta=0.0037
- pessimistic, n_sdg16_cpi: average score_delta=0.0035
- pessimistic, n_sdg16_detain: average score_delta=-0.1711
- base, n_sdg16_detain: average score_delta=-0.1801
- optimistic, n_sdg16_detain: average score_delta=-0.1891

## Assumptions
- Data was split by time, with the most recent years reserved for testing.
- Missing values were forward/backward-filled within country when panel data was available, then train-set median imputation was used inside model pipelines.
- Lag and rolling features were generated within country for panel data.
- Model 1 excludes historical features of the target score, such as target lags and target rolling means, because its role is interpretation of SDG indicators.
- Country dummies are disabled by default so Model 1 hidden weights focus on numeric indicators.
- Model 3 uses Model 1 hidden weights only to order candidate exogenous variables; it does not compare or select a best model.
- Model 3 policy impact is a what-if sensitivity test: one exogenous indicator is increased by 1% while the selected scenario context is held fixed.

## Current Limitations
- Model 2 requires the external `xgboost` package.
- Forecast scenarios depend on simple historical trend extrapolation for exogenous variables.
- The pipeline does not compare the three models or choose a best model because each model has a separate role.
