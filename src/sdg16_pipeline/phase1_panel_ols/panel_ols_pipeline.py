"""Panel OLS with entity and time fixed effects.

Model:
    Goal16_it = alpha + sum(beta_k * indicator_kit) + mu_i + lambda_t + epsilon_it

Where:
    mu_i     = entity fixed effects, country-specific effects
    lambda_t = time fixed effects, global year shocks
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PanelOLSPipeline:
    """Run Phase 1 Panel OLS / Fixed Effects for SDG16."""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.features: list[str] = []

    def load_data(self, path: str | Path = "data/clean/sdg16_spark.csv") -> pd.DataFrame:
        """Load cleaned SDG16 CSV used by the rest of the project."""
        logger.info("Loading panel data from %s", path)
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df

    def prepare_panel_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], str]:
        """Prepare country-year panel data for regression."""
        df = df.copy()
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        required = {"country", "year", "goal16"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required panel columns: {sorted(missing)}")

        feature_cols = [col for col in df.columns if col.startswith("n_sdg16_")]
        target_col = "goal16"
        if not feature_cols:
            raise ValueError("No n_sdg16_* features found for Panel OLS.")

        use_cols = ["country", "year", target_col] + feature_cols
        df_panel = df[use_cols].dropna(subset=["country", "year", target_col]).copy()
        df_panel["year"] = pd.to_numeric(df_panel["year"], errors="coerce")
        df_panel[target_col] = pd.to_numeric(df_panel[target_col], errors="coerce")
        for col in feature_cols:
            df_panel[col] = pd.to_numeric(df_panel[col], errors="coerce")

        df_panel = df_panel.dropna(subset=["year", target_col])
        medians = df_panel[feature_cols].median(numeric_only=True)
        df_panel[feature_cols] = df_panel[feature_cols].fillna(medians)
        df_panel["year"] = df_panel["year"].astype(int)
        df_panel = df_panel.set_index(["country", "year"]).sort_index()

        self.features = feature_cols
        return df_panel, feature_cols, target_col

    def _with_constant(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        if "const" not in x.columns:
            x.insert(0, "const", 1.0)
        return x

    def run_panel_ols(self, df: pd.DataFrame) -> dict:
        """Run Panel OLS with entity FE and entity+time FE."""
        logger.info("Running Panel OLS with Fixed Effects...")

        df_panel, features, target = self.prepare_panel_data(df)
        y = df_panel[target]
        x = self._with_constant(df_panel[features])

        model_entity = PanelOLS(
            y,
            x,
            entity_effects=True,
            time_effects=False,
            drop_absorbed=True,
            check_rank=False,
        )
        model_both = PanelOLS(
            y,
            x,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=False,
        )

        results_entity = model_entity.fit(cov_type="clustered", cluster_entity=True)
        results_both = model_both.fit(cov_type="clustered", cluster_entity=True)

        self.models["entity_fe"] = model_entity
        self.models["both_fe"] = model_both
        self.results["entity_fe"] = results_entity
        self.results["both_fe"] = results_both

        coeffs = {
            "entity_fe": self._summarize_result(results_entity),
            "both_fe": self._summarize_result(results_both),
        }

        logger.info(
            "Panel OLS complete. Entity+Time FE R² within: %.4f",
            coeffs["both_fe"]["r2_within"],
        )
        return coeffs

    def run_pooled_ols(self, df: pd.DataFrame) -> dict:
        """Run Pooled OLS for comparison."""
        logger.info("Running Pooled OLS...")

        df_panel, features, target = self.prepare_panel_data(df)
        y = df_panel[target]
        x = self._with_constant(df_panel[features])

        model = PooledOLS(y, x)
        results = model.fit()

        self.models["pooled"] = model
        self.results["pooled"] = results

        return self._summarize_result(results)

    def compare_models(self) -> dict:
        """Compare available panel models."""
        logger.info("Comparing panel models...")
        return {
            name: {
                "r2_within": self._safe_float(getattr(result, "rsquared_within", None)),
                "r2_between": self._safe_float(getattr(result, "rsquared_between", None)),
                "r2_overall": self._safe_float(getattr(result, "rsquared", None)),
                "nobs": int(result.nobs),
                "df_model": int(result.df_model),
            }
            for name, result in self.results.items()
        }

    def diagnostic_tests(self) -> dict:
        """Compute lightweight diagnostics for fitted panel models.

        These diagnostics avoid statsmodels so the pipeline remains compatible
        with the current Python/scipy environment.
        """
        diagnostics = {}
        for name, result in self.results.items():
            residuals = result.resids.dropna()
            fitted = result.fitted_values.iloc[:, 0].reindex(residuals.index)
            abs_resid = residuals.abs()
            resid_sq = residuals.pow(2)

            dw_values = []
            for _, group in residuals.groupby(level=0):
                values = group.sort_index().to_numpy(dtype=float)
                denom = float(np.sum(values**2))
                if len(values) > 1 and denom:
                    dw_values.append(float(np.sum(np.diff(values) ** 2) / denom))

            corr_abs_resid_fitted = (
                float(abs_resid.corr(fitted)) if fitted.notna().sum() > 2 else None
            )
            diagnostics[name] = {
                "rmse": float(np.sqrt(np.mean(residuals.to_numpy(dtype=float) ** 2))),
                "mae": float(np.mean(np.abs(residuals.to_numpy(dtype=float)))),
                "residual_mean": float(residuals.mean()),
                "residual_std": float(residuals.std()),
                "residual_skew": float(residuals.skew()),
                "residual_kurtosis": float(residuals.kurtosis()),
                "durbin_watson_panel_mean": float(np.mean(dw_values)) if dw_values else None,
                "durbin_watson_panel_min": float(np.min(dw_values)) if dw_values else None,
                "durbin_watson_panel_max": float(np.max(dw_values)) if dw_values else None,
                "abs_residual_fitted_correlation": corr_abs_resid_fitted,
                "heteroskedasticity_flag": (
                    abs(corr_abs_resid_fitted) > 0.25
                    if corr_abs_resid_fitted is not None
                    else None
                ),
                "max_abs_residual": float(np.max(np.abs(residuals.to_numpy(dtype=float)))),
                "nobs": int(result.nobs),
            }
        return diagnostics

    def multicollinearity_diagnostics(self, df: pd.DataFrame) -> dict:
        """Compute VIF values for SDG16 features before FE estimation."""
        df_panel, features, _ = self.prepare_panel_data(df)
        x = df_panel[features].to_numpy(dtype=float)
        vif = {}
        for idx, feature in enumerate(features):
            y = x[:, idx]
            others = np.delete(x, idx, axis=1)
            others = np.column_stack([np.ones(len(others)), others])
            beta, *_ = np.linalg.lstsq(others, y, rcond=None)
            pred = others @ beta
            sse = float(np.sum((y - pred) ** 2))
            sst = float(np.sum((y - float(np.mean(y))) ** 2))
            r2 = 1.0 - sse / sst if sst else 0.0
            vif[feature] = float(1.0 / max(1e-9, 1.0 - r2))
        return {
            "vif": dict(sorted(vif.items(), key=lambda item: item[1], reverse=True)),
            "high_vif_features_gt_10": [
                feature for feature, value in vif.items() if value > 10.0
            ],
        }

    def get_significant_factors(self, p_value_threshold: float = 0.05) -> dict:
        """Return factors with p-value below threshold."""
        significant = {}
        for model_name, result in self.results.items():
            pvals = result.pvalues
            significant[model_name] = pvals[pvals < p_value_threshold].index.tolist()
        return significant

    def save_results(self, path: str | Path = "artifacts/panel_ols") -> dict:
        """Save panel OLS results."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "model_type": "panel_ols_fixed_effects",
            "formula": (
                "goal16_it = alpha + beta*sdg16_indicators_it "
                "+ country_fe + year_fe + error_it"
            ),
            "features": self.features,
            "models": {
                name: self._summarize_result(result)
                for name, result in self.results.items()
            },
            "comparison": self.compare_models(),
            "diagnostics": self.diagnostic_tests(),
            "multicollinearity": self.multicollinearity_diagnostics(
                self.load_data("data/clean/sdg16_spark.csv")
            ),
            "significant_factors_p05": self.get_significant_factors(0.05),
        }

        output_path = output_dir / "panel_ols_results.json"
        output_path.write_text(
            json.dumps(results_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Results saved to %s", output_path)
        return results_data

    def _summarize_result(self, result) -> dict:
        return {
            "params": {
                key: float(value)
                for key, value in result.params.to_dict().items()
            },
            "pvalues": {
                key: float(value)
                for key, value in result.pvalues.to_dict().items()
            },
            "r2_within": self._safe_float(getattr(result, "rsquared_within", None)),
            "r2_between": self._safe_float(getattr(result, "rsquared_between", None)),
            "r2_overall": self._safe_float(getattr(result, "rsquared", None)),
            "nobs": int(result.nobs),
            "df_model": int(result.df_model),
        }

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            if np.isnan(value):
                return None
        except TypeError:
            pass
        return float(value)


def main() -> int:
    pipeline = PanelOLSPipeline()
    df = pipeline.load_data("data/clean/sdg16_spark.csv")
    panel_results = pipeline.run_panel_ols(df)
    pooled_results = pipeline.run_pooled_ols(df)
    saved = pipeline.save_results()

    print("Panel OLS completed.")
    print(f"Entity+Time FE R² within: {panel_results['both_fe']['r2_within']:.4f}")
    print(f"Entity+Time FE R² overall: {panel_results['both_fe']['r2_overall']:.4f}")
    print(f"Pooled OLS R² overall: {pooled_results['r2_overall']:.4f}")
    print("Artifacts written to artifacts/panel_ols/panel_ols_results.json")
    print(f"Features: {len(saved['features'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
