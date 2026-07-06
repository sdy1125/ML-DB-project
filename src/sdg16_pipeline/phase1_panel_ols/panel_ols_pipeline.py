"""
Panel OLS with Fixed Effects
Goal16_lt = α + ΣB-1nd_lt + μ + λ + ε
μ = Entity FE (country-specific)
λ = Time FE (global shocks)
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
from linearmodels.panel import compare
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanelOLSPipeline:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def prepare_panel_data(self, df):
        """Prepare data for panel regression"""
        # Set multi-index for panel data
        df_panel = df.set_index(['country', 'year'])
        
        # Identify features
        feature_cols = [col for col in df.columns if col.startswith('n_sdg16_')]
        target_col = 'goal16'
        
        # Ensure numeric
        df_panel = df_panel[feature_cols + [target_col]]
        df_panel = df_panel.astype(float)
        
        return df_panel, feature_cols, target_col
    
    def run_panel_ols(self, df):
        """Run Panel OLS with Fixed Effects"""
        logger.info("Running Panel OLS with Fixed Effects...")
        
        df_panel, features, target = self.prepare_panel_data(df)
        y = df_panel[target]
        X = df_panel[features]
        
        # Add intercept
        X = sm.add_constant(X)
        
        # Entity and Time Fixed Effects
        model_entity = PanelOLS(y, X, entity_effects=True, time_effects=False)
        model_both = PanelOLS(y, X, entity_effects=True, time_effects=True)
        
        # Fit models
        results_entity = model_entity.fit(cov_type='clustered', cluster_entity=True)
        results_both = model_both.fit(cov_type='clustered', cluster_entity=True)
        
        # Store results
        self.models['entity_fe'] = model_entity
        self.models['both_fe'] = model_both
        self.results['entity_fe'] = results_entity
        self.results['both_fe'] = results_both
        
        # Extract coefficients and statistics
        coeffs = {
            'entity_fe': {
                'coefficients': results_entity.params.to_dict(),
                'p_values': results_entity.pvalues.to_dict(),
                'r2_within': results_entity.rsquared_within,
                'r2_between': results_entity.rsquared_between,
                'r2_overall': results_entity.rsquared
            },
            'both_fe': {
                'coefficients': results_both.params.to_dict(),
                'p_values': results_both.pvalues.to_dict(),
                'r2_within': results_both.rsquared_within,
                'r2_between': results_both.rsquared_between,
                'r2_overall': results_both.rsquared
            }
        }
        
        logger.info(f"Panel OLS complete. R² within: {coeffs['both_fe']['r2_within']:.3f}")
        return coeffs
    
    def run_pooled_ols(self, df):
        """Run Pooled OLS for comparison"""
        logger.info("Running Pooled OLS...")
        
        df_panel, features, target = self.prepare_panel_data(df)
        y = df_panel[target]
        X = df_panel[features]
        X = sm.add_constant(X)
        
        model = PooledOLS(y, X)
        results = model.fit()
        
        self.models['pooled'] = model
        self.results['pooled'] = results
        
        coeffs = {
            'coefficients': results.params.to_dict(),
            'p_values': results.pvalues.to_dict(),
            'r2': results.rsquared
        }
        
        return coeffs
    
    def compare_models(self):
        """Compare different panel models"""
        logger.info("Comparing models...")
        
        comparison = {}
        for name, result in self.results.items():
            comparison[name] = {
                'r2_within': getattr(result, 'rsquared_within', None) or result.rsquared,
                'r2_between': getattr(result, 'rsquared_between', None),
                'r2_overall': getattr(result, 'rsquared', None),
                'nobs': result.nobs,
                'df_model': result.df_model
            }
        
        return comparison
    
    def get_significant_factors(self, p_value_threshold=0.05):
        """Get factors with p-value < threshold"""
        significant = {}
        
        for model_name, result in self.results.items():
            pvals = result.pvalues
            sig_factors = pvals[pvals < p_value_threshold].index.tolist()
            significant[model_name] = sig_factors
            
        return significant
    
    def save_results(self, path="artifacts/panel_ols"):
        """Save panel OLS results"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'models': {}
        }
        
        for name, result in self.results.items():
            results_data['models'][name] = {
                'params': result.params.to_dict(),
                'pvalues': result.pvalues.to_dict(),
                'rsquared_within': getattr(result, 'rsquared_within', None),
                'rsquared_between': getattr(result, 'rsquared_between', None),
                'rsquared_overall': getattr(result, 'rsquared', None)
            }
        
        with open(f"{path}/panel_ols_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {path}")

# Example usage
if __name__ == "__main__":
    df = pd.read_parquet("data/clean/sdg16.parquet")
    
    pipeline = PanelOLSPipeline()
    results = pipeline.run_panel_ols(df)
    pooled_results = pipeline.run_pooled_ols(df)
    
    comparison = pipeline.compare_models()
    significant = pipeline.get_significant_factors()
    
    pipeline.save_results()