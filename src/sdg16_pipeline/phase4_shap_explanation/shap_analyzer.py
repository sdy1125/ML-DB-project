"""
SHAP Analysis for model interpretation
- TreeExplainer for XGBoost
- LinearExplainer for OLS
- Cross-validate consistency
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime  # THÊM DÒNG NÀY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    def __init__(self):
        self.shap_values = {}
        self.explainers = {}
        
    def analyze_xgboost(self, model, X_test, feature_names=None):
        """SHAP analysis for XGBoost model using TreeExplainer"""
        logger.info("Running SHAP analysis for XGBoost...")
        
        if feature_names is None:
            feature_names = X_test.columns.tolist()
        
        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Store
        self.explainers['xgboost'] = explainer
        self.shap_values['xgboost'] = shap_values
        
        # Global importance
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        logger.info(f"Top 3 SHAP features: {shap_importance.head(3)['feature'].tolist()}")
        
        return shap_values, shap_importance
    
    def analyze_linear(self, model, X_test, feature_names=None):
        """SHAP analysis for linear model using LinearExplainer"""
        logger.info("Running SHAP analysis for Linear model...")
        
        if feature_names is None:
            feature_names = X_test.columns.tolist()
        
        # Create LinearExplainer
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
        
        # Store
        self.explainers['linear'] = explainer
        self.shap_values['linear'] = shap_values
        
        # Global importance
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        return shap_values, shap_importance
    
    def cross_validate_consistency(self, model_func, X, y, n_splits=5):
        """Cross-validate SHAP consistency"""
        logger.info("Cross-validating SHAP consistency...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        shap_importances = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = model_func(X_train, y_train)
            
            # Get SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            importance = np.abs(shap_values).mean(axis=0)
            shap_importances.append(importance)
        
        # Calculate consistency
        shap_importances = np.array(shap_importances)
        consistency = np.std(shap_importances, axis=0) / (np.mean(shap_importances, axis=0) + 1e-8)
        
        logger.info(f"Average SHAP consistency: {np.mean(consistency):.3f}")
        
        return consistency
    
    def plot_beeswarm(self, shap_values, feature_names=None, save_path=None):
        """Generate beeswarm plot"""
        plt.figure(figsize=(12, 8))
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
        
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.title("SHAP Beeswarm Plot - Global Feature Importance")
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Beeswarm plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return plt.gcf()
    
    def plot_waterfall(self, shap_values, instance_idx, feature_names=None, 
                       instance_value=None, save_path=None):
        """Generate waterfall plot for a single instance"""
        plt.figure(figsize=(12, 8))
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
        
        # Get base value from explainer if available
        base_value = 0
        if 'xgboost' in self.explainers:
            base_value = self.explainers['xgboost'].expected_value
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[instance_idx],
                base_values=base_value,
                data=shap_values[instance_idx],
                feature_names=feature_names
            ),
            show=False
        )
        
        plt.title("SHAP Waterfall Plot - Individual Prediction Explanation")
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return plt.gcf()
    
    def plot_force(self, shap_values, instance_idx, feature_names=None, 
                   save_path=None):
        """Generate force plot"""
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]
        
        # Get base value from explainer
        base_value = 0
        if 'xgboost' in self.explainers:
            base_value = self.explainers['xgboost'].expected_value
        
        plt.figure(figsize=(20, 4))
        
        # Try different force plot methods
        try:
            # Method 1: Modern SHAP
            shap.force_plot(
                base_value,
                shap_values[instance_idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        except:
            try:
                # Method 2: Alternative syntax
                shap.force_plot(
                    base_value,
                    shap_values[instance_idx],
                    feature_names=feature_names,
                    show=False
                )
            except:
                try:
                    # Method 3: Using plots module
                    shap.plots.force(
                        base_value,
                        shap_values[instance_idx],
                        feature_names=feature_names,
                        show=False
                    )
                except:
                    # Method 4: Simple text output
                    logger.warning("Force plot not available. Using text summary...")
                    plt.text(0.5, 0.5, "Force plot not available in this version", 
                            ha='center', va='center', fontsize=16)
                    plt.axis('off')
        
        plt.title("SHAP Force Plot - Feature Contributions")
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return plt.gcf()
    
    def get_top_gaps(self, shap_values, feature_names, top_n=3):
        """Get top gaps based on SHAP values"""
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_shap)[-top_n:][::-1]
        
        top_gaps = [{
            'feature': feature_names[i],
            'shap_importance': float(mean_shap[i])
        } for i in top_indices]
        
        return top_gaps
    
    def save_results(self, path="artifacts/shap"):
        """Save SHAP analysis results"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'shap_values': {}
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for model, values in self.shap_values.items():
            if values is not None:
                if isinstance(values, np.ndarray):
                    results['shap_values'][model] = values.tolist()
                else:
                    results['shap_values'][model] = str(values)
        
        with open(f"{path}/shap_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"SHAP results saved to {path}")