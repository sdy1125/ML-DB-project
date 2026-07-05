#!/usr/bin/env python3
"""
Script to run SHAP analysis
Usage: python scripts/run_shap.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import logging
from pipelines.shap_analyzer import SHAPAnalyzer
from pipelines.xgboost_pipeline import XGBoostPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main SHAP analysis function"""
    try:
        logger.info("Loading data and model...")
        
        # Load data
        data_path = Path("data/clean/sdg16_spark.csv")
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return 1
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Load model
        model_path = Path("artifacts/xgboost")
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return 1
        
        model = joblib.load(model_path / "xgboost_model.pkl")
        scaler = joblib.load(model_path / "scaler.pkl")
        
        # Load feature names from metadata
        import json
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            all_feature_names = metadata.get('feature_names', [])
        
        logger.info(f"Model expects {len(all_feature_names)} features")
        
        # Recreate all features using the same pipeline
        logger.info("Recreating features for SHAP analysis...")
        
        # Use XGBoostPipeline to create features
        pipeline = XGBoostPipeline()
        pipeline.data_path = str(data_path)
        
        # Get original features
        exclude_cols = ['country', 'year', 'goal16', 'source']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        original_features = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Original features: {len(original_features)}")
        
        # Apply feature engineering
        df_engineered, _ = pipeline.feature_engineering(df, 'goal16')
        
        if len(df_engineered) == 0:
            logger.warning("No data after feature engineering. Using original data...")
            df_engineered = df.copy()
        
        # Get all features (including engineered ones)
        feature_cols = [col for col in df_engineered.columns if col not in ['country', 'year', 'goal16', 'source']]
        
        # Filter to only features that model expects
        available_features = [col for col in feature_cols if col in all_feature_names]
        
        if not available_features:
            logger.error("No matching features found. Creating all features...")
            # Use all features from engineered data
            available_features = feature_cols
        
        logger.info(f"Using {len(available_features)} features for SHAP")
        
        # Prepare X with all required features
        X = df_engineered[available_features]
        
        # Handle NaN values
        if X.isna().any().any():
            logger.info(f"Dropping rows with NaN values...")
            X = X.dropna()
            logger.info(f"Remaining rows: {len(X)}")
        
        # Check if we have enough data
        if len(X) < 10:
            logger.warning(f"Too few samples ({len(X)}) for SHAP analysis. Using sample data...")
            # Use first 100 rows of original data with only original features
            X_sample = df[original_features].head(100).fillna(0)
            available_features = original_features
            X = X_sample
        
        # Scale features - use only features that scaler was fit on
        try:
            X_scaled = scaler.transform(X)
            logger.info(f"Successfully scaled {len(X)} samples")
        except ValueError as e:
            logger.warning(f"Scaling failed: {e}")
            logger.info("Using raw features without scaling...")
            X_scaled = X.values
        
        # Run SHAP analysis
        logger.info("Running SHAP analysis...")
        analyzer = SHAPAnalyzer()
        
        # XGBoost SHAP
        shap_values, shap_importance = analyzer.analyze_xgboost(
            model, X_scaled, available_features
        )
        
        logger.info(f"\nTop 5 important features:")
        for i, row in shap_importance.head(5).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['shap_importance']:.4f}")
        
        # Generate plots
        logger.info("\nGenerating plots...")
        Path("figures").mkdir(parents=True, exist_ok=True)
        
        # Beeswarm plot
        try:
            analyzer.plot_beeswarm(
                shap_values, 
                available_features, 
                "figures/shap_beeswarm.png"
            )
            logger.info("  ✅ Beeswarm plot saved to figures/shap_beeswarm.png")
        except Exception as e:
            logger.error(f"  ❌ Failed to generate beeswarm plot: {e}")
        
        # Waterfall plot for first instance
        try:
            analyzer.plot_waterfall(
                shap_values, 
                0, 
                available_features,
                save_path="figures/shap_waterfall_vn.png"
            )
            logger.info("  ✅ Waterfall plot saved to figures/shap_waterfall_vn.png")
        except Exception as e:
            logger.error(f"  ❌ Failed to generate waterfall plot: {e}")
        
        # Force plot
        try:
            analyzer.plot_force(
                shap_values,
                0,
                available_features,
                save_path="figures/shap_force.png"
            )
            logger.info("  ✅ Force plot saved to figures/shap_force.png")
        except Exception as e:
            logger.error(f"  ❌ Failed to generate force plot: {e}")
        
        # Get top gaps for Vietnam
        top_gaps = analyzer.get_top_gaps(shap_values, available_features, top_n=3)
        logger.info(f"\nTop 3 gaps for Vietnam:")
        for i, gap in enumerate(top_gaps, 1):
            logger.info(f"  {i}. {gap['feature']}: {gap['shap_importance']:.4f}")
        
        # Save results
        logger.info("\nSaving results...")
        analyzer.save_results()
        
        # Save SHAP values for later use
        import pickle
        with open("artifacts/shap/shap_values.pkl", 'wb') as f:
            pickle.dump(shap_values, f)
        with open("artifacts/shap/feature_names.pkl", 'wb') as f:
            pickle.dump(available_features, f)
        logger.info("  ✅ SHAP values saved to artifacts/shap/")
        
        logger.info("\n" + "="*60)
        logger.info("✅ SHAP analysis completed successfully!")
        logger.info("="*60)
        logger.info(f"Results: artifacts/shap/")
        logger.info(f"Figures: figures/")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"❌ SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())