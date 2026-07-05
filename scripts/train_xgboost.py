#!/usr/bin/env python3
"""
Script to train XGBoost model using global data
Usage: python scripts/train_xgboost.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from pipelines.xgboost_pipeline import XGBoostPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    try:
        # Check if data exists
        data_path = Path("data/clean/sdg16_spark.csv")
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            logger.info("Please place sdg16_spark.csv in data/clean/ directory")
            return 1
        
        logger.info("=" * 60)
        logger.info("XGBoost Training Pipeline")
        logger.info("=" * 60)
        
        # Train model
        logger.info("Starting XGBoost training...")
        pipeline = XGBoostPipeline()
        metrics = pipeline.train(optimize=True)
        
        logger.info("\nTraining Metrics:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE:  {metrics['mae']:.4f}")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        logger.info(f"  Train size: {metrics['train_size']}")
        logger.info(f"  Test size:  {metrics['test_size']}")
        
        # Run ablation study
        logger.info("\nRunning ablation study...")
        ablation_metrics = pipeline.ablation_study(top_n=3)
        
        logger.info("Ablation Results (removed top 3 features):")
        logger.info(f"  RMSE: {ablation_metrics['rmse']:.4f}")
        logger.info(f"  MAE:  {ablation_metrics['mae']:.4f}")
        logger.info(f"  R²:   {ablation_metrics['r2']:.4f}")
        logger.info(f"  Removed features: {ablation_metrics['removed_features']}")
        
        # Show top features
        logger.info("\nTop 10 Feature Importances:")
        for i, row in pipeline.feature_importances.head(10).iterrows():
            logger.info(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        logger.info("\nSaving model...")
        pipeline.save_model()
        
        logger.info("\n✅ XGBoost training completed successfully!")
        logger.info(f"Model saved to: artifacts/xgboost/")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())