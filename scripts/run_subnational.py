#!/usr/bin/env python3
"""
Script to run subnational analysis
Usage: python scripts/run_subnational.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from pipelines.subnational_analyzer import SubnationalAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main subnational analysis function"""
    try:
        # Check if data exists
        data_path = Path("data/subnational/sdg16_provinces.csv")
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            logger.info("Creating sample data...")
        
        logger.info("=" * 60)
        logger.info("Subnational Analysis Pipeline")
        logger.info("=" * 60)
        
        # Initialize analyzer
        analyzer = SubnationalAnalyzer()
        analyzer.load_data()
        
        # Run ranking
        logger.info("\nProvince Rankings:")
        rankings = analyzer.rank_provinces()
        
        logger.info(f"\nTop 10 Strongest Provinces ({rankings['year']}):")
        for i, province in enumerate(rankings['top_10_strongest'], 1):
            score = rankings['rankings'][province]
            logger.info(f"  {i:2d}. {province}: {score:.4f}")
        
        logger.info(f"\nTop 10 Weakest Provinces ({rankings['year']}):")
        for i, province in enumerate(rankings['top_10_weakest'], 1):
            score = rankings['rankings'][province]
            logger.info(f"  {i:2d}. {province}: {score:.4f}")
        
        # Run panel regression
        logger.info("\nPanel Regression Results:")
        panel_results = analyzer.run_panel_regression()
        
        logger.info(f"  R² Within:  {panel_results['r2_within']:.4f}")
        logger.info(f"  R² Between: {panel_results['r2_between']:.4f}")
        logger.info(f"  R² Overall: {panel_results['r2_overall']:.4f}")
        logger.info(f"  Observations: {panel_results['nobs']}")
        
        logger.info("\n  Coefficients:")
        for var, coef in panel_results['params'].items():
            pval = panel_results['pvalues'].get(var, 1.0)
            significant = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            logger.info(f"    {var}: {coef:.4f} (p={pval:.4f}) {significant}")
        
        # Map SDG to PAPI
        logger.info("\nMapping SDG16 to PAPI:")
        mapping = analyzer.map_sdg_to_papi()
        for sdg, papi in mapping.items():
            logger.info(f"  {sdg} → {papi['papi_component']}: correlation={papi['correlation']:.4f}")
        
        # Create heatmap
        logger.info("\nCreating heatmap...")
        Path("figures").mkdir(parents=True, exist_ok=True)
        analyzer.create_heatmap("figures/provincial_heatmap.png")
        
        # Save results
        logger.info("Saving results...")
        analyzer.save_results()
        
        logger.info("\n✅ Subnational analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Subnational analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())