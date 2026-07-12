"""
Subnational Analysis for Vietnam
- Mapping SDG16 → PAPI/PCI
- Provincial ranking
- Choropleth map
- Panel FE with provincial data
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
try:
    from linearmodels.panel import PanelOLS, PooledOLS
except ModuleNotFoundError:  # optional; fallback to statsmodels OLS below.
    PanelOLS = None
    PooledOLS = None
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubnationalAnalyzer:
    def __init__(self):
        self.province_data = None
        self.data_path = "data/subnational/sdg16_provinces.csv"
        self.mapping = {
            'CPI': 'bribery_people',
            'Admin': 'admin_procedure',
            'Justice': 'vertical_accountability',
            'Power': 'transparency',
            'Security': 'citizen_participation'
        }
        
    def load_data(self):
        """Load provincial data from CSV"""
        logger.info(f"Loading provincial data from {self.data_path}")
        
        if not Path(self.data_path).exists():
            logger.warning(f"File {self.data_path} not found, creating sample data")
            self._create_sample_data()
        
        self.province_data = pd.read_csv(self.data_path)
        
        # Standardize column names: convert to lowercase
        self.province_data.columns = self.province_data.columns.str.lower()
        
        # Check if 'year' column exists
        if 'year' not in self.province_data.columns:
            # Try to find any column with 'year' in name
            year_cols = [col for col in self.province_data.columns if 'year' in col.lower()]
            if year_cols:
                self.province_data = self.province_data.rename(columns={year_cols[0]: 'year'})
            else:
                raise KeyError("No 'year' column found in the data")
        
        # Create goal16 from PAPI/PCI components if not exists
        if 'goal16' not in self.province_data.columns:
            logger.info("Creating goal16 from available indicators...")
            self._create_goal16()
        
        logger.info(f"Loaded {len(self.province_data)} rows, {len(self.province_data.columns)} columns")
        logger.info(f"Columns: {self.province_data.columns.tolist()}")
        logger.info(f"Years available: {sorted(self.province_data['year'].unique())}")
        
        return self.province_data
    
    def _create_goal16(self):
        """Create goal16 score from available indicators"""
        # Use PAPI score as base if available
        if 'papi_score' in self.province_data.columns:
            # Normalize PAPI score to 0-1 range (assuming PAPI is 0-100)
            self.province_data['goal16'] = self.province_data['papi_score'] / 100
            logger.info("Created goal16 from PAPI score (normalized to 0-1)")
        else:
            # Create from components if PAPI not available
            components = ['citizen_participation', 'transparency', 'vertical_accountability', 
                         'bribery_people', 'admin_procedure']
            
            # Check which components exist
            existing_components = [col for col in components if col in self.province_data.columns]
            
            if existing_components:
                # Normalize each component to 0-1 (assuming 0-10 scale)
                for col in existing_components:
                    if self.province_data[col].max() > 1:  # Not normalized
                        self.province_data[col] = self.province_data[col] / 10
                
                # Average all components
                self.province_data['goal16'] = self.province_data[existing_components].mean(axis=1)
                logger.info(f"Created goal16 from components: {existing_components}")
            else:
                # Last resort: use random values or PCI
                if 'pci_transparency' in self.province_data.columns:
                    self.province_data['goal16'] = self.province_data['pci_transparency'] / 100
                    logger.info("Created goal16 from PCI transparency")
                else:
                    # Generate random scores for demonstration
                    np.random.seed(42)
                    self.province_data['goal16'] = np.random.uniform(0.4, 0.8, len(self.province_data))
                    logger.warning("No indicators found. Generated random goal16 for demonstration.")
    
    def _create_sample_data(self):
        """Create sample provincial data for demonstration"""
        provinces = [
            "Hà Nội", "TP.HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
            "Bắc Ninh", "Hưng Yên", "Hải Dương", "Hà Nam", "Nam Định",
            "Thái Bình", "Ninh Bình", "Thanh Hóa", "Nghệ An", "Hà Tĩnh",
            "Quảng Bình", "Quảng Trị", "Thừa Thiên Huế", "Quảng Nam", "Quảng Ngãi",
            "Bình Định", "Phú Yên", "Khánh Hòa", "Ninh Thuận", "Bình Thuận",
            "Kon Tum", "Gia Lai", "Đắk Lắk", "Đắk Nông", "Lâm Đồng",
            "Bình Phước", "Tây Ninh", "Bình Dương", "Đồng Nai", "Bà Rịa - Vũng Tàu",
            "Long An", "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long",
            "Đồng Tháp", "An Giang", "Kiên Giang", "Hậu Giang", "Sóc Trăng",
            "Bạc Liêu", "Cà Mau", "Lào Cai", "Yên Bái", "Điện Biên",
            "Lai Châu", "Sơn La", "Hòa Bình", "Phú Thọ", "Vĩnh Phúc",
            "Bắc Giang", "Lạng Sơn", "Cao Bằng", "Bắc Kạn", "Thái Nguyên",
            "Tuyên Quang", "Hà Giang", "Quảng Ninh"
        ]
        
        np.random.seed(42)
        years = [2019, 2020, 2021, 2022, 2023]
        data = []
        
        for province in provinces:
            base_goal16 = np.random.uniform(0.4, 0.8)
            for year in years:
                data.append({
                    'province': province,
                    'year': year,
                    'goal16': base_goal16 + np.random.normal(0, 0.02),
                    'bribery_people': np.random.uniform(1, 10),
                    'admin_procedure': np.random.uniform(1, 10),
                    'vertical_accountability': np.random.uniform(1, 10),
                    'transparency': np.random.uniform(1, 10),
                    'citizen_participation': np.random.uniform(1, 10),
                    'papi_score': np.random.uniform(30, 70),
                    'pci_transparency': np.random.uniform(50, 80),
                    'grdp_index': np.random.uniform(10000, 50000)
                })
        
        df = pd.DataFrame(data)
        
        # Ensure path exists
        Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_path, index=False)
        logger.info(f"Created sample provincial data at {self.data_path}")
    
    def map_sdg_to_papi(self):
        """Map SDG16 indicators to PAPI/PCI components"""
        logger.info("Mapping SDG16 to PAPI/PCI...")
        
        if self.province_data is None:
            self.load_data()
        
        mapping_results = {}
        for sdg_indicator, papi_component in self.mapping.items():
            # Find matching column (case insensitive)
            matching_cols = [col for col in self.province_data.columns if papi_component.lower() in col.lower()]
            if matching_cols:
                actual_col = matching_cols[0]
                # Calculate correlation with goal16
                if 'goal16' in self.province_data.columns:
                    corr = self.province_data['goal16'].corr(self.province_data[actual_col])
                else:
                    corr = 0
                mapping_results[sdg_indicator] = {
                    'papi_component': actual_col,
                    'correlation': corr
                }
        
        return mapping_results
    
    def rank_provinces(self, metric='goal16'):
        """Rank provinces by specified metric"""
        logger.info(f"Ranking provinces by {metric}...")
        
        if self.province_data is None:
            self.load_data()
        
        # Check if metric exists
        if metric not in self.province_data.columns:
            logger.error(f"Metric '{metric}' not found. Available columns: {self.province_data.columns.tolist()}")
            # Use first available numeric column as fallback
            numeric_cols = self.province_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'year' in numeric_cols:
                numeric_cols.remove('year')
            if numeric_cols:
                metric = numeric_cols[0]
                logger.info(f"Using '{metric}' as fallback metric")
            else:
                raise ValueError(f"No numeric columns found for ranking")
        
        # Get latest year data
        latest_year = self.province_data['year'].max()
        latest_data = self.province_data[self.province_data['year'] == latest_year]
        
        rankings = latest_data.groupby('province')[metric].mean().sort_values(ascending=False)
        
        top_10_strongest = rankings.head(10).index.tolist()
        top_10_weakest = rankings.tail(10).index.tolist()
        
        return {
            'rankings': rankings,
            'top_10_strongest': top_10_strongest,
            'top_10_weakest': top_10_weakest,
            'year': latest_year,
            'metric': metric
        }
    
    def run_panel_regression(self):
        """Run leakage-safe panel regression on provincial data.

        The provincial drill-down predicts current-year ``goal16`` from lagged
        PAPI/PCI/GRDP features at t-1. This avoids leaking same-year proxy
        components directly into the target year.
        """
        logger.info("Running panel regression on provincial data...")
        
        if self.province_data is None:
            self.load_data()
        
        df_work = self.province_data.copy().sort_values(["province", "year"])
        
        # Find available features
        feature_options = ['papi_score', 'pci_transparency', 'grdp_index']
        available_features = [f for f in feature_options if f in df_work.columns]
        
        if not available_features:
            logger.warning("No PAPI/PCI/GRDP columns found, using available numeric columns")
            numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['goal16', 'year']
            available_features = [col for col in numeric_cols if col not in exclude_cols][:3]
        
        # Target
        if 'goal16' not in df_work.columns:
            logger.warning("goal16 not found, using papi_score as target")
            target = 'papi_score'
        else:
            target = 'goal16'

        base_features = available_features[:3]
        lag_features = []
        for feature in base_features:
            lag_col = f"{feature}_lag1"
            df_work[lag_col] = df_work.groupby("province")[feature].shift(1)
            lag_features.append(lag_col)

        rows_before = len(df_work)
        df_work = df_work.dropna(subset=[target] + lag_features).copy()
        rows_after = len(df_work)

        # Prepare panel data after lagging
        df_panel = df_work.set_index(['province', 'year']).sort_index()
        
        y = df_panel[target].astype(float)
        X = df_panel[lag_features].astype(float).copy()
        if 'const' not in X.columns:
            X.insert(0, 'const', 1.0)

        if PanelOLS is None:
            logger.warning("linearmodels is not installed. Falling back to numpy OLS.")
            beta, *_ = np.linalg.lstsq(X.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None)
            predictions = X.to_numpy(dtype=float) @ beta
            residuals = y.to_numpy(dtype=float) - predictions
            sse = float(np.sum(residuals ** 2))
            sst = float(np.sum((y.to_numpy(dtype=float) - float(y.mean())) ** 2))
            r2 = 1.0 - sse / sst if sst else 0.0
            return {
                'r2_within': r2,
                'r2_between': r2,
                'r2_overall': r2,
                'params': {key: float(value) for key, value in zip(X.columns, beta)},
                'pvalues': {},
                'nobs': int(len(y)),
                'base_features': base_features,
                'features_used': lag_features,
                'target': target,
                'estimator': 'numpy_ols_fallback',
                'leakage_control': 'uses_lagged_t_minus_1_features_only',
                'dropped_rows_due_to_lag': int(rows_before - rows_after),
            }
        
        # Entity + time fixed effects when possible.
        model = PanelOLS(
            y,
            X,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=False,
        )
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        logger.info(f"Panel regression complete. R²: {results.rsquared:.3f}")
        
        # Get results summary
        results_dict = {
            'r2_within': results.rsquared_within,
            'r2_between': results.rsquared_between,
            'r2_overall': results.rsquared,
            'params': results.params.to_dict(),
            'pvalues': results.pvalues.to_dict(),
            'nobs': results.nobs,
            'base_features': base_features,
            'features_used': lag_features,
            'target': target,
            'estimator': 'linearmodels_panel_ols_entity_time_fe_lagged',
            'leakage_control': 'uses_lagged_t_minus_1_features_only',
            'dropped_rows_due_to_lag': int(rows_before - rows_after),
        }
        
        return results_dict

    def province_dimension_breakdown(self, province_name="Đắk Nông"):
        """Return latest PAPI/PCI dimension breakdown for one province."""
        if self.province_data is None:
            self.load_data()

        df = self.province_data.copy()
        if "province" not in df.columns or "year" not in df.columns:
            return {"status": "missing_province_or_year_columns"}

        target_name = province_name.casefold()
        province_rows = df[df["province"].astype(str).str.casefold() == target_name]
        if province_rows.empty:
            # Some Windows-created demo files can contain mojibake. Fall back to
            # a readable ASCII alias when exact Unicode does not match.
            province_rows = df[
                df["province"].astype(str).str.contains("Nông|Nong|NÃ´ng", case=False, regex=True, na=False)
                & df["province"].astype(str).str.contains("Đắk|Dak|Ä", case=False, regex=True, na=False)
            ]
        if province_rows.empty:
            return {
                "status": "not_found",
                "province_requested": province_name,
                "available_examples": df["province"].dropna().astype(str).head(10).tolist(),
            }

        latest_year = int(province_rows["year"].max())
        latest = province_rows[province_rows["year"] == latest_year].iloc[0]
        latest_all = df[df["year"] == latest_year].copy()

        dimension_cols = [
            "bribery_people",
            "admin_procedure",
            "vertical_accountability",
            "transparency",
            "citizen_participation",
            "papi_score",
            "pci_transparency",
            "grdp_index",
            "goal16",
        ]
        available = [col for col in dimension_cols if col in latest_all.columns]

        dimensions = {}
        for col in available:
            values = pd.to_numeric(latest_all[col], errors="coerce")
            current = float(pd.to_numeric(pd.Series([latest[col]]), errors="coerce").iloc[0])
            national_mean = float(values.mean())
            rank_desc = int((values > current).sum() + 1)
            dimensions[col] = {
                "value": current,
                "national_mean": national_mean,
                "delta_vs_mean": current - national_mean,
                "rank_desc": rank_desc,
                "province_count": int(values.notna().sum()),
            }

        return {
            "status": "ready",
            "province": str(latest["province"]),
            "year": latest_year,
            "dimensions": dimensions,
        }
    
    def create_heatmap(self, save_path=None):
        """Create heatmap by dimension"""
        if self.province_data is None:
            self.load_data()
        
        plt.figure(figsize=(14, 10))
        
        # Get latest year
        latest_year = self.province_data['year'].max()
        latest_data = self.province_data[self.province_data['year'] == latest_year]
        
        # Select dimensions
        dim_cols = ['bribery_people', 'admin_procedure', 'vertical_accountability', 
                   'transparency', 'citizen_participation']
        
        # Filter existing columns
        existing_dims = [col for col in dim_cols if col in latest_data.columns]
        
        if not existing_dims:
            logger.warning("No dimension columns found, using available numeric columns")
            numeric_cols = latest_data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['year', 'goal16', 'papi_score', 'grdp_index', 'population', 
                           'population_density', 'pci_transparency', 'pci_time_cost', 
                           'pci_informal_charges', 'pci_legal_institutions']
            existing_dims = [col for col in numeric_cols if col not in exclude_cols][:5]
        
        # Get top 20 provinces by goal16
        if 'goal16' in latest_data.columns:
            top_provinces = latest_data.nlargest(20, 'goal16')['province'].values
        else:
            top_provinces = latest_data.iloc[:20]['province'].values
        
        # Create matrix
        matrix = latest_data[latest_data['province'].isin(top_provinces)][existing_dims].values
        
        # Create heatmap
        plt.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        plt.colorbar(label='Score')
        plt.title(f'SDG16 Dimensions by Province ({latest_year})')
        plt.xlabel('Dimensions')
        plt.ylabel('Provinces')
        plt.xticks(range(len(existing_dims)), existing_dims, rotation=45)
        plt.yticks(range(len(top_provinces)), top_provinces)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        plt.close()
        return plt.gcf()
    
    def save_results(self, path="artifacts/subnational"):
        """Save subnational analysis results"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        rankings = self.rank_provinces()
        panel_results = self.run_panel_regression()
        mapping = self.map_sdg_to_papi()
        dak_nong_breakdown = self.province_dimension_breakdown("Đắk Nông")
        
        # Get top 10 weakest with scores
        weakest_data = []
        for rank, province in enumerate(rankings['top_10_weakest'], 1):
            score = rankings['rankings'][province]
            weakest_data.append({
                'province': province,
                'score': float(score),
                'rank': rank
            })
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'rankings': {
                'metric': rankings['metric'],
                'year': int(rankings['year']),
                'top_10_strongest': rankings['top_10_strongest'],
                'top_10_weakest': rankings['top_10_weakest'],
                'scores': {k: float(v) for k, v in rankings['rankings'].items()}
            },
            'panel_regression': panel_results,
            'mapping': mapping,
            'dak_nong_breakdown': dak_nong_breakdown,
            'top_10_weakest_details': weakest_data
        }
        
        with open(f"{path}/subnational_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("SUBNATIONAL ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Metric used for ranking: {rankings['metric']}")
        logger.info(f"Year: {rankings['year']}")
        logger.info(f"\nTop 5 Strongest Provinces:")
        for i, p in enumerate(rankings['top_10_strongest'][:5], 1):
            logger.info(f"  {i}. {p}: {rankings['rankings'][p]:.4f}")
        logger.info(f"\nTop 5 Weakest Provinces:")
        for i, p in enumerate(rankings['top_10_weakest'][:5], 1):
            logger.info(f"  {i}. {p}: {rankings['rankings'][p]:.4f}")
        logger.info(f"\nPanel Regression R²: {panel_results['r2_overall']:.4f}")
        logger.info("="*60)

# Example usage
if __name__ == "__main__":
    analyzer = SubnationalAnalyzer()
    analyzer.load_data()
    
    # Run analysis
    rankings = analyzer.rank_provinces()
    panel_results = analyzer.run_panel_regression()
    
    # Create heatmap
    Path("figures").mkdir(parents=True, exist_ok=True)
    analyzer.create_heatmap("figures/provincial_heatmap.png")
    
    # Save results
    analyzer.save_results()
