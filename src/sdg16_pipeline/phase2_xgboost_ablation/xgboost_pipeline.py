"""
XGBoost Pipeline with advanced feature engineering
Includes: Lag-1, Lag-2, Rolling mean, YoY delta, Country trends
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
try:
    import optuna
except ImportError:  # Optuna is optional for local report/training runs.
    optuna = None
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_XGBOOST_BASELINE_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200,
    'random_state': 42
}

TUNED_XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 500,
    'min_child_weight': 5,
    'reg_lambda': 5.0,
    'random_state': 42
}

class XGBoostPipeline:
    def __init__(self, config_path="configs/project.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances = None
        self.metrics = {}
        self.data_path = "data/clean/sdg16_spark.csv"
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            logger.warning("Config file not found, using defaults")
            return {}
    
    def load_data(self):
        """Load data from CSV file"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Convert column names to lowercase and clean
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Show available columns for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Identify target column
        target_col = None
        possible_targets = ['goal16', 'goal_16', 'sdg16', 'sdg16_score']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Try to find any column with 'goal' or 'sdg' in name
            for col in df.columns:
                if 'goal' in col or 'sdg' in col:
                    target_col = col
                    break
        
        if target_col is None:
            raise ValueError("Could not find target column. Available columns: " + str(df.columns.tolist()))
        
        logger.info(f"Target column: {target_col}")
        return df, target_col
    
    def _get_numeric_columns(self, df, exclude_cols):
        """Get only numeric columns, excluding specified columns"""
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return numeric_cols
    
    def add_lag_features(self, df, columns, lags=[1, 2]):
        """Add lag features to dataframe"""
        df_sorted = df.sort_values(['country', 'year'])
        for col in columns:
            for lag in lags:
                df_sorted[f'{col}_lag_{lag}'] = df_sorted.groupby('country')[col].shift(lag)
        return df_sorted
    
    def add_rolling_features(self, df, columns, window=3):
        """Add rolling mean features"""
        df_sorted = df.sort_values(['country', 'year'])
        for col in columns:
            try:
                df_sorted[f'{col}_rolling_{window}'] = df_sorted.groupby('country')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            except Exception as e:
                logger.warning(f"Could not add rolling feature for {col}: {e}")
        return df_sorted
    
    def add_yoy_delta(self, df, columns):
        """Add year-over-year delta features"""
        df_sorted = df.sort_values(['country', 'year'])
        for col in columns:
            try:
                df_sorted[f'{col}_yoy'] = df_sorted.groupby('country')[col].pct_change()
            except Exception as e:
                logger.warning(f"Could not add YoY feature for {col}: {e}")
        return df_sorted
    
    def add_country_trends(self, df, target_col='goal16'):
        """Add country-specific trends"""
        df_sorted = df.sort_values(['country', 'year'])
        try:
            df_sorted['country_trend'] = df_sorted.groupby('country')[target_col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        except Exception as e:
            logger.warning(f"Could not add country trend: {e}")
        return df_sorted
    
    def feature_engineering(self, df, target_col='goal16'):
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering...")
        logger.info(f"Initial shape: {df.shape}")
        
        # Exclude non-feature columns
        exclude_cols = ['country', 'year', target_col, 'source']
        
        # Get only numeric feature columns
        feature_cols = self._get_numeric_columns(df, exclude_cols)
        
        logger.info(f"Found {len(feature_cols)} numeric feature columns: {feature_cols[:5]}...")
        
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found! Please check your data.")
        
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Add features (only on numeric columns)
        df_copy = self.add_lag_features(df_copy, feature_cols, lags=[1, 2])
        df_copy = self.add_rolling_features(df_copy, feature_cols, window=3)
        df_copy = self.add_yoy_delta(df_copy, feature_cols)
        df_copy = self.add_country_trends(df_copy, target_col)
        
        # Drop rows with NaN (from lag/rolling operations)
        df_before_drop = len(df_copy)
        df_copy = df_copy.dropna()
        df_after_drop = len(df_copy)
        
        logger.info(f"Dropped {df_before_drop - df_after_drop} rows with NaN values")
        logger.info(f"Remaining rows: {df_after_drop}")
        
        # If too few rows remain, use less aggressive feature engineering
        if df_after_drop < 100:
            logger.warning("Too few rows remaining after full feature engineering. Using simpler approach...")
            
            # Reset and use only basic features
            df_copy = df.copy()
            
            # Only add lag-1 and rolling mean (less aggressive)
            df_copy = self.add_lag_features(df_copy, feature_cols, lags=[1])
            df_copy = self.add_rolling_features(df_copy, feature_cols, window=2)
            
            # Drop NaN but keep more data
            df_copy = df_copy.dropna()
            logger.info(f"After simpler feature engineering: {len(df_copy)} rows")
        
        logger.info(f"Feature engineering complete. Shape: {df_copy.shape}")
        
        # Store feature columns for later
        self.feature_cols = feature_cols
        
        return df_copy, feature_cols
    
    def prepare_data(self, df, target_col='goal16'):
        """Prepare features and target for training"""
        # Feature engineering
        df, feature_cols = self.feature_engineering(df, target_col)
        
        if len(df) == 0:
            # Last resort: use original data without feature engineering
            logger.warning("No data after feature engineering. Using original data without engineered features...")
            df = df.copy()
            
            # Use only original numeric columns
            exclude_cols = ['country', 'year', target_col, 'source']
            feature_cols = self._get_numeric_columns(df, exclude_cols)
            
            if len(feature_cols) == 0:
                raise ValueError("No data remaining after feature engineering. Please check your data.")
            
            # Use the data as is
            X = df[feature_cols]
            y = df[target_col]
            metadata = df[['country', 'year']]
            
            self.feature_names = feature_cols
            logger.info(f"Using {len(feature_cols)} original features, {len(X)} samples")
            return X, y, metadata
        
        # Separate features and target
        exclude_cols = ['country', 'year', target_col, 'source']
        feature_cols_final = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols_final]
        y = df[target_col]
        
        # Store feature names for later
        self.feature_names = feature_cols_final
        
        logger.info(f"Final feature count: {len(feature_cols_final)}")
        logger.info(f"Training samples: {len(X)}")
        
        return X, y, df[['country', 'year']]
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=20):
        """Optimize XGBoost hyperparameters using Optuna"""

        if optuna is None:
            logger.warning(
                "Optuna is not installed. Using tuned XGBoost parameters from "
                "artifacts/core_model_tuning instead."
            )
            return TUNED_XGBOOST_PARAMS.copy()
        
        if len(X_train) < 50:
            logger.warning("Too few samples for hyperparameter optimization. Using default parameters.")
            return PROJECT_XGBOOST_BASELINE_PARAMS.copy()
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'random_state': 42
            }
            
            # Simple validation split
            split_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            pred = model.predict(X_val)
            score = np.sqrt(mean_squared_error(y_val, pred))
            
            return score
        
        study = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
        study.optimize(objective, n_trials=n_trials, timeout=200)
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        
        return study.best_params
    
    def train(self, optimize=True):
        """Train XGBoost model"""
        # Load data
        df, target_col = self.load_data()
        
        logger.info("Preparing data...")
        X, y, metadata = self.prepare_data(df, target_col)
        
        # Train-test split by time
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        self.train_metadata = metadata.iloc[:split_idx]
        self.test_metadata = metadata.iloc[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters
        if optimize:
            best_params = self.optimize_hyperparameters(X_train_scaled, y_train)
        else:
            best_params = PROJECT_XGBOOST_BASELINE_PARAMS.copy()
        
        # Train model
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBRegressor(**best_params)
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'best_params': best_params,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Feature importance
        self.feature_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Training complete. Metrics: {self.metrics}")
        logger.info(f"Top 5 features: {self.feature_importances.head(5)[['feature', 'importance']].to_dict()}")
        
        return self.metrics
    
    def ablation_study(self, top_n=3):
        """Remove top N features and retrain to see impact"""
        if len(self.feature_importances) < top_n:
            logger.warning(f"Not enough features for ablation study. Need at least {top_n} features.")
            return None
        
        logger.info(f"Running ablation study: removing top {top_n} features")
        
        df, target_col = self.load_data()
        X, y, _ = self.prepare_data(df, target_col)
        
        # Get top features
        top_features = self.feature_importances.head(top_n)['feature'].tolist()
        logger.info(f"Removing features: {top_features}")
        
        # Remove top features
        X_ablation = X.drop(columns=top_features)
        
        # Train new model
        split_idx = int(len(X_ablation) * 0.8)
        X_train, X_test = X_ablation.iloc[:split_idx], X_ablation.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = xgb.XGBRegressor(**self.metrics['best_params'])
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        ablation_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'removed_features': top_features
        }
        
        logger.info(f"Ablation metrics: {ablation_metrics}")
        return ablation_metrics
    
    def save_model(self, path="artifacts/xgboost"):
        """Save model and artifacts"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"{path}/xgboost_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
        # Save metadata
        metadata = {
            'metrics': self.metrics,
            'feature_importances': self.feature_importances.to_dict(),
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'data_path': self.data_path
        }
        
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path="artifacts/xgboost"):
        """Load model and artifacts"""
        self.model = joblib.load(f"{path}/xgboost_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
            self.metrics = metadata['metrics']
            self.feature_importances = pd.DataFrame(metadata['feature_importances'])
            self.feature_names = metadata['feature_names']
        
        logger.info(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Train model
    pipeline = XGBoostPipeline()
    metrics = pipeline.train(optimize=True)
    
    # Run ablation study
    ablation_metrics = pipeline.ablation_study(top_n=3)
    if ablation_metrics:
        logger.info(f"Ablation R²: {ablation_metrics['r2']:.4f}")
    
    # Save model
    pipeline.save_model()
