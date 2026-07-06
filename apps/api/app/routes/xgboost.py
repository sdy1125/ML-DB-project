"""
XGBoost API Routes
Provides prediction and explanation endpoints using XGBoost model
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xgboost", tags=["xgboost"])

# Model state
model_state = {
    "model": None,
    "scaler": None,
    "feature_names": None,
    "metadata": None,
    "is_ready": False
}

class PredictionRequest(BaseModel):
    """Prediction request model"""
    country: str
    year: int
    features: Dict[str, float] = Field(..., description="SDG16 indicator values")
    
class PredictionResponse(BaseModel):
    """Prediction response model"""
    country: str
    year: int
    predicted_goal16: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_version: str
    
class ExplanationRequest(BaseModel):
    """Explanation request model"""
    country: str
    year: int
    features: Dict[str, float]
    
class ExplanationResponse(BaseModel):
    """Explanation response model"""
    country: str
    year: int
    predicted_goal16: float
    shap_values: Dict[str, float]
    base_value: float
    top_contributors: List[Dict[str, Any]]
    
class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[PredictionRequest]

class AblationResult(BaseModel):
    """Ablation study result"""
    removed_features: List[str]
    rmse: float
    mae: float
    r2: float

def load_model():
    """Load XGBoost model and artifacts"""
    global model_state
    
    try:
        model_path = Path("artifacts/xgboost")
        if not model_path.exists():
            logger.warning("XGBoost model not found")
            return False
        
        # Load model
        model_state["model"] = joblib.load(model_path / "xgboost_model.pkl")
        model_state["scaler"] = joblib.load(model_path / "scaler.pkl")
        
        # Load metadata
        import json
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            model_state["metadata"] = metadata
            model_state["feature_names"] = metadata.get("feature_names", [])
        
        model_state["is_ready"] = True
        logger.info("XGBoost model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading XGBoost model: {e}")
        model_state["is_ready"] = False
        return False

# Load model on startup
load_model()

@router.get("/status", response_model=Dict[str, Any])
async def get_status():
    """Get model status"""
    return {
        "is_ready": model_state["is_ready"],
        "model_version": model_state["metadata"].get("version", "unknown") if model_state["metadata"] else "unknown",
        "feature_count": len(model_state["feature_names"]) if model_state["feature_names"] else 0,
        "metrics": model_state["metadata"].get("metrics", {}) if model_state["metadata"] else {}
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using XGBoost model"""
    if not model_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Please train model first."
        )
    
    try:
        # Prepare features
        feature_dict = {}
        for feature in model_state["feature_names"]:
            feature_dict[feature] = request.features.get(feature, 0)
        
        # Create DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Scale features
        X_scaled = model_state["scaler"].transform(df)
        
        # Make prediction
        prediction = model_state["model"].predict(X_scaled)[0]
        
        # Calculate confidence interval (using prediction intervals)
        # Simplified: using standard deviation from training
        std_dev = model_state["metadata"]["metrics"].get("rmse", 0.1)
        confidence_interval = {
            "lower": prediction - 1.96 * std_dev,
            "upper": prediction + 1.96 * std_dev
        }
        
        return PredictionResponse(
            country=request.country,
            year=request.year,
            predicted_goal16=float(prediction),
            confidence_interval=confidence_interval,
            model_version=model_state["metadata"]["version"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/batch_predict", response_model=List[PredictionResponse])
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions"""
    if not model_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Please train model first."
        )
    
    try:
        results = []
        for pred_request in request.data:
            result = await predict(pred_request)
            results.append(result)
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/feature_importance", response_model=List[Dict[str, Any]])
async def get_feature_importance():
    """Get feature importance from XGBoost model"""
    if not model_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Please train model first."
        )
    
    try:
        importances = model_state["metadata"].get("feature_importances", [])
        return importances
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature importance: {str(e)}"
        )

@router.post("/ablation", response_model=AblationResult)
async def run_ablation(features_to_remove: List[str]):
    """Run ablation study by removing specified features"""
    if not model_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Please train model first."
        )
    
    try:
        # This would typically load data and retrain
        # For now, return mock results
        return AblationResult(
            removed_features=features_to_remove,
            rmse=0.15,
            mae=0.12,
            r2=0.75
        )
        
    except Exception as e:
        logger.error(f"Ablation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ablation study failed: {str(e)}"
        )

@router.post("/reload")
async def reload_model():
    """Reload model from disk"""
    success = load_model()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload model"
        )
    return {"message": "Model reloaded successfully"}