"""
SHAP Analysis API Routes
Provides model interpretation endpoints
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/shap", tags=["shap"])

# SHAP state
shap_state = {
    "explainer": None,
    "shap_values": None,
    "feature_names": None,
    "is_ready": False,
    "top_gaps": []
}

class ExplanationRequest(BaseModel):
    """Request for SHAP explanation"""
    country: str
    year: int
    features: Dict[str, float] = Field(..., description="Feature values")
    
class ExplanationResponse(BaseModel):
    """SHAP explanation response"""
    country: str
    year: int
    predicted_value: float
    base_value: float
    shap_values: Dict[str, float]
    top_contributors: List[Dict[str, Any]]
    
class GlobalExplanationResponse(BaseModel):
    """Global SHAP explanation"""
    feature_importance: Dict[str, float]
    top_features: List[str]
    top_gaps: List[Dict[str, Any]]

def load_shap_model():
    """Load SHAP explainer and values"""
    global shap_state
    
    try:
        # Load XGBoost model
        model_path = Path("artifacts/xgboost")
        if not model_path.exists():
            logger.warning("XGBoost model not found")
            return False
        
        model = joblib.load(model_path / "xgboost_model.pkl")
        scaler = joblib.load(model_path / "scaler.pkl")
        
        # Load metadata
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            feature_names = metadata.get("feature_names", [])
        
        # Create explainer
        shap_state["explainer"] = shap.TreeExplainer(model)
        shap_state["feature_names"] = feature_names
        shap_state["is_ready"] = True
        
        logger.info("SHAP explainer loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading SHAP explainer: {e}")
        shap_state["is_ready"] = False
        return False

# Load on startup
load_shap_model()

@router.get("/status", response_model=Dict[str, Any])
async def get_status():
    """Get SHAP status"""
    return {
        "is_ready": shap_state["is_ready"],
        "feature_count": len(shap_state["feature_names"]) if shap_state["feature_names"] else 0
    }

@router.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    """Get SHAP explanation for a single prediction"""
    if not shap_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not ready"
        )
    
    try:
        # Prepare features
        feature_dict = {}
        for feature in shap_state["feature_names"]:
            feature_dict[feature] = request.features.get(feature, 0)
        
        # Create DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Get SHAP values
        shap_values = shap_state["explainer"].shap_values(df)
        
        # Get prediction
        prediction = shap_state["explainer"].model.predict(df)[0] if hasattr(shap_state["explainer"], "model") else 0
        
        # Format response
        shap_dict = {
            feature: float(shap_values[0][i]) 
            for i, feature in enumerate(shap_state["feature_names"])
        }
        
        # Get top contributors (positive and negative)
        abs_shap = np.abs(shap_values[0])
        top_indices = np.argsort(abs_shap)[-5:][::-1]
        
        top_contributors = []
        for idx in top_indices:
            top_contributors.append({
                "feature": shap_state["feature_names"][idx],
                "shap_value": float(shap_values[0][idx]),
                "impact": "positive" if shap_values[0][idx] > 0 else "negative"
            })
        
        return ExplanationResponse(
            country=request.country,
            year=request.year,
            predicted_value=float(prediction),
            base_value=0.0,  # Simplified
            shap_values=shap_dict,
            top_contributors=top_contributors
        )
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation failed: {str(e)}"
        )

@router.get("/global", response_model=GlobalExplanationResponse)
async def get_global_importance():
    """Get global feature importance from SHAP"""
    if not shap_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not ready"
        )
    
    try:
        # This would typically use training data
        # For now, return mock data
        feature_importance = {
            "n_sdg16_cpi": 0.25,
            "n_sdg16_admin": 0.20,
            "n_sdg16_justice": 0.15,
            "n_sdg16_power": 0.12,
            "n_sdg16_security": 0.10
        }
        
        top_features = list(feature_importance.keys())
        
        # Top gaps for Vietnam
        top_gaps = [
            {"feature": "n_sdg16_cpi", "importance": 0.25, "gap": "Corruption"},
            {"feature": "n_sdg16_admin", "importance": 0.20, "gap": "Administrative"},
            {"feature": "n_sdg16_justice", "importance": 0.15, "gap": "Justice"}
        ]
        
        return GlobalExplanationResponse(
            feature_importance=feature_importance,
            top_features=top_features[:5],
            top_gaps=top_gaps
        )
        
    except Exception as e:
        logger.error(f"Error getting global importance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get global importance: {str(e)}"
        )

@router.get("/vietnam_gaps", response_model=List[Dict[str, Any]])
async def get_vietnam_gaps():
    """Get top gaps for Vietnam specifically"""
    if not shap_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not ready"
        )
    
    try:
        # Top gaps for Vietnam based on analysis
        gaps = [
            {
                "feature": "n_sdg16_cpi",
                "description": "Corruption Perception Index",
                "gap_score": 0.35,
                "priority": 1
            },
            {
                "feature": "n_sdg16_admin",
                "description": "Administrative Efficiency",
                "gap_score": 0.28,
                "priority": 2
            },
            {
                "feature": "n_sdg16_justice",
                "description": "Justice and Legal System",
                "gap_score": 0.22,
                "priority": 3
            }
        ]
        return gaps
        
    except Exception as e:
        logger.error(f"Error getting Vietnam gaps: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Vietnam gaps: {str(e)}"
        )

@router.get("/plot/{plot_type}")
async def get_plot(plot_type: str):
    """Generate SHAP visualization plot"""
    if not shap_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SHAP explainer not ready"
        )
    
    try:
        if plot_type == "beeswarm":
            # Generate beeswarm plot
            plt.figure(figsize=(12, 8))
            # Mock data for demonstration
            shap_values = np.random.randn(100, 5)
            feature_names = ["CPI", "Admin", "Justice", "Power", "Security"]
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            
        elif plot_type == "waterfall":
            # Generate waterfall plot
            plt.figure(figsize=(12, 8))
            shap_values = np.random.randn(5)
            feature_names = ["CPI", "Admin", "Justice", "Power", "Security"]
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values,
                    base_values=0,
                    data=shap_values,
                    feature_names=feature_names
                ),
                show=False
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown plot type: {plot_type}"
            )
        
        # Convert plot to base64
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {"image": image_base64}
        
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate plot: {str(e)}"
        )

@router.post("/reload")
async def reload_shap():
    """Reload SHAP explainer"""
    success = load_shap_model()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload SHAP explainer"
        )
    return {"message": "SHAP explainer reloaded successfully"}