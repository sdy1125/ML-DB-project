"""
Subnational Analysis API Routes
Provides provincial-level analysis endpoints
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subnational", tags=["subnational"])

# Subnational state
subnational_state = {
    "data": None,
    "rankings": None,
    "mapping": None,
    "is_ready": False
}

class ProvinceRanking(BaseModel):
    province: str
    goal16_score: float
    rank: int
    papi_score: Optional[float] = None
    pci_score: Optional[float] = None
    grdp: Optional[float] = None

class ProvinceDetail(BaseModel):
    province: str
    year: int
    goal16: float
    papi: float
    pci: float
    grdp: float
    dimensions: Dict[str, float]
    
class PolicyRecommendation(BaseModel):
    province: str
    priority: int
    dimension: str
    recommendation: str
    expected_impact: float

def load_subnational_data():
    """Load subnational data"""
    global subnational_state
    
    try:
        data_path = Path("data/subnational/sdg16_provinces.csv")
        
        # Create sample data if not exists
        if not data_path.exists():
            from src.sdg16_pipeline.phase5_subnational_drilldown.subnational_analyzer import (
                SubnationalAnalyzer,
            )
            analyzer = SubnationalAnalyzer()
            analyzer.load_data()
        
        # Load data
        df = pd.read_csv(data_path)
        subnational_state["data"] = df
        subnational_state["is_ready"] = True
        
        logger.info(f"Loaded {len(df)} rows of provincial data")
        return True
        
    except Exception as e:
        logger.error(f"Error loading subnational data: {e}")
        subnational_state["is_ready"] = False
        return False

# Load on startup
load_subnational_data()

@router.get("/status", response_model=Dict[str, Any])
async def get_status():
    """Get subnational analysis status"""
    return {
        "is_ready": subnational_state["is_ready"],
        "province_count": len(subnational_state["data"]) // 5 if subnational_state["data"] is not None else 0,
        "year_range": [2019, 2023]
    }

@router.get("/provinces", response_model=List[ProvinceRanking])
async def get_provinces():
    """Get list of all provinces with rankings"""
    if not subnational_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subnational data not ready"
        )
    
    try:
        df = subnational_state["data"]
        
        # Calculate average scores
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        # Sort by goal16
        sorted_data = latest_data.sort_values('goal16', ascending=False)
        rankings = []
        
        for idx, (_, row) in enumerate(sorted_data.iterrows(), 1):
            rankings.append(ProvinceRanking(
                province=row['province'],
                goal16_score=row['goal16'],
                rank=idx,
                papi_score=row.get('PAPI'),
                pci_score=row.get('PCI'),
                grdp=row.get('GRDP')
            ))
        
        return rankings
        
    except Exception as e:
        logger.error(f"Error getting provinces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provinces: {str(e)}"
        )

@router.get("/weakest", response_model=List[ProvinceRanking])
async def get_weakest_provinces(limit: int = 10):
    """Get weakest provinces"""
    if not subnational_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subnational data not ready"
        )
    
    try:
        rankings = await get_provinces()
        return rankings[-limit:][::-1]  # Reverse to show worst first
        
    except Exception as e:
        logger.error(f"Error getting weakest provinces: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get weakest provinces: {str(e)}"
        )

@router.get("/province/{province_name}", response_model=ProvinceDetail)
async def get_province_detail(province_name: str):
    """Get detailed data for a specific province"""
    if not subnational_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subnational data not ready"
        )
    
    try:
        df = subnational_state["data"]
        province_data = df[df['province'] == province_name]
        
        if province_data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Province '{province_name}' not found"
            )
        
        latest = province_data.iloc[-1]
        
        dimensions = {
            "Bribery_people": latest.get('Bribery_people', 0),
            "Admin_procedure": latest.get('Admin_procedure', 0),
            "Vertical_account": latest.get('Vertical_account', 0),
            "Transparency": latest.get('Transparency', 0),
            "Citizen_participation": latest.get('Citizen_participation', 0)
        }
        
        return ProvinceDetail(
            province=latest['province'],
            year=int(latest['year']),
            goal16=latest['goal16'],
            papi=latest.get('PAPI', 0),
            pci=latest.get('PCI', 0),
            grdp=latest.get('GRDP', 0),
            dimensions=dimensions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting province detail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get province detail: {str(e)}"
        )

@router.get("/recommendations/{province_name}", response_model=List[PolicyRecommendation])
async def get_recommendations(province_name: str):
    """Get policy recommendations for a specific province"""
    if not subnational_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subnational data not ready"
        )
    
    try:
        # Get province data
        detail = await get_province_detail(province_name)
        
        # Generate recommendations based on dimensions
        recommendations = []
        
        # Check each dimension
        dimensions = [
            ("Bribery_people", "Anti-corruption"),
            ("Admin_procedure", "Administrative Reform"),
            ("Vertical_account", "Accountability"),
            ("Transparency", "Transparency"),
            ("Citizen_participation", "Citizen Participation")
        ]
        
        # Prioritize dimensions with lowest scores
        sorted_dims = sorted(dimensions, key=lambda x: detail.dimensions.get(x[0], 0))
        
        for i, (dim_key, dim_name) in enumerate(sorted_dims[:3], 1):
            recommendations.append(PolicyRecommendation(
                province=province_name,
                priority=i,
                dimension=dim_name,
                recommendation=f"Improve {dim_name} by enhancing local governance",
                expected_impact=0.05 * (4 - i)  # Higher impact for higher priority
            ))
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )

@router.get("/map")
async def get_choropleth_map():
    """Generate choropleth map of provinces"""
    if not subnational_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subnational data not ready"
        )
    
    try:
        # This would normally generate a map
        # For now, return a placeholder image
        plt.figure(figsize=(12, 8))
        
        # Mock choropleth
        data = np.random.rand(10, 10)
        plt.imshow(data, cmap='RdYlGn', interpolation='nearest')
        plt.colorbar(label='SDG16 Score')
        plt.title('Provincial SDG16 Performance')
        plt.axis('off')
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {"image": image_base64, "description": "Choropleth map placeholder"}
        
    except Exception as e:
        logger.error(f"Error generating map: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate map: {str(e)}"
        )

@router.get("/heatmap")
async def get_heatmap():
    """Generate heatmap by dimension"""
    if not subnational_state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subnational data not ready"
        )
    
    try:
        df = subnational_state["data"]
        
        # Prepare data for heatmap
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        # Select dimensions
        dim_cols = ['Bribery_people', 'Admin_procedure', 'Vertical_account', 
                   'Transparency', 'Citizen_participation']
        
        # Get top 20 provinces by goal16
        top_provinces = latest_data.nlargest(20, 'goal16')['province'].values
        
        # Create matrix
        matrix = latest_data[latest_data['province'].isin(top_provinces)][dim_cols].values
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        plt.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        plt.colorbar(label='Score')
        plt.title('SDG16 Dimensions by Province')
        plt.xlabel('Dimensions')
        plt.ylabel('Provinces')
        plt.xticks(range(len(dim_cols)), dim_cols, rotation=45)
        plt.yticks(range(len(top_provinces)), top_provinces)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {"image": image_base64, "description": "Heatmap by dimension"}
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate heatmap: {str(e)}"
        )

@router.post("/reload")
async def reload_data():
    """Reload subnational data"""
    success = load_subnational_data()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload subnational data"
        )
    return {"message": "Subnational data reloaded successfully"}
