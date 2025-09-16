"""
🧠 Brain Tumor Segmentation API - FastAPI REST API
Professional API for serving precomputed brain tumor analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
import uvicorn
from typing import Any, Dict, List
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel
from .model import initialize_data_service, data_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="Professional API for serving precomputed brain tumor analysis results",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def current_time_iso():
    return datetime.now().isoformat()

# ── State ─────────────────────────────
data_loaded = False
startup_time = None

# ── Models ─────────────────────────────
class HealthResponse(BaseModel):
    status: str
    data_loaded: bool
    startup_time: str
    timestamp: str
    version: str

class CaseInfo(BaseModel):
    case_id: str
    npz_file: str
    WT_dice: float
    TC_dice: float
    ET_dice: float
    WT_vol_true_cm3: float
    WT_vol_pred_cm3: float

class CaseMetrics(BaseModel):
    case_id: str
    test_file: str
    inference_time: float
    WT_dice: float
    TC_dice: float
    ET_dice: float
    WT_hausdorff: float
    TC_hausdorff: float
    ET_hausdorff: float
    WT_vol_true: float
    WT_vol_pred: float
    TC_vol_true: float
    TC_vol_pred: float
    ET_vol_true: float
    ET_vol_pred: float
    WT_vol_error: float
    TC_vol_error: float
    ET_vol_error: float

# ── Startup events ─────────────────────────────
@app.on_event("startup")
async def startup_event():
    global data_loaded, startup_time
    startup_time = current_time_iso()
    logger.info("Starting Brain Tumor API...")
    try:
        data_loaded = initialize_data_service()
        logger.info("Data service ready" if data_loaded else "Data service failed to load")
    except Exception as e:
        logger.error(f"Startup error: {e}")

# ── Routes ─────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok" if data_loaded else "error",
        data_loaded=data_loaded,
        version="1.0.0",
        startup_time=startup_time,
        timestamp=current_time_iso(),
    )

@app.get("/cases", response_model=List[CaseInfo])
@limiter.limit("10/minute")
async def get_cases(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Cases list requested")
        cases = data_service.get_demo_cases()
        return [CaseInfo(**case) for case in cases]
    except Exception as e:
        logger.error(f"Error getting cases: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve cases")

@app.get("/case/{case_id}", response_model=CaseMetrics)
@limiter.limit("10/minute")
async def get_case_details(request: Request, case_id: str):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Case details requested: {case_id}")
        case_data = data_service.get_case_metrics(case_id)
        if not case_data:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Case {case_id} not found")
        return CaseMetrics(**case_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting case details: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve case details")
