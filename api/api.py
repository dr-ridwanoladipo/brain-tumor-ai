"""
🧠 Brain Tumor Segmentation API - FastAPI REST API
Professional API for serving precomputed brain tumor analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
import uvicorn
import traceback
from typing import Any, Dict, List
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
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
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

class TumorVolumes(BaseModel):
    whole_tumor_cm3: float
    tumor_core_cm3: float
    enhancing_tumor_cm3: float

class ClinicalReport(BaseModel):
    patient_id: str
    analysis_date: str
    tumor_volumes: TumorVolumes
    ai_confidence: str
    clinical_urgency: str
    recommendation: str
    technical_notes: str

class ModelPerformance(BaseModel):
    whole_tumor_dice: str
    tumor_core_dice: str
    enhancing_tumor_dice: str
    average_inference_time: str
    training_time: str

class MetricsSummary(BaseModel):
    model_name: str
    version: str
    architecture: str
    performance_metrics: ModelPerformance
    test_volumes: int
    timestamp: str

class RobustnessSummary(BaseModel):
    noise_robustness: Dict[str, Dict[str, List[float]]]
    intensity_robustness: Dict[str, Dict[str, List[float]]]
    timestamp: str

class SegmentationData(BaseModel):
    case_id: str
    image_shape: List[int]
    label_shape: List[int]
    prediction_shape: List[int]
    modalities: List[str]
    message: str

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
@app.get("/", summary="Brain Tumor Segmentation API Overview", tags=["App Info"])
async def root():
    return {
        "app": "Brain Tumor Segmentation API",
        "purpose": "Serve precomputed brain tumor analysis results with clinical-grade precision.",
        "model": {
            "type": "nnU-Net 2025 (5-level U-Net)",
            "performance": {"WT_dice": "86.1%", "TC_dice": "77.8%", "ET_dice": "64.6%"},
            "training_data": "484 brain MRI volumes (Medical Segmentation Decathlon)"
        },
        "author": "Ridwan Oladipo, MD | AI Specialist",
        "version": "1.0.0",
        "documentation": "/docs",
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok" if data_loaded else "error",
        data_loaded=data_loaded,
        version="1.0.0",
        startup_time=startup_time,
        timestamp=current_time_iso(),
    )

@app.get("/cases", response_model=List[CaseInfo], tags=["Cases"])
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

@app.get("/case/{case_id}", response_model=CaseMetrics, tags=["Cases"])
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

@app.get("/clinical-report/{case_id}", response_model=ClinicalReport, tags=["Clinical"])
@limiter.limit("10/minute")
async def get_clinical_report(request: Request, case_id: str):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Clinical report requested: {case_id}")
        report = data_service.get_clinical_report(case_id)
        if not report:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Clinical report for case {case_id} not found")
        return ClinicalReport(
            tumor_volumes=TumorVolumes(**report['tumor_volumes']),
            **{k: v for k, v in report.items() if k != 'tumor_volumes'}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting clinical report: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve clinical report")

@app.post("/generate-report/{case_id}", response_model=ClinicalReport, tags=["Clinical"])
@limiter.limit("5/minute")
async def generate_clinical_report(request: Request, case_id: str):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Report generation requested: {case_id}")
        report = data_service.generate_clinical_report(case_id)
        if not report:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Cannot generate report for case {case_id}")
        return ClinicalReport(
            tumor_volumes=TumorVolumes(**report['tumor_volumes']),
            **{k: v for k, v in report.items() if k != 'tumor_volumes'}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating clinical report: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate clinical report")

@app.get("/metrics-summary", response_model=MetricsSummary, tags=["Performance"])
@limiter.limit("10/minute")
async def get_metrics_summary(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Metrics summary requested")
        metrics = data_service.get_metrics_summary()
        return MetricsSummary(**metrics, timestamp=current_time_iso())
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve metrics summary")

@app.get("/robustness-summary", response_model=RobustnessSummary, tags=["Performance"])
@limiter.limit("10/minute")
async def get_robustness_summary(request: Request):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info("Robustness summary requested")
        robustness = data_service.get_robustness_summary()
        return RobustnessSummary(
            noise_robustness=robustness.get('noise', {}),
            intensity_robustness=robustness.get('intensity', {}),
            timestamp=current_time_iso()
        )
    except Exception as e:
        logger.error(f"Error getting robustness summary: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve robustness summary")

@app.get("/case/{case_id}/segmentation", response_model=SegmentationData, tags=["Cases"])
@limiter.limit("5/minute")
async def get_case_segmentation(request: Request, case_id: str):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Segmentation data requested: {case_id}")
        seg_data = data_service.get_segmentation_info(case_id)
        if not seg_data:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Segmentation data for case {case_id} not found")
        return SegmentationData(**seg_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting segmentation data: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve segmentation data")

@app.get("/case/{case_id}/videos", tags=["Media"])
@limiter.limit("10/minute")
async def get_case_videos(request: Request, case_id: str):
    if not data_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Data service not loaded")
    try:
        logger.info(f"Video paths requested: {case_id}")
        videos = data_service.get_case_videos(case_id)
        if not videos:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Videos for case {case_id} not found")
        return videos
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting videos: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve video paths")

# ── Exception handlers ─────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"detail": f"Validation error: {exc}", "time": current_time_iso()})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Unexpected error occurred", "time": current_time_iso()})

# ── Uvicorn entry ─────────────────────────────
if __name__ == "__main__":
    uvicorn.run("brain_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
