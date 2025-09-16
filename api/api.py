"""
🧠 Brain Tumor Segmentation API - FastAPI REST API
Professional API for serving precomputed brain tumor analysis results.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

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