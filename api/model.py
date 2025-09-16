"""
🧠 Brain Tumor Segmentation - Data Service Module
Precomputed data serving functions for FastAPI backend.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainDataService:
    """Brain Tumor Analysis Data Service - Serves precomputed results."""

    def __init__(self):
        """Initialize empty placeholders; populate via load_data()."""
        self.manifest = None
        self.results_df = None
        self.sample_reports = None
        self.robustness_data = None
        self.model_card = None
        self.data_path = Path("evaluation_results")