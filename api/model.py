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

    def load_data(self) -> bool:
        """Load all precomputed data files."""
        try:
            logger.info("Loading brain tumor analysis data...")

            with open(self.data_path / 'demo_manifest.json', 'r') as f:
                self.manifest = json.load(f)

            self.results_df = pd.read_csv(self.data_path / 'detailed_evaluation_results.csv')

            with open(self.data_path / 'sample_clinical_reports.json', 'r') as f:
                self.sample_reports = json.load(f)

            with open(self.data_path / 'robustness_analysis.json', 'r') as f:
                self.robustness_data = json.load(f)

            with open(self.data_path / 'model_card.json', 'r') as f:
                self.model_card = json.load(f)

            logger.info("All data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False

    def get_demo_cases(self) -> List[Dict[str, Any]]:
        """Return list of all demo cases."""
        if not self.manifest:
            return []
        return self.manifest

    def get_case_metrics(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific case."""
        if self.results_df is None:
            return None
        case_row = self.results_df[self.results_df['case_id'] == case_id]
        if case_row.empty:
            return None
        return case_row.iloc[0].to_dict()

    def get_clinical_report(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get clinical report for a specific case."""
        if not self.sample_reports:
            return None
        for report in self.sample_reports:
            if report.get('patient_id') == case_id:
                return report
        return None

    def generate_clinical_report(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Generate/retrieve clinical report (returns precomputed result)."""
        return self.get_clinical_report(case_id)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get model performance summary."""
        if not self.model_card:
            return {}
        return {
            'model_name': self.model_card.get('model_name', 'nnU-Net 2025'),
            'version': self.model_card.get('version', '1.0'),
            'architecture': self.model_card.get('architecture', '5-level U-Net'),
            'performance_metrics': self.model_card.get('performance_metrics', {}),
            'test_volumes': len(self.manifest) if self.manifest else 0
        }

    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get robustness analysis results."""
        if not self.robustness_data:
            return {'noise': {}, 'intensity': {}}
        return self.robustness_data

# ── Global service instance ───────────────────────────────
data_service = BrainDataService()

def initialize_data_service() -> bool:
    return data_service.load_data()
