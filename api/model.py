"""
🧠 Brain Tumor Segmentation - Data Service Module
Data serving functions for FastAPI backend.

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

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                            Core Data Service Class
# ═══════════════════════════════════════════════════════════════════════════════
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

    # ───────────────────────────────────────────────────────────────────────────
    # Data loading
    # ───────────────────────────────────────────────────────────────────────────
    def load_data(self) -> bool:
        """Load all data files."""
        try:
            logger.info("Loading brain tumor analysis data...")

            # Load manifest
            with open(self.data_path / 'demo_manifest.json', 'r') as f:
                self.manifest = json.load(f)

            # Load evaluation results
            self.results_df = pd.read_csv(self.data_path / 'detailed_evaluation_results.csv')

            # Load sample reports
            with open(self.data_path / 'sample_clinical_reports.json', 'r') as f:
                self.sample_reports = json.load(f)

            # Load robustness analysis
            with open(self.data_path / 'robustness_analysis.json', 'r') as f:
                self.robustness_data = json.load(f)

            # Load model card
            with open(self.data_path / 'model_card.json', 'r') as f:
                self.model_card = json.load(f)

            logger.info("All data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False

    # ───────────────────────────────────────────────────────────────────────────
    # Demo cases
    # ───────────────────────────────────────────────────────────────────────────
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

    def get_segmentation_info(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get segmentation data information for a case."""
        # Check if NPZ file exists
        npz_file = f"inferenced_{case_id}_preprocessed.npz"
        npz_path = self.data_path / npz_file

        if not npz_path.exists():
            return None

        try:
            # Load NPZ to get shape information
            with np.load(npz_path) as data:
                image_shape = data['image'].shape
                label_shape = data['label'].shape
                prediction_shape = data['prediction'].shape

            return {
                'case_id': case_id,
                'image_shape': list(image_shape),
                'label_shape': list(label_shape),
                'prediction_shape': list(prediction_shape),
                'modalities': ['FLAIR', 'T1w', 'T1Gd', 'T2w'],
                'message': f"NPZ data available for case {case_id}"
            }
        except Exception as e:
            logger.error(f"Error loading segmentation info for {case_id}: {e}")
            return None

    # ───────────────────────────────────────────────────────────────────────────
    # Clinical reports
    # ───────────────────────────────────────────────────────────────────────────
    def get_clinical_report(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get clinical report for a specific case."""
        if not self.sample_reports:
            return None

        # Find report by patient_id
        for report in self.sample_reports:
            if report.get('patient_id') == case_id:
                return report

        return None

    def generate_clinical_report(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Generate clinical report (returns precomputed result)."""
        # For this API, "generate" returns existing report
        return self.get_clinical_report(case_id)

    # ───────────────────────────────────────────────────────────────────────────
    # Performance metrics
    # ───────────────────────────────────────────────────────────────────────────
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

    # ───────────────────────────────────────────────────────────────────────────
    # Media files
    # ───────────────────────────────────────────────────────────────────────────
    def get_case_videos(self, case_id: str) -> Optional[Dict[str, str]]:
        """Get video file paths for a case (if available)."""
        # Check if videos exist for the case
        plain_video = self.data_path / f"{case_id}_cine_plain.mp4"
        overlay_video = self.data_path / f"{case_id}_cine_overlay.mp4"

        videos = {}

        if plain_video.exists():
            videos['cine_plain'] = f"{case_id}_cine_plain.mp4"

        if overlay_video.exists():
            videos['cine_overlay'] = f"{case_id}_cine_overlay.mp4"

        if not videos:
            return None

        videos['case_id'] = case_id
        videos['message'] = f"Video files available for case {case_id}"

        return videos

    # ───────────────────────────────────────────────────────────────────────────
    # Data validation
    # ───────────────────────────────────────────────────────────────────────────
    def validate_data(self) -> Dict[str, bool]:
        """Validate that all required data is loaded."""
        return {
            'manifest_loaded': self.manifest is not None,
            'results_loaded': self.results_df is not None,
            'reports_loaded': self.sample_reports is not None,
            'robustness_loaded': self.robustness_data is not None,
            'model_card_loaded': self.model_card is not None
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        if not self.manifest or self.results_df is None:
            return {'error': 'Data not loaded'}

        return {
            'total_cases': len(self.manifest),
            'demo_cases': len(self.manifest),
            'evaluation_results': len(self.results_df),
            'clinical_reports': len(self.sample_reports) if self.sample_reports else 0,
            'robustness_tests_available': bool(self.robustness_data),
            'model_info_available': bool(self.model_card)
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                        Global instance + Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════
data_service = BrainDataService()


def initialize_data_service() -> bool:
    """Initialize the global data service instance."""
    return data_service.load_data()


def get_data_service() -> BrainDataService:
    """Get the global data service instance."""
    return data_service


# ── Back-compat helper functions (if needed) ──────────────────────────────────
def load_demo_data():
    """Load demo data (backwards compatibility)."""
    if not data_service.manifest:
        data_service.load_data()
    return (
        data_service.manifest,
        data_service.results_df,
        data_service.sample_reports,
        data_service.robustness_data,
        data_service.model_card
    )


def get_case_data(case_id: str):
    """Get case data (backwards compatibility)."""
    return data_service.get_case_metrics(case_id)


def get_clinical_report_data(case_id: str):
    """Get clinical report (backwards compatibility)."""
    return data_service.get_clinical_report(case_id)