"""
🧠 Brain Tumor Segmentation UI - Helper Functions
by Ridwan Oladipo, MD | AI Specialist

All reusable functions for the Streamlit UI application
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time


def load_custom_css():
    """Load custom CSS for professional medical interface"""
    st.markdown("""
    <style>
    /* Import medical-grade fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit's default chrome */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}

    /* Reduce top/bottom padding of main container */
    div.block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0rem !important;
        margin-bottom: 7rem !important;
        /* max-width: 1200px; */
    }

    /* Medical header styling */
    .medical-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .medical-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .medical-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Status indicators */
    .success-indicator {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 0.5rem 0;
    }

    .warning-indicator {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 0.5rem 0;
    }

    /* Patient selection */
    .patient-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .patient-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
    }

    .patient-card.selected {
        border-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }

    /* Clinical report styling */
    .clinical-report {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .clinical-report h4 {
        color: #0c4a6e;
        margin-bottom: 1rem;
    }

    /* MRI viewer styling */
    .mri-viewer {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }

    /* Prediction button */
    .predict-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }

    .predict-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(239, 68, 68, 0.3);
    }

    /* Footer styling */
    .medical-footer {
        background: #1f2937;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)


def load_demo_data():
    """Load all demo data files"""
    try:
        # Base path to evaluation results
        base_path = Path('evaluation_results')

        # Load manifest
        with open(base_path / 'demo_manifest.json', 'r') as f:
            manifest = json.load(f)

        # Load evaluation results
        results_df = pd.read_csv(base_path / 'detailed_evaluation_results.csv')

        # Load sample reports
        with open(base_path / 'sample_clinical_reports.json', 'r') as f:
            sample_reports = json.load(f)

        # Load robustness analysis
        with open(base_path / 'robustness_analysis.json', 'r') as f:
            robustness_data = json.load(f)

        # Load model card
        with open(base_path / 'model_card.json', 'r') as f:
            model_card = json.load(f)

        return manifest, results_df, sample_reports, robustness_data, model_card
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None, None, None


def load_npz_case(case_file):
    """Load a specific NPZ case file"""
    try:
        base_path = Path("evaluation_results")
        with np.load(base_path / case_file) as data:
            return {
                'image': data['image'],
                'label': data['label'],
                'prediction': data['prediction']
            }
    except FileNotFoundError:
        st.error(f"Case file not found: {base_path / case_file}")
        return None


def display_footer():
    """Display professional footer with responsive layout"""
    st.markdown("""
    <style>
    .footer-links {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
    }
    .footer-links a {
        color: white;
        margin: 5px 0;
        text-decoration: none;
    }
    @media (max-width: 768px) {
        .footer-links {
            flex-direction: column;
            align-items: center;
        }
    }
    </style>
    <div class="medical-footer">
        <h4>🔗 Project Links</h4>
        <div class="footer-links">
            <a href="https://github.com/dr-ridwanoladipo/brain-tumor-ai">💻 GitHub Repository</a>
            <a href="https://www.kaggle.com/code/ridwanoladipoai/nnunet-brain-tumor-preprocessing">📊 Preprocessing Notebook</a>
            <a href="https://www.kaggle.com/code/ridwanoladipoai/nnunet-brain-tumor-training">🚀 Training Notebook</a>
            <a href="https://www.kaggle.com/code/ridwanoladipoai/nnunet-brain-tumor-evaluation">📈 Evaluation Notebook</a>
        </div>
        <br>
        <p>© 2025 Ridwan Oladipo, MD | Medical AI Specialist</p>
        <p><strong>🏥 Advanced Healthcare AI Solutions</strong></p>
        <p style="font-size: 0.9rem; opacity: 0.8;">
            ⚠️ This AI tool is for research demonstration only and not approved for clinical diagnosis.
            All medical decisions should be made in consultation with qualified healthcare providers.
        </p>
    </div>
    """, unsafe_allow_html=True)