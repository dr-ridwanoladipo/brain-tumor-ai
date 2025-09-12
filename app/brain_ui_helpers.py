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


def create_mri_viewer(image_data, slice_idx=None, modality_idx=0):
    """Create MRI slice viewer"""
    if slice_idx is None:
        slice_idx = image_data.shape[2] // 2

    # Get slice
    slice_data = image_data[:, :, slice_idx, modality_idx]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=slice_data,
        colorscale='gray',
        showscale=False
    ))

    fig.update_layout(
        title=f"MRI Slice {slice_idx + 1} - {'FLAIR' if modality_idx == 0 else 'T1w' if modality_idx == 1 else 'T1Gd' if modality_idx == 2 else 'T2w'}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=400,
        height=400,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def create_segmentation_overlay(image_data, label_data, pred_data, slice_idx, modality_idx=0):
    """
    BraTS-standard visualization:
      - Ground Truth: semi-transparent fills (Edema=Green, Non-enhancing=Yellow, Enhancing=Red)
      - AI Prediction: Cyan dashed outline for Whole Tumor
    """
    # Extract slice
    base_slice = image_data[:, :, slice_idx, modality_idx]
    gt_slice = label_data[:, :, slice_idx]

    # Collapse AI prediction to WT
    pred_wt_slice = (pred_data[:, :, slice_idx] > 0).astype(np.uint8)

    fig = go.Figure()

    # Base MRI
    fig.add_trace(go.Heatmap(
        z=base_slice,
        colorscale="gray",
        showscale=False,
        name="MRI"
    ))

    # Ground Truth regions
    gt_colors = {
        1: ("rgba(0,255,0,0.65)", "Edema (GT)"),
        2: ("rgba(255,255,0,0.65)", "Non-enhancing (GT)"),
        3: ("rgba(255,0,0,0.65)", "Enhancing (GT)")
    }

    for label_val, (fill_color, name) in gt_colors.items():
        mask = (gt_slice == label_val).astype(int)
        if np.any(mask):
            fig.add_trace(go.Contour(
                z=mask,
                showscale=False,
                contours=dict(start=0.5, end=1.5, size=1, coloring="fill"),
                line=dict(width=0),
                name=name,
                opacity=0.65,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, fill_color]]
            ))

    # AI Prediction outline
    if np.any(pred_wt_slice > 0):
        fig.add_trace(go.Contour(
            z=pred_wt_slice,
            showscale=False,
            contours=dict(start=0.5, end=1.5, size=1, coloring="lines"),
            line=dict(color="cyan", width=4, dash="dash"),
            name="AI Prediction (WT)"
        ))

    # Layout
    fig.update_layout(
        title=f"AI Segmentation vs Ground Truth - Slice {slice_idx + 1}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=600,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    return fig


def create_performance_charts(results_df):
    """Create performance visualization charts"""
    # Dice scores by region
    regions = ['WT', 'TC', 'ET']
    colors = ['#3b82f6', '#ef4444', '#10b981']

    fig_dice = go.Figure()

    for i, region in enumerate(regions):
        dice_scores = results_df[f'{region}_dice'].values
        fig_dice.add_trace(go.Box(
            y=dice_scores,
            name=region,
            marker_color=colors[i],
            boxmean=True
        ))

    fig_dice.update_layout(
        title="Dice Scores by Tumor Region",
        yaxis_title="Dice Coefficient",
        showlegend=False,
        height=400
    )

    # Hausdorff distances
    fig_hd = go.Figure()

    for i, region in enumerate(regions):
        hd_scores = results_df[f'{region}_hausdorff'].replace([np.inf, -np.inf], np.nan).dropna().values
        fig_hd.add_trace(go.Box(
            y=hd_scores,
            name=region,
            marker_color=colors[i],
            boxmean=True
        ))

    fig_hd.update_layout(
        title="Hausdorff Distances by Tumor Region",
        yaxis_title="Distance (mm)",
        showlegend=False,
        height=400
    )

    return fig_dice, fig_hd


def create_volume_correlation(results_df):
    """Create volume correlation plot"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df['WT_vol_true'],
        y=results_df['WT_vol_pred'],
        mode='markers',
        marker=dict(
            size=8,
            color='#3b82f6',
            opacity=0.6
        ),
        name='WT Volume'
    ))

    # Perfect correlation line
    max_vol = max(results_df['WT_vol_true'].max(), results_df['WT_vol_pred'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_vol],
        y=[0, max_vol],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Correlation'
    ))

    fig.update_layout(
        title="Volume Correlation - Whole Tumor",
        xaxis_title="True Volume (cm³)",
        yaxis_title="Predicted Volume (cm³)",
        height=400
    )

    return fig


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


def simulate_prediction_progress():
    """Simulate AI prediction progress"""
    progress_text = st.empty()
    progress_bar = st.progress(0)

    stages = [
        "Loading MRI volume...",
        "Preprocessing image data...",
        "Running AI inference...",
        "Post-processing results...",
        "Generating segmentation masks...",
        "Finalizing prediction..."
    ]

    for i, stage in enumerate(stages):
        progress_text.text(stage)
        progress_bar.progress((i + 1) / len(stages))
        time.sleep(0.3)

    progress_text.text("✅ Prediction complete!")
    time.sleep(0.5)
    progress_bar.empty()
    progress_text.empty()


def get_patient_summary(case_data, manifest_entry):
    """Get patient summary for selection"""
    return {
        'case_id': manifest_entry['case_id'],
        'wt_dice': manifest_entry['WT_dice'],
        'wt_volume': manifest_entry['WT_vol_true_cm3'],
        'description': f"WT Dice: {manifest_entry['WT_dice']:.3f} | Volume: {manifest_entry['WT_vol_true_cm3']:.1f} cm³"
    }


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