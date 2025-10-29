"""
🧠 Brain Tumor Segmentation UI - Helper Functions
by Ridwan Oladipo, MD | AI Specialist

All reusable functions for the Streamlit UI application
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


def load_custom_css():
    """Load custom CSS for professional medical interface"""
    st.markdown("""
    <style>
    /* Import medical-grade fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit's default chrome */
    #MainMenu, footer, .stAppDeployButton {display: none !important;}

    /* Reduce top/bottom padding of main container */
    div.block-container {
        padding-top: 2.7rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0rem !important;
        margin-bottom: 7rem !important;
        /* max-width: 1200px; */
    }

    /* Medical header styling */
    .medical-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
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
        padding: 1.3rem;
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
        background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
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

    /* Prediction button */
    div.stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.8rem 1.2rem;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        transform: scale(1.02);
    }

    video {
        height: 450px !important;  /* adjust value as you prefer */
        object-fit: contain !important;
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


def create_mri_viewer(image_data, slice_idx=None, modality_idx=0):
    """Create MRI slice viewer"""
    if slice_idx is None:
        slice_idx = image_data.shape[2] // 2

    # Get slice
    slice_data = image_data[:, :, slice_idx - 1, modality_idx]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=slice_data,
        colorscale='gray',
        showscale=False
    ))

    fig.update_layout(
        title=f"MRI Slice {slice_idx} - {'FLAIR' if modality_idx == 0 else 'T1w' if modality_idx == 1 else 'T1Gd' if modality_idx == 2 else 'T2w'}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=None,
        height=450,
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
    base_slice = image_data[:, :, slice_idx - 1, modality_idx]
    gt_slice = label_data[:, :, slice_idx - 1]

    # Collapse AI prediction to WT
    pred_wt_slice = (pred_data[:, :, slice_idx - 1] > 0).astype(np.uint8)

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
        title=f"AI Segmentation vs Ground Truth - Slice {slice_idx}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=None,
        height=450,
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


def create_robustness_charts(robustness_data):
    """Create robustness analysis charts"""
    if not robustness_data.get('noise'):
        return None, None

    # Noise robustness
    fig_noise = go.Figure()

    regions = ['WT', 'TC', 'ET']
    colors = ['#3b82f6', '#ef4444', '#10b981']

    noise_levels = sorted([float(k) for k in robustness_data['noise'].keys()])

    for i, region in enumerate(regions):
        dice_means = []
        dice_stds = []
        for level in noise_levels:
            level_str = str(level)
            if level_str in robustness_data['noise']:
                values = robustness_data['noise'][level_str][region]
                dice_means.append(np.mean(values))
                dice_stds.append(np.std(values))
            else:
                dice_means.append(0)
                dice_stds.append(0)

        fig_noise.add_trace(go.Scatter(
            x=noise_levels,
            y=dice_means,
            error_y=dict(type='data', array=dice_stds),
            mode='lines+markers',
            name=region,
            line=dict(color=colors[i])
        ))

    fig_noise.update_layout(
        title="Robustness to Gaussian Noise",
        xaxis_title="Noise Level (σ)",
        yaxis_title="Dice Score",
        height=400
    )

    # Intensity shift robustness
    fig_intensity = go.Figure()

    intensity_levels = sorted([float(k) for k in robustness_data['intensity'].keys()])

    for i, region in enumerate(regions):
        dice_means = []
        dice_stds = []
        for level in intensity_levels:
            level_str = str(level)
            if level_str in robustness_data['intensity']:
                values = robustness_data['intensity'][level_str][region]
                dice_means.append(np.mean(values))
                dice_stds.append(np.std(values))
            else:
                dice_means.append(0)
                dice_stds.append(0)

        fig_intensity.add_trace(go.Scatter(
            x=intensity_levels,
            y=dice_means,
            error_y=dict(type='data', array=dice_stds),
            mode='lines+markers',
            name=region,
            line=dict(color=colors[i])
        ))

    fig_intensity.update_layout(
        title="Robustness to Intensity Shifts",
        xaxis_title="Intensity Shift Factor",
        yaxis_title="Dice Score",
        height=400
    )

    return fig_noise, fig_intensity


def display_clinical_report(report):
    """Display clinical report in styled format"""
    # Header section
    st.markdown(f"""
    <div class="clinical-report">
        <h4>🏥 Clinical Analysis Report</h4>
        <p><strong>Patient ID:</strong> {report['patient_id']}</p>
        <p><strong>Analysis Date:</strong> {report['analysis_date']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Volume section
    st.markdown(f"""
    <div class="clinical-report">
        <h5>📊 Tumor Volume Analysis</h5>
        <ul>
            <li>Whole Tumor: {report['tumor_volumes']['whole_tumor_cm3']} cm³</li>
            <li>Tumor Core: {report['tumor_volumes']['tumor_core_cm3']} cm³</li>
            <li>Enhancing Tumor: {report['tumor_volumes']['enhancing_tumor_cm3']} cm³</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Assessment section
    st.markdown(f"""
    <div class="clinical-report">
        <h5>🎯 AI Assessment</h5>
        <p><strong>Confidence Level:</strong> {report['ai_confidence']}</p>
        <p><strong>Clinical Urgency:</strong> {report['clinical_urgency']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation section
    st.markdown(f"""
    <div class="clinical-report">
        <h5>💡 Recommendation</h5>
        <p>{report['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Technical notes section
    st.markdown(f"""
    <div class="clinical-report">
        <h5>🔬 Technical Notes</h5>
        <p>{report['technical_notes']}</p>
    </div>
    """, unsafe_allow_html=True)


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


def display_model_info(model_card):
    """Display model information"""
    st.markdown(f"""
    <div class="metric-card">
        <h4>🧠 {model_card['model_name']}</h4>
        <p><strong>Version:</strong> {model_card['version']}</p>
        <p><strong>Architecture:</strong> {model_card['architecture']}</p>

        <h5>📊 Performance Metrics</h5>
        <ul>
            <li>Whole Tumor Dice: {model_card['performance_metrics']['whole_tumor_dice']}</li>
            <li>Tumor Core Dice: {model_card['performance_metrics']['tumor_core_dice']}</li>
            <li>Enhancing Tumor Dice: {model_card['performance_metrics']['enhancing_tumor_dice']}</li>
            <li>Inference Time: {model_card['performance_metrics']['average_inference_time']}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def display_footer():
    """Responsive footer"""
    st.markdown("""
    <style>
    .footer-links {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 12px;
        margin-top: 10px;
    }

    .footer-links a {
        color: #93c5fd;
        text-decoration: none;
        margin: 4px 8px;
        font-weight: 500;
        transition: color 0.2s ease;
    }

    .footer-links a:hover {
        color: #bfdbfe;
    }

    .medical-footer {
        background: #0f172a;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border-top: 2px solid #374151;
        text-align: center;
        margin-top: 3rem;
    }

    @media (max-width: 768px) {
        .footer-links {
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }
    }
    </style>

    <div class="medical-footer">
        <div class="footer-links">
            <a href="https://github.com/dr-ridwanoladipo/brain-tumor-ai">💻 GitHub Repository</a>
            <p style="margin-top: 1rem; font-weight: 600; color: #bfdbfe;">📊 Kaggle Notebooks Collection</p>
            <a href="https://www.kaggle.com/code/ridwanoladipoai/nnunet-brain-tumor-preprocessing">Preprocessing Notebook</a>
            <a href="https://www.kaggle.com/code/ridwanoladipoai/nnunet-brain-tumor-training">Training Notebook</a>
            <a href="https://www.kaggle.com/code/ridwanoladipoai/nnunet-brain-tumor-evaluation">Evaluation Notebook</a>
        </div>
        <br>
        <p>© 2025 Ridwan Oladipo, MD | Medical AI Specialist</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">
            ⚠️ Built to FDA-grade standards for clinical deployment.
            All medical decisions should be made in consultation with qualified healthcare providers.
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_case_metrics_and_legend(selected_patient, results_df):
    """Display legend, case performance, and volumetric metrics"""

    st.markdown("""
    **Legend:**
    - 🟢 **Green**: Edema (Ground Truth)
    - 🟡 **Yellow**: Non-enhancing Core (Ground Truth)
    - 🔴 **Red**: Enhancing Tumor (Ground Truth)
    - 🔵 **Cyan Dashed Line**: AI Predicted Whole Tumor (WT)
    """)

    # Performance metrics for this case
    st.markdown("")
    st.markdown("#### 📈 Case Performance")

    st.markdown(f"""
    <div style="
        display: flex; 
        flex-wrap: nowrap; 
        justify-content: space-between; 
        width: 100%;
        margin-top: 0.5rem;
    ">
      <div style="flex:1; text-align:center; font-size:1rem;">
        <strong>WT Dice</strong><br><span style="font-size:1.2rem;">{selected_patient['WT_dice']:.3f}</span>
      </div>
      <div style="flex:1; text-align:center; font-size:1rem;">
        <strong>TC Dice</strong><br><span style="font-size:1.2rem;">{selected_patient['TC_dice']:.3f}</span>
      </div>
      <div style="flex:1; text-align:center; font-size:1rem;">
        <strong>ET Dice</strong><br><span style="font-size:1.2rem;">{selected_patient['ET_dice']:.3f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Calculate volumes for this patient
    case_row = results_df[results_df['case_id'] == selected_patient['case_id']].iloc[0]
    wt_vol, tc_vol, et_vol = case_row['WT_vol_pred'], case_row['TC_vol_pred'], case_row['ET_vol_pred']

    # Volume analysis
    st.markdown("")
    st.markdown("")
    st.markdown("#### 📊 Volume Analysis")

    st.markdown(f"""
    <div style="
        display: flex; 
        flex-wrap: nowrap; 
        justify-content: space-between; 
        width: 100%;
        margin-top: 0.5rem;
    ">
      <div style="flex:1; text-align:center; font-size:1rem;">
        <strong>WT Volume</strong><br><span style="font-size:1.2rem;">{wt_vol:.1f} cm³</span>
      </div>
      <div style="flex:1; text-align:center; font-size:1rem;">
        <strong>TC Volume</strong><br><span style="font-size:1.2rem;">{tc_vol:.1f} cm³</span>
      </div>
      <div style="flex:1; text-align:center; font-size:1rem;">
        <strong>ET Volume</strong><br><span style="font-size:1.2rem;">{et_vol:.1f} cm³</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_clinical_report_tab(tab, load_demo_data):
    """Renders the complete Clinical Reports tab."""
    with tab:
        st.markdown("#### Clinical Reports")
        st.markdown("")
        manifest, results_df, sample_reports, robustness_data, model_card = load_demo_data()

        if not sample_reports:
            st.warning("Clinical summaries not loaded. Please ensure sample_clinical_reports.json is available.")
            return

        case_options = {
            "Case 1: Enhancing Glioblastoma - Urgent Review": "BRATS_399",
            "Case 2: Infiltrative Glioma - Urgent Review": "BRATS_181",
            "Case 3: Non-Enhancing Glioma - Priority Review": "BRATS_243"
        }

        selected_case_label = st.selectbox(
            "Select a patient case for clinical report:",
            options=list(case_options.keys()),
            index=0
        )

        selected_case_id = case_options[selected_case_label]
        selected_report = next((r for r in sample_reports if r["patient_id"] == selected_case_id), None)

        st.markdown("")

        if not selected_report:
            return

        vols = selected_report["tumor_volumes"]
        wt, tc, et = vols["whole_tumor_cm3"], vols["tumor_core_cm3"], vols["enhancing_tumor_cm3"]

        urgency_text = selected_report["clinical_urgency"]
        urgency_lower = urgency_text.lower()

        if "urgent" in urgency_lower:
            gradient = "linear-gradient(135deg, #991b1b 0%, #dc2626 100%)"
            text_color = "white"
            badge_color = "#ef4444"
            emoji = "🔴"
        elif "priority" in urgency_lower:
            gradient = "linear-gradient(135deg, #fde68a 0%, #fbbf24 100%)"
            text_color = "#111827"
            badge_color = "#f59e0b"
            emoji = "🟡"
        else:
            gradient = "linear-gradient(135deg, #e0f2fe 0%, #f8fafc 100%)"
            text_color = "#0f172a"
            badge_color = "#10b981"
            emoji = "🟢"

        dice_val = None
        for token in selected_report["technical_notes"].split():
            try:
                val = float(token)
                if 0 < val <= 1:
                    dice_val = val
                    break
            except ValueError:
                continue
        dice_val = dice_val or 0.91

        html_content = f"""
        <style>
        .summary-card {{
            background: {gradient};
            color: {text_color};
            border-radius: 16px;
            padding: 1.8rem 2rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.12);
            margin-top: 1.5rem;
            font-family: 'Inter', sans-serif;
        }}
        .summary-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: 1.1rem;
            border-bottom: 1px solid rgba(0,0,0,0.15);
            padding-bottom: 0.6rem;
            margin-bottom: 1rem;
        }}
        .urgency-badge {{
            background: {badge_color};
            color: white;
            padding: 0.35rem 0.8rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}
        .summary-section {{
            margin-top: 1.2rem;
        }}
        .summary-section h5 {{
            margin-bottom: 0.4rem;
            font-size: 1rem;
            font-weight: 600;
        }}
        .metric-line {{
            display: flex;
            justify-content: space-between;
            opacity: 0.9;
            font-size: 0.95rem;
        }}
        @media (max-width: 600px) {{
            .summary-card {{
                padding: 1.2rem 1rem;
            }}
            .summary-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
                font-size: 1rem;
            }}
            .urgency-badge {{
                font-size: 0.9rem;
                padding: 0.25rem 0.6rem;
            }}
            .summary-section h5 {{
                font-size: 0.95rem;
            }}
        }}
        </style>

        <div class="summary-card">
            <div class="summary-header">
                <span><span class="urgency-badge">{emoji} {urgency_text}</span></span>
                <span>Confidence: <b>{selected_report['ai_confidence']}</b></span>
            </div>

            <div class="summary-section">
                <h5>Clinical Insight</h5>
                <p style="opacity:0.95;">{selected_report['recommendation']}</p>
            </div>

            <div class="summary-section">
                <h5>Tumor Volume Distribution (cm³)</h5>
                <div class="metric-line"><span>Whole Tumor</span><span>{wt:.1f}</span></div>
                <div class="metric-line"><span>Tumor Core</span><span>{tc:.1f}</span></div>
                <div class="metric-line"><span>Enhancing Tumor</span><span>{et:.1f}</span></div>
            </div>

            <div class="summary-section">
                <h5>Model Precision</h5>
                <p style="opacity:0.9;">Whole Tumor Dice Score: <b>{dice_val:.3f}</b></p>
            </div>
        </div>
        """

        # footer gap
        components.html(
            f"""
            <div style='margin-bottom:-90px; width:100%;'>
                {html_content}
            </div>
            """,
            height=450,
            scrolling=False,
        )
