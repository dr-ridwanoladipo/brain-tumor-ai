"""
🧠 Brain Tumor Segmentation AI - Streamlit Application
Medical AI interface for brain tumor analysis.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from brain_ui_helpers import *

# ================ 🛠 SIDEBAR TOGGLE ================
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

if st.button("🧠", help="Toggle sidebar"):
    st.session_state.sidebar_state = (
        'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
    )
    st.rerun()

st.markdown(
    '<div style="font-size:0.75rem; color:#6b7280; margin-top:-10px;">Menu</div>',
    unsafe_allow_html=True
)

def main():
    # Load custom CSS
    load_custom_css()

    # Header
    st.markdown("""
    <div class="medical-header">
        <h1>🧠 Brain Tumor Segmentation AI</h1>
        <p>AI-powered brain tumor analysis with clinical precision</p>
        <p><strong>By Ridwan Oladipo, MD | AI Specialist</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading AI model and demo data..."):
        manifest, results_df, sample_reports, robustness_data, model_card = load_demo_data()

    if manifest is None:
        st.error("Failed to load demo data. Please ensure all data files are present.")
        return

    # Status indicator
    st.markdown("""
    <div class="success-indicator">
        ✅ AI Model Ready | 86.1% WT Dice Performance | Clinical Precision
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with model info
    with st.sidebar:
        st.markdown("### 🎯 Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">86%</div>
                <div class="metric-label">WT Dice Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">78%</div>
                <div class="metric-label">TC Dice Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ℹ️ About This AI")
        st.markdown("""
        - **Architecture**: nnU-Net 2025 (5-level U-Net)
        - **Training**: 484 brain MRI volumes  
        - **Modalities**: FLAIR, T1w, T1Gd, T2w
        - **Validation**: 49 test cases
        - **Performance**: Clinical-grade accuracy
        """)

        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.markdown(f"**Demo Cases**: {len(manifest)} selected patients")
        st.markdown(f"**Total Evaluated**: {len(results_df)} volumes")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Tumor Segmentation",
        "📈 Performance Metrics",
        "🏥 Clinical Reports",
        "🧪 Robustness Testing"
    ])

    # TAB 1: Tumor Segmentation
    with tab1:
        st.markdown("## 🧠 AI-Powered Tumor Segmentation")
        st.markdown("Select a patient case below to analyze their brain MRI with our AI model.")

        # Patient selection
        st.markdown("### 👥 Patient Selection")

        cols = st.columns(3)
        selected_patient = None

        for i, case in enumerate(manifest):
            with cols[i]:
                case_summary = get_patient_summary(None, case)

                if st.button(
                        f"**Patient {case['case_id']}**\n\n{case_summary['description']}",
                        key=f"patient_{i}",
                        use_container_width=True
                ):
                    selected_patient = case
                    st.session_state.selected_patient = case
                    # Reset prediction state when switching patients
                    st.session_state.show_prediction = False

        # Use session state to maintain selection
        if 'selected_patient' in st.session_state:
            selected_patient = st.session_state.selected_patient

        if selected_patient:
            st.markdown(f"### 📋 Patient {selected_patient['case_id']} - MRI Analysis")

            # Load patient data
            case_data = load_npz_case(selected_patient['npz_file'])

            if case_data:
                # Auto-play toggle
                auto_play = st.checkbox("▶️ Auto-play slices", key="auto_play")

                # Reset prediction state when toggling auto-play mode
                if 'previous_auto_play' not in st.session_state:
                    st.session_state.previous_auto_play = auto_play
                elif st.session_state.previous_auto_play != auto_play:
                    st.session_state.show_prediction = False
                    st.session_state.previous_auto_play = auto_play

                # MRI viewer structure placeholder
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown('<div class="mri-viewer">', unsafe_allow_html=True)
                    st.markdown("#### 🔍 MRI Volume Viewer")
                    st.info("MRI visualization will be implemented based on mode selection")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 🎯 AI Analysis")

                    # Patient info
                    st.markdown(f"**Patient ID**: {selected_patient['case_id']}")
                    st.markdown(f"**WT Dice Score**: {selected_patient['WT_dice']:.3f}")
                    st.markdown(f"**True WT Volume**: {selected_patient['WT_vol_true_cm3']:.1f} cm³")
                    st.markdown(f"**Predicted Volume**: {selected_patient['WT_vol_pred_cm3']:.1f} cm³")

                    st.markdown("---")

                    # Prediction button placeholder
                    if st.button("🔮 **Run AI Prediction**", key="predict_btn", use_container_width=True):
                        st.success("✅ AI Prediction functionality will be implemented")

        else:
            st.info("👆 Please select a patient case above to begin MRI analysis.")

    # TAB 2: Performance Metrics (unchanged)
    with tab2:
        st.markdown("## 📈 Model Performance Metrics")
        st.markdown("Comprehensive evaluation results across all test cases")
        st.info("📊 Performance visualizations will be displayed here")

    # TAB 3: Clinical Reports (unchanged)
    with tab3:
        st.markdown("## 🏥 Clinical Report Generator")
        st.markdown("AI-generated clinical reports for automated tumor analysis")
        st.info("📋 Clinical reports interface will be implemented here")

    # TAB 4: Robustness Testing (unchanged)
    with tab4:
        st.markdown("## 🧪 Robustness Testing Results")
        st.markdown("Model stability analysis under various imaging conditions")
        st.info("🔬 Robustness testing results will be displayed here")

    # Footer
    st.markdown("---")
    display_footer()


if __name__ == "__main__":
    main()