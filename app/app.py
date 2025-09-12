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
        st.info("👆 Patient selection interface will be implemented here")

    # TAB 2: Performance Metrics
    with tab2:
        st.markdown("## 📈 Model Performance Metrics")
        st.markdown("Comprehensive evaluation results across all test cases")
        st.info("📊 Performance visualizations will be displayed here")

    # TAB 3: Clinical Reports
    with tab3:
        st.markdown("## 🏥 Clinical Report Generator")
        st.markdown("AI-generated clinical reports for automated tumor analysis")
        st.info("📋 Clinical reports interface will be implemented here")

    # TAB 4: Robustness Testing
    with tab4:
        st.markdown("## 🧪 Robustness Testing Results")
        st.markdown("Model stability analysis under various imaging conditions")
        st.info("🔬 Robustness testing results will be displayed here")

    # Footer
    st.markdown("---")
    display_footer()

if __name__ == "__main__":
    main()