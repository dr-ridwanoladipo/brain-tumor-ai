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

                # AUTO-PLAY MODE
                if auto_play:
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.markdown('<div class="mri-viewer">', unsafe_allow_html=True)
                        st.markdown("#### 🎬 MRI Cine Loop")

                        if not st.session_state.get('show_prediction', False):
                            # Before prediction - show only plain
                            video_path = Path("evaluation_results") / selected_patient["cine_plain"]
                            st.video(str(video_path))
                            st.info("📹 Plain MRI cine loop - Click 'Run AI Prediction' to see overlay")
                        else:
                            # After prediction - show both side by side
                            video_col1, video_col2 = st.columns(2)

                            with video_col1:
                                st.markdown("**Plain MRI**")
                                plain_video_path = Path("evaluation_results") / selected_patient["cine_plain"]
                                st.video(str(plain_video_path))

                            with video_col2:
                                st.markdown("**AI Segmentation Overlay**")
                                overlay_video_path = Path("evaluation_results") / selected_patient["cine_overlay"]
                                st.video(str(overlay_video_path))

                            st.success("📹 Side-by-side comparison: Plain vs AI Segmentation")

                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown("#### 🎯 AI Analysis")

                        # Patient info
                        st.markdown(f"**Patient ID**: {selected_patient['case_id']}")
                        st.markdown(f"**WT Dice Score**: {selected_patient['WT_dice']:.3f}")
                        st.markdown(f"**True WT Volume**: {selected_patient['WT_vol_true_cm3']:.1f} cm³")
                        st.markdown(f"**Predicted Volume**: {selected_patient['WT_vol_pred_cm3']:.1f} cm³")

                        st.markdown("---")

                        # Prediction button
                        if st.button("🔮 **Run AI Prediction**", key="predict_btn_auto", use_container_width=True):
                            # Simulate prediction
                            simulate_prediction_progress()
                            st.success("✅ AI Prediction Complete!")
                            st.session_state.show_prediction = True
                            st.rerun()

                        # Show prediction results metrics
                        if st.session_state.get('show_prediction', False):
                            st.markdown("#### 📊 Segmentation Results")

                            # Performance metrics for this case
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("WT Dice", f"{selected_patient['WT_dice']:.3f}")
                            with col_b:
                                st.metric("TC Dice", f"{selected_patient['TC_dice']:.3f}")
                            with col_c:
                                st.metric("ET Dice", f"{selected_patient['ET_dice']:.3f}")

                            # Calculate volumes for this patient
                            case_row = results_df[results_df['case_id'] == selected_patient['case_id']].iloc[0]
                            wt_vol = case_row['WT_vol_pred']
                            tc_vol = case_row['TC_vol_pred']
                            et_vol = case_row['ET_vol_pred']

                            # Volume analysis
                            st.markdown("#### 📊 Volume Analysis")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("WT Volume", f"{wt_vol:.1f} cm³")
                            with col2:
                                st.metric("TC Volume", f"{tc_vol:.1f} cm³")
                            with col3:
                                st.metric("ET Volume", f"{et_vol:.1f} cm³")

                # SLIDER MODE
                else:
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.markdown('<div class="mri-viewer">', unsafe_allow_html=True)
                        st.markdown("#### 🔍 MRI Volume Viewer")

                        # MRI controls
                        slice_idx = st.slider(
                            "Select Slice",
                            0,
                            case_data['image'].shape[2] - 1,
                            case_data['image'].shape[2] // 2,
                            key="slice_slider"
                        )

                        modality_idx = st.selectbox(
                            "MRI Modality",
                            [0, 1, 2, 3],
                            format_func=lambda x: ["FLAIR", "T1w", "T1Gd", "T2w"][x],
                            key="modality_select"
                        )

                        # Display MRI slice
                        mri_fig = create_mri_viewer(case_data['image'], slice_idx, modality_idx)
                        st.plotly_chart(mri_fig, use_container_width=True)

                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown("#### 🎯 AI Analysis")

                        # Patient info
                        st.markdown(f"**Patient ID**: {selected_patient['case_id']}")
                        st.markdown(f"**WT Dice Score**: {selected_patient['WT_dice']:.3f}")
                        st.markdown(f"**True WT Volume**: {selected_patient['WT_vol_true_cm3']:.1f} cm³")
                        st.markdown(f"**Predicted Volume**: {selected_patient['WT_vol_pred_cm3']:.1f} cm³")

                        st.markdown("---")

                        # Prediction button
                        if st.button("🔮 **Run AI Prediction**", key="predict_btn_slider", use_container_width=True):
                            # Simulate prediction
                            simulate_prediction_progress()
                            st.success("✅ AI Prediction Complete!")
                            st.session_state.show_prediction = True

                        # Show prediction results
                        if st.session_state.get('show_prediction', False):
                            st.markdown("#### 📊 Segmentation Results")

                            # Create overlay visualization
                            overlay_fig = create_segmentation_overlay(
                                case_data['image'],
                                case_data['label'],
                                case_data['prediction'],
                                slice_idx,
                                modality_idx
                            )

                            st.plotly_chart(overlay_fig, use_container_width=True)

                            st.markdown("""
                                **Legend:**
                                - 🟢 **Green**: Edema (Ground Truth)
                                - 🟡 **Yellow**: Non-enhancing Core (Ground Truth)
                                - 🔴 **Red**: Enhancing Tumor (Ground Truth)
                                - 🔵 **Cyan Dashed Line**: AI Predicted Whole Tumor (WT)
                                """)

                            # Performance metrics for this case
                            st.markdown("#### 📈 Case Performance")

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("WT Dice", f"{selected_patient['WT_dice']:.3f}")
                            with col_b:
                                st.metric("TC Dice", f"{selected_patient['TC_dice']:.3f}")
                            with col_c:
                                st.metric("ET Dice", f"{selected_patient['ET_dice']:.3f}")

                            # Calculate volumes for this patient
                            case_row = results_df[results_df['case_id'] == selected_patient['case_id']].iloc[0]
                            wt_vol = case_row['WT_vol_pred']
                            tc_vol = case_row['TC_vol_pred']
                            et_vol = case_row['ET_vol_pred']

                            # Volume analysis
                            st.markdown("#### 📊 Volume Analysis")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("WT Volume", f"{wt_vol:.1f} cm³")
                            with col2:
                                st.metric("TC Volume", f"{tc_vol:.1f} cm³")
                            with col3:
                                st.metric("ET Volume", f"{et_vol:.1f} cm³")

        else:
            st.info("👆 Please select a patient case above to begin MRI analysis.")


        # TAB 2: Performance Metrics
        with tab2:
            st.markdown("## 📈 Model Performance Metrics")
            st.markdown("Comprehensive evaluation results across all test cases")

            # Summary statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">86.1%</div>
                    <div class="metric-label">Whole Tumor Dice</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">77.8%</div>
                    <div class="metric-label">Tumor Core Dice</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">64.6%</div>
                    <div class="metric-label">Enhancing Tumor Dice</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Performance charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Dice Score Distribution")
                dice_fig, _ = create_performance_charts(results_df)
                st.plotly_chart(dice_fig, use_container_width=True)

            with col2:
                st.markdown("### 📏 Hausdorff Distance Distribution")
                _, hd_fig = create_performance_charts(results_df)
                st.plotly_chart(hd_fig, use_container_width=True)

            # Volume correlation
            st.markdown("### 🎯 Volume Prediction Accuracy")
            vol_fig = create_volume_correlation(results_df)
            st.plotly_chart(vol_fig, use_container_width=True)

            # Detailed results table
            st.markdown("### 📋 Detailed Results by Case")

            # Filter for demo cases only
            demo_case_ids = [case['case_id'] for case in manifest]
            demo_results = results_df[results_df['case_id'].isin(demo_case_ids)].copy()

            # Format for display
            display_cols = ['case_id', 'WT_dice', 'TC_dice', 'ET_dice',
                            'WT_vol_true', 'WT_vol_pred', 'inference_time']

            demo_results_display = demo_results[display_cols].round(3)
            demo_results_display.columns = ['Case ID', 'WT Dice', 'TC Dice', 'ET Dice',
                                            'True Volume (cm³)', 'Pred Volume (cm³)', 'Time (s)']

            st.dataframe(demo_results_display, use_container_width=True)

            # Volume analysis for both modes
            if st.session_state.get('show_prediction', False) and 'selected_patient' in st.session_state:
                selected_patient = st.session_state.selected_patient
                case_row = results_df[results_df['case_id'] == selected_patient['case_id']].iloc[0]
                wt_vol = case_row['WT_vol_pred']
                tc_vol = case_row['TC_vol_pred']
                et_vol = case_row['ET_vol_pred']

                st.markdown("#### 📊 Volume Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("WT Volume", f"{wt_vol:.1f} cm³")
                with col2:
                    st.metric("TC Volume", f"{tc_vol:.1f} cm³")
                with col3:
                    st.metric("ET Volume", f"{et_vol:.1f} cm³")

        # TAB 3: Clinical Reports
        with tab3:
            st.markdown("## 🏥 Clinical Report Generator")
            st.markdown("AI-generated clinical reports for automated tumor analysis")

            if sample_reports:
                # Filter reports for demo cases
                demo_case_ids = [case['case_id'] for case in manifest]
                demo_reports = [r for r in sample_reports if r['patient_id'] in demo_case_ids]

                if demo_reports:
                    st.markdown("### 📋 Select Patient Report")

                    report_case = st.selectbox(
                        "Choose patient for clinical report",
                        demo_reports,
                        format_func=lambda x: f"Patient {x['patient_id']} - {x['clinical_urgency']}"
                    )

                    if report_case:
                        # Display selected report
                        display_clinical_report(report_case)

                        # Report summary metrics
                        st.markdown("### 📊 Report Summary")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "AI Confidence",
                                report_case['ai_confidence']
                            )
                        with col2:
                            st.metric(
                                "Clinical Urgency",
                                report_case['clinical_urgency']
                            )
                        with col3:
                            st.metric(
                                "Total Volume",
                                f"{report_case['tumor_volumes']['whole_tumor_cm3']} cm³"
                            )
                else:
                    st.warning("No clinical reports available for demo cases.")
            else:
                st.warning("Clinical reports not available. Please ensure sample_clinical_reports.json is loaded.")

        # TAB 4: Robustness Testing
        with tab4:
            st.markdown("## 🧪 Robustness Testing Results")
            st.markdown("Model stability analysis under various imaging conditions")

            if robustness_data and robustness_data.get('noise'):
                # Create robustness charts
                noise_fig, intensity_fig = create_robustness_charts(robustness_data)

                if noise_fig and intensity_fig:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🔊 Noise Robustness")
                        st.plotly_chart(noise_fig, use_container_width=True)

                        st.markdown("""
                        **Analysis**: Model maintains stable performance across different noise levels,
                        demonstrating robustness to scanner variations and acquisition artifacts.
                        """)

                    with col2:
                        st.markdown("### 🎛️ Intensity Robustness")
                        st.plotly_chart(intensity_fig, use_container_width=True)

                        st.markdown("""
                        **Analysis**: Performance remains consistent across intensity variations,
                        showing adaptability to different MRI scanner settings and protocols.
                        """)

                    # Summary statistics
                    st.markdown("### 📈 Robustness Summary")

                    try:
                        if 'noise' in robustness_data and '0.05' in robustness_data['noise'] and '0.15' in \
                                robustness_data['noise']:
                            # Get baseline and high noise performance
                            baseline_wt = np.mean([v for v in robustness_data['noise']['0.05']['WT']])
                            noise_15_wt = np.mean([v for v in robustness_data['noise']['0.15']['WT']])

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Baseline WT Dice",
                                    f"{baseline_wt:.3f}"
                                )
                            with col2:
                                st.metric(
                                    "High Noise WT Dice",
                                    f"{noise_15_wt:.3f}",
                                    f"{noise_15_wt - baseline_wt:.3f}"
                                )
                            with col3:
                                if baseline_wt > 0:
                                    robustness_score = (noise_15_wt / baseline_wt) * 100
                                    st.metric(
                                        "Robustness Score",
                                        f"{robustness_score:.1f}%"
                                    )
                    except (KeyError, ValueError, TypeError):
                        st.info("Robustness summary statistics could not be calculated from available data.")
                else:
                    st.warning("Robustness charts could not be generated from available data.")
            else:
                st.warning("Robustness testing data not available. Please ensure robustness_analysis.json is loaded.")

    # Footer
    st.markdown("---")
    display_footer()


if __name__ == "__main__":
    main()