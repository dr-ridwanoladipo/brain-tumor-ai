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

if __name__ == "__main__":
    main()