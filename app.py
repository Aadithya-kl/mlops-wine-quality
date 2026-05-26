# =============================================================================
# app.py
# Streamlit web application for Wine Quality Prediction.
#
# Run with:
#   streamlit run app.py
# =============================================================================

import sys
import os
import streamlit as st
import numpy as np

# ── Allow imports from src/ when running from project root ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict, FEATURE_COLS

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Background ─────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d0e1e 50%, #1a0a0a 100%);
        color: #f0e6e6;
    }
    /* ── Header ─────────────────────────────────────── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e05c5c, #c0392b, #922b21);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #c9a0a0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* ── Result card ─────────────────────────────────── */
    .result-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-top: 1.5rem;
    }
    .score-big {
        font-size: 4rem;
        font-weight: 900;
        color: #e05c5c;
        line-height: 1.1;
    }
    .category-badge {
        display: inline-block;
        padding: 0.35rem 1.2rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .badge-low    { background:#7f1d1d; color:#fca5a5; }
    .badge-medium { background:#78350f; color:#fcd34d; }
    .badge-high   { background:#14532d; color:#86efac; }
    /* ── Sidebar ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(40, 10, 20, 0.9);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title"> Wine Quality Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Enter the physicochemical properties of a red wine and get an instant quality score.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Feature inputs
# ---------------------------------------------------------------------------
st.sidebar.header(" Wine Physicochemical Properties")
st.sidebar.markdown("Adjust the sliders or type values for each feature.")

# Default values sourced from UCI dataset averages / typical ranges
defaults = {
    "fixed acidity":        7.4,
    "volatile acidity":     0.52,
    "citric acid":          0.26,
    "residual sugar":       2.2,
    "chlorides":            0.079,
    "free sulfur dioxide":  15.0,
    "total sulfur dioxide": 46.0,
    "density":              0.9968,
    "pH":                   3.31,
    "sulphates":            0.66,
    "alcohol":              10.42,
}

# Min / Max / Step for each feature (for the number_input widget)
ranges = {
    "fixed acidity":        (4.0,  16.0,  0.1),
    "volatile acidity":     (0.08,  1.60, 0.01),
    "citric acid":          (0.00,  1.00, 0.01),
    "residual sugar":       (1.0,  16.0,  0.1),
    "chlorides":            (0.01,  0.62, 0.001),
    "free sulfur dioxide":  (1.0,  72.0,  1.0),
    "total sulfur dioxide": (6.0, 289.0,  1.0),
    "density":              (0.990, 1.004, 0.0001),
    "pH":                   (2.70,  4.01,  0.01),
    "sulphates":            (0.33,  2.00,  0.01),
    "alcohol":              (8.0,  15.0,   0.1),
}

feature_values = []
for col in FEATURE_COLS:
    mn, mx, step = ranges[col]
    val = st.sidebar.number_input(
        label=col.title(),
        min_value=float(mn),
        max_value=float(mx),
        value=float(defaults[col]),
        step=float(step),
        format="%.4f" if step < 0.01 else ("%.3f" if step < 0.1 else "%.2f"),
        key=col,
    )
    feature_values.append(val)

# ---------------------------------------------------------------------------
# Prediction button
# ---------------------------------------------------------------------------
st.markdown("###  Prediction")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button(" Predict Quality", use_container_width=True)

if predict_btn:
    try:
        result = predict(feature_values)
        score    = result["score"]
        category = result["category"]

        # Choose badge class
        if "Low" in category:
            badge_cls = "badge-low"
            
            message    = "This wine falls below average quality. It may have notable flaws in acidity or fermentation."
        elif "Medium" in category:
            badge_cls = "badge-medium"
            
            message    = "This is an average to good quality wine. Pleasant for everyday drinking."
        else:
            badge_cls = "badge-high"
            
            message    = "Excellent wine! Outstanding balance of flavours and physicochemical properties."

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:1.1rem; color:#c9a0a0; margin-bottom:0.4rem;">
                    Predicted Quality Score
                </div>
                <div class="score-big">{score}</div>
                <div>
                    <span class="category-badge {badge_cls}"> {category}</span>
                </div>
                <p style="color:#c9a0a0; margin-top:1rem; font-size:0.95rem;">
                    {message}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Expandable feature summary
        with st.expander(" Input Feature Summary"):
            import pandas as pd
            df_display = pd.DataFrame(
                {"Feature": FEATURE_COLS, "Value": feature_values}
            )
            st.dataframe(df_display.set_index("Feature"), use_container_width=True)

    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            " **To train the model**, open a terminal in the project root and run:\n\n"
            "```\npython src/train.py\n```"
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b3a3a; font-size:0.82rem;'>"
    "End-to-End MLOps Pipeline · Wine Quality Prediction · MLSD Course Project"
    "</div>",
    unsafe_allow_html=True,
)
