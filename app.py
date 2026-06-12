import streamlit as st
import requests
import pandas as pd
import numpy as np

API_URL = "https://stress-detection-api-my05.onrender.com/predict"

st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .big-label { font-size: 1.1rem; font-weight: 600; color: #E8EDF5; }
    .metric-card {
        background: #1A1F2E; border-radius: 10px; padding: 16px 20px;
        border: 1px solid #2A3550; text-align: center;
    }
    .metric-val  { font-size: 2.2rem; font-weight: 700; }
    .metric-lbl  { font-size: 0.85rem; color: #8A97B0; margin-top: 4px; }
    .section-hdr {
        background: #1A1F2E; border-left: 4px solid #1E88E5;
        padding: 8px 14px; border-radius: 4px; margin: 18px 0 10px 0;
        font-size: 1.05rem; font-weight: 600; color: #E8EDF5;
    }
    .result-box {
        background: #0D2318; border: 1px solid #00C853;
        border-radius: 10px; padding: 20px; text-align: center;
        margin: 10px 0;
    }
    .result-label { font-size: 1.8rem; font-weight: 800; color: #00C853; }
    .note-box {
        background: #1A2535; border: 1px solid #2A3550;
        border-radius: 8px; padding: 12px 16px;
        font-size: 0.88rem; color: #8A97B0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Stress Detection System")
st.markdown("Upload a raw physiological signal CSV — the app extracts features and predicts your stress state automatically.")

st.markdown('<div class="note-box">📁 <b>Expected CSV columns:</b> <code>ECG</code>, <code>EDA</code>, <code>Resp</code>, <code>Temp</code> &nbsp;|&nbsp; Any number of rows (a window of at least 100 samples recommended) &nbsp;|&nbsp; Matches the <b>WESAD chest sensor</b> (RespiBAN) format available on Kaggle.</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your sensor CSV file",
    type=["csv"],
    help="CSV with columns: ECG, EDA, Resp, Temp"
)

COLUMN_MAP = {
    # Flexible matching — handle casing/naming variants
    "ecg":  ["ecg", "ECG", "Ecg"],
    "eda":  ["eda", "EDA", "Eda", "gsr", "GSR"],
    "resp": ["resp", "Resp", "RESP", "respiration", "Respiration"],
    "temp": ["temp", "Temp", "TEMP", "temperature", "Temperature"],
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    for col in df.columns:
        for c in candidates:
            if col.lower() == c.lower():
                return col
    return None

def extract_features(df):
    """Extract 20 statistical features in the exact order the RF model expects."""
    cols = {}
    for key, candidates in COLUMN_MAP.items():
        found = find_col(df, candidates)
        if found is None:
            return None, f"Could not find '{key.upper()}' column. Found: {list(df.columns)}"
        cols[key] = df[found].dropna().values.astype(float)

    features = []
    names = []
    for sig in ["ecg", "eda", "resp", "temp"]:
        arr = cols[sig]
        features += [arr.mean(), arr.std(), arr.min(), arr.max(), np.median(arr)]
        label = sig.upper()
        names += [f"{label}_mean", f"{label}_std", f"{label}_min", f"{label}_max", f"{label}_median"]

    return features, names, cols

# ── Main logic ────────────────────────────────────────────────────────────────
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success(f"✅ File loaded — **{len(df)} rows × {len(df.columns)} columns**")

    # Preview
    with st.expander("📄 Preview raw data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    # Feature extraction
    result = extract_features(df)
    if result[0] is None:
        st.error(f"⚠️ Column error: {result[1]}")
        st.stop()

    features, feat_names, signal_cols = result

    # ── Signal plots ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📈 Raw Signal Visualisation</div>', unsafe_allow_html=True)

    plot_labels = {"ecg": "ECG (Electrocardiogram)", "eda": "EDA (Electrodermal Activity)",
                   "resp": "Respiration", "temp": "Temperature"}
    plot_colors = {"ecg": "#1E88E5", "eda": "#EF5350", "resp": "#00C853", "temp": "#FFD54F"}

    col1, col2 = st.columns(2)
    for i, (sig, arr) in enumerate(signal_cols.items()):
        # downsample for display — max 1000 points
        display = arr if len(arr) <= 1000 else arr[::max(1, len(arr)//1000)]
        chart_df = pd.DataFrame({plot_labels[sig]: display})
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            st.markdown(f"**{plot_labels[sig]}**")
            st.line_chart(chart_df, color=plot_colors[sig], height=180)

    # ── Extracted features table ──────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🔢 Extracted Features (sent to model)</div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({"Feature": feat_names, "Value": [round(v, 5) for v in features]})
    col_a, col_b, col_c, col_d = st.columns(4)
    for i, col in enumerate([col_a, col_b, col_c, col_d]):
        chunk = feat_df.iloc[i*5:(i+1)*5]
        with col:
            st.dataframe(chunk, hide_index=True, use_container_width=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Predict Stress Level", use_container_width=True, type="primary"):
        with st.spinner("Calling model API..."):
            try:
                resp = requests.post(API_URL, json={"features": features}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    label = data["label"]
                    probs = data["probabilities"]
                    pred_class = data["predicted_class"]

                    # Result box
                    st.markdown(f'<div class="result-box"><div class="result-label">🧠 {label}</div><div style="color:#8A97B0;margin-top:6px">Predicted Stress State</div></div>', unsafe_allow_html=True)

                    # Probability cards
                    st.markdown('<div class="section-hdr">📊 Class Probabilities</div>', unsafe_allow_html=True)
                    colors = {"Baseline": "#1E88E5", "Stress": "#EF5350", "Amusement": "#00C853"}
                    pc1, pc2, pc3 = st.columns(3)
                    for (state, prob), col in zip(probs.items(), [pc1, pc2, pc3]):
                        color = colors.get(state, "#8A97B0")
                        pct = round(prob * 100, 1)
                        with col:
                            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{pct}%</div><div class="metric-lbl">{state}</div></div>', unsafe_allow_html=True)
                            st.progress(prob)

                    # Interpretation
                    st.markdown("---")
                    interp = {
                        "Baseline": "✅ **Baseline (Relaxed)** — Physiological signals are within normal resting range. No stress indicators detected.",
                        "Stress":   "⚠️ **Stress Detected** — Elevated physiological arousal detected. Consider relaxation techniques.",
                        "Amusement":"😊 **Amusement** — Signals indicate a positive, engaged state. Mild arousal consistent with amusement."
                    }
                    st.info(interp.get(label, ""))

                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")

            except requests.exceptions.Timeout:
                st.warning("⏳ API is waking up (Render free tier cold start). Please wait 30 seconds and try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")
else:
    # Landing state — show example
    st.markdown('<div class="section-hdr">📌 How it works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload CSV", "Raw sensor data with ECG, EDA, Resp, Temp columns"),
        ("2️⃣", "Auto Extraction", "App computes 20 statistical features per signal"),
        ("3️⃣", "Signal Plots", "Visualize your raw physiological signals"),
        ("4️⃣", "Prediction", "RF model returns class + probability breakdown"),
    ]
    for (num, title, desc), col in zip(steps, [c1, c2, c3, c4]):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.8rem">{num}</div><div style="font-weight:700;color:#E8EDF5;margin:8px 0 4px">{title}</div><div class="metric-lbl">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="note-box">💡 <b>Sample CSV format:</b><br><code>ECG,EDA,Resp,Temp</code><br><code>-0.12,0.45,0.23,33.8</code><br><code>-0.08,0.47,0.21,33.9</code><br>...<br>Download sample data from <a href="https://www.kaggle.com/datasets/mohamedasem318/wesad-full-dataset" target="_blank">WESAD on Kaggle</a></div>', unsafe_allow_html=True)
