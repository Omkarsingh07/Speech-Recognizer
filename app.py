import os
os.environ["KERAS_BACKEND"] = "torch"
import time
import glob
import streamlit as st
import numpy as np
import librosa
import keras

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Theme Session State & Persistence Setup
# ---------------------------------------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

theme = st.session_state.theme_mode

if theme == "light":
    theme_vars = """
    --bg:           #F9FAFB;
    --surface:      #FFFFFF;
    --border:       rgba(0, 0, 0, 0.08);
    --border-focus: rgba(91, 92, 235, 0.50);
    --primary:      #5B5CEB;
    --primary-hover:#4849D6;
    --text:         #09090B;
    --text-muted:   #4B5563;
    --text-subtle:  #6B7280;
    --card-bg:      #FFFFFF;
    --uploader-bg:  #F3F4F6;
    --badge-bg:     #F3F4F6;
    --track-bg:     #E5E7EB;
    --shadow:       0 4px 20px rgba(0, 0, 0, 0.06);
    """
else:
    theme_vars = """
    --bg:           #09090B;
    --surface:      #111113;
    --border:       rgba(255, 255, 255, 0.08);
    --border-focus: rgba(91, 92, 235, 0.40);
    --primary:      #5B5CEB;
    --primary-hover:#4849D6;
    --text:         #FFFFFF;
    --text-muted:   #A1A1AA;
    --text-subtle:  #71717A;
    --card-bg:      #111113;
    --uploader-bg:  rgba(255, 255, 255, 0.015);
    --badge-bg:     rgba(255, 255, 255, 0.03);
    --track-bg:     rgba(255, 255, 255, 0.05);
    --shadow:       0 8px 24px rgba(0, 0, 0, 0.35);
    """

# ---------------------------------------------------------
# Design System — CSS (Apple San Francisco Typography & Theme Tokens)
# ---------------------------------------------------------
st.markdown(f"""
<style>
/* ── Design Tokens ── */
:root {{
    {theme_vars}
    --radius-lg:    20px;
    --radius-md:    14px;
    --radius-sm:    10px;
    --font:         -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif;
}}

/* ── Global Reset & Typography Upgrade ── */
html, body, [data-testid="stAppViewContainer"], button, input, select, textarea {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    transition: background-color 0.25s ease, color 0.25s ease, border-color 0.25s ease;
}}
section[data-testid="stSidebar"]  {{ display: none !important; }}
header[data-testid="stHeader"]    {{ background: transparent !important; }}
footer                             {{ visibility: hidden !important; }}
#MainMenu                          {{ visibility: hidden !important; }}

/* ── Container ── */
div.block-container {{
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1160px !important;
}}

/* ── Header Layout ── */
.ser-header-container {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}}
.ser-header-left {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
}}
.ser-logo {{
    width: 30px;
    height: 30px;
    background: var(--primary);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    box-shadow: 0 0 0 1px rgba(91,92,235,0.3), 0 4px 12px rgba(91,92,235,0.2);
}}
.ser-brand {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
}}
.ser-tagline {{
    font-size: 0.82rem;
    color: var(--text-subtle);
    font-weight: 500;
    letter-spacing: 0;
}}

/* ── Glass Card ── */
.card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.65rem 1.75rem;
    box-shadow: var(--shadow);
    height: 100%;
    transition: border-color 0.2s ease, background-color 0.25s ease, box-shadow 0.25s ease;
}}
.card:focus-within {{
    border-color: var(--border-focus);
}}

/* ── Card Headers ── */
.card-title {{
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-subtle);
    margin-bottom: 1.15rem;
}}

/* ── Card Description ── */
.card-desc {{
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.55;
    margin-bottom: 1.25rem;
}}

/* ── Helper Text ── */
.helper-text {{
    font-size: 0.78rem;
    color: var(--text-subtle);
    margin-top: 0.6rem;
    text-align: center;
    line-height: 1.4;
}}

/* ── Analyze Button (Streamlit override) ── */
div.stButton > button {{
    height: 46px !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0 1.5rem !important;
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(91, 92, 235, 0.3) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}}
div.stButton > button:hover:not(:disabled) {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(91, 92, 235, 0.45) !important;
}}
div.stButton > button:disabled {{
    opacity: 0.38 !important;
    cursor: not-allowed !important;
    transform: none !important;
}}

/* ── Theme Switcher Toggle Button Styling ── */
.theme-switcher-btn div.stButton > button {{
    height: 34px !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0 0.9rem !important;
    background: var(--badge-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    width: auto !important;
}}
.theme-switcher-btn div.stButton > button:hover {{
    border-color: var(--primary) !important;
    transform: none !important;
}}

/* ── File Uploader ── */
div[data-testid="stFileUploader"] {{
    background: var(--uploader-bg) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease !important;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: var(--primary) !important;
}}

/* ── Audio player chrome ── */
audio {{
    width: 100% !important;
    border-radius: 10px !important;
    height: 40px !important;
}}

/* ── Selectbox override ── */
div[data-baseweb="select"] > div {{
    background: var(--badge-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
}}

/* ── Empty State ── */
.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    text-align: center;
    gap: 0.5rem;
}}
.empty-icon {{
    font-size: 2.25rem;
    opacity: 0.35;
    margin-bottom: 0.35rem;
}}
.empty-title {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-muted);
}}
.empty-hint {{
    font-size: 0.8rem;
    color: var(--text-subtle);
    max-width: 220px;
    line-height: 1.5;
}}

/* ── Prediction Result ── */
.result-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.75rem 0 1.25rem 0;
    text-align: center;
}}
.result-emoji {{
    font-size: 3.75rem;
    line-height: 1;
    margin-bottom: 0.55rem;
}}
.result-name {{
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1.1;
}}
.result-meta {{
    font-size: 0.78rem;
    color: var(--text-subtle);
    margin-top: 0.45rem;
    margin-bottom: 1.35rem;
    line-height: 1.6;
}}
.result-meta strong {{
    color: var(--text-muted);
    font-weight: 600;
}}

/* ── Reduced motion ── */
@media (prefers-reduced-motion: no-preference) {{
    .result-emoji {{
        animation: floatAnim 3.5s ease-in-out infinite alternate;
    }}
}}
@keyframes floatAnim {{
    0%   {{ transform: translateY(0px);  }}
    100% {{ transform: translateY(-5px); }}
}}

/* ── Probability Bars ── */
.prob-section-title {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-subtle);
    margin-bottom: 0.75rem;
}}
.prob-row {{
    margin-bottom: 0.55rem;
    transition: opacity 0.3s ease;
}}
.prob-row.dimmed {{
    opacity: 0.38;
}}
.prob-row.active {{
    opacity: 1;
}}
.prob-label-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.22rem;
}}
.prob-track {{
    width: 100%;
    background: var(--track-bg);
    border-radius: 999px;
    overflow: hidden;
}}
.prob-track.top {{
    height: 10px;
}}
.prob-track.normal {{
    height: 7px;
}}
.prob-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.65s cubic-bezier(0.16, 1, 0.3, 1);
}}

/* ── Emotion Grid ── */
.emotion-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.15rem;
    transition: border-color 0.2s ease, transform 0.2s ease, background-color 0.25s ease;
    height: 100%;
}}
.emotion-card:hover {{
    border-color: var(--border-focus);
    transform: translateY(-2px);
}}
.emotion-header {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.3rem;
}}
.emotion-emoji {{ font-size: 1.35rem; line-height: 1; }}
.emotion-name  {{ font-size: 0.95rem; font-weight: 700; color: var(--text); }}
.emotion-desc  {{ font-size: 0.78rem; color: var(--text-muted); line-height: 1.45; }}

/* ── Expander override ── */
details > summary {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-subtle) !important;
    cursor: pointer !important;
    list-style: none !important;
    padding: 0.5rem 0 !important;
    border-top: 1px solid var(--border) !important;
    margin-top: 1.5rem !important;
}}
details > summary:hover {{ color: var(--text-muted) !important; }}

/* ── Tech Footer ── */
.tech-footer {{
    text-align: center;
    padding: 2rem 0 0.5rem 0;
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
}}
.tech-label {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-subtle);
    margin-bottom: 0.65rem;
}}
.tech-badge {{
    display: inline-block;
    background: var(--badge-bg);
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 0.28rem 0.7rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.2rem;
}}
.copyright {{
    margin-top: 1rem;
    font-size: 0.75rem;
    color: var(--text-subtle);
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ML Engine Setup
# ---------------------------------------------------------
@st.cache_resource
def load_ser_model():
    return keras.models.load_model('best_model.keras')

try:
    model = load_ser_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

emotion_mapping = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
}

emotion_meta = {
    "Angry":   {"emoji": "😠", "color": "#EF4444", "gradient": "linear-gradient(90deg,#EF4444,#F87171)", "desc": "High pitch variance and intense vocal energy."},
    "Disgust":  {"emoji": "🤢", "color": "#10B981", "gradient": "linear-gradient(90deg,#10B981,#34D399)", "desc": "Low fundamental frequency with guttural articulation."},
    "Fear":     {"emoji": "😨", "color": "#8B5CF6", "gradient": "linear-gradient(90deg,#8B5CF6,#A78BFA)", "desc": "Rapid speech rate with trembling micro-pitch shifts."},
    "Happy":    {"emoji": "😊", "color": "#F59E0B", "gradient": "linear-gradient(90deg,#F59E0B,#FBBF24)", "desc": "Elevated mean pitch and wide dynamic amplitude range."},
    "Neutral":  {"emoji": "😐", "color": "#6B7280", "gradient": "linear-gradient(90deg,#6B7280,#9CA3AF)", "desc": "Balanced cadence with steady fundamental frequency."},
    "Sad":      {"emoji": "😢", "color": "#3B82F6", "gradient": "linear-gradient(90deg,#3B82F6,#60A5FA)", "desc": "Downward pitch contour with subdued acoustic energy."},
}

def preprocess_audio(audio_data, orig_sr=16000):
    audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=16000)
    return np.expand_dims(audio_data, axis=0)

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# ---------------------------------------------------------
# Header Bar with Integrated Dark/Light Mode Theme Switcher
# ---------------------------------------------------------
head_c1, head_c2 = st.columns([0.82, 0.18])

with head_c1:
    st.markdown("""
    <div class="ser-header-left" style="padding: 0.75rem 0;">
        <div class="ser-logo">🎙️</div>
        <span class="ser-brand">Speech Emotion Recognition</span>
        <span class="ser-tagline">Deep Learning · Audio Analysis</span>
    </div>
    """, unsafe_allow_html=True)

with head_c2:
    st.markdown('<div class="theme-switcher-btn" style="padding-top: 0.75rem; text-align: right;">', unsafe_allow_html=True)
    toggle_label = "☀️ Light Mode" if st.session_state.theme_mode == "dark" else "🌙 Dark Mode"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.theme_mode = "light" if st.session_state.theme_mode == "dark" else "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="border-bottom: 1px solid var(--border); margin-bottom: 2rem;"></div>', unsafe_allow_html=True)

# Model error banner
if not model_loaded:
    st.error(f"⚠️ Model failed to load: {model_error}")
    st.stop()

# ---------------------------------------------------------
# Main Workspace: 2-Column (60% upload | 40% prediction)
# ---------------------------------------------------------
left_col, right_col = st.columns([1.2, 0.8], gap="large")

SUPPORTED_AUDIO_TYPES = [
    "wav", "mp3", "m4a", "aac", "ogg", "opus", "flac", "wma",
    "amr", "3gp", "webm", "aiff", "aif", "au"
]

# ── LEFT: Audio Input Card ──────────────────────────────
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Audio Input</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="card-desc">Drop an audio file to detect its emotional content. '
        'Supports <strong style="color:var(--text-muted)">WAV · MP3 · M4A · AAC · OGG · OPUS · FLAC · 3GP · WEBM</strong> and more.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=SUPPORTED_AUDIO_TYPES,
        label_visibility="collapsed"
    )

    selected_file = None
    if uploaded_file:
        selected_file = uploaded_file

    sample_files = sorted(glob.glob("*.wav"))
    if sample_files and not uploaded_file:
        st.markdown(
            '<p class="helper-text">or try a sample file</p>',
            unsafe_allow_html=True
        )
        chosen_sample = st.selectbox(
            "Sample files",
            options=["— select a sample —"] + sample_files,
            label_visibility="collapsed"
        )
        if chosen_sample and chosen_sample != "— select a sample —":
            selected_file = chosen_sample

    if selected_file:
        st.markdown("<div style='margin-top:0.85rem;'>", unsafe_allow_html=True)
        st.audio(selected_file)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.15rem;'>", unsafe_allow_html=True)

    analyze_clicked = st.button(
        "Analyze Emotion",
        disabled=(selected_file is None)
    )

    if selected_file is None:
        st.markdown(
            '<p class="helper-text">Select an audio file to begin.</p>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Run inference
if analyze_clicked and selected_file and model_loaded:
    with st.spinner("Running inference…"):
        t0 = time.time()
        try:
            if hasattr(selected_file, "seek"):
                selected_file.seek(0)
            audio_data, sr = librosa.load(selected_file, sr=16000)
            audio_decoded_successfully = True
        except Exception as decode_err:
            audio_decoded_successfully = False
            st.error("⚠️ Unable to decode the uploaded audio file. Please ensure the file is a valid, uncorrupted audio recording.")

        if audio_decoded_successfully:
            processed = preprocess_audio(audio_data, orig_sr=16000)
            probs = model.predict(processed, verbose=0)[0]
            latency = round((time.time() - t0) * 1000, 1)

            top_idx = int(np.argmax(probs))
            top_emotion = emotion_mapping.get(top_idx, f"Class {top_idx}")
            top_conf = round(float(probs[top_idx]) * 100, 1)
            duration = round(len(audio_data) / 16000, 2)

            st.session_state.prediction_data = {
                "top_emotion": top_emotion,
                "top_conf":    top_conf,
                "probs":       probs,
                "latency":     latency,
                "duration":    duration,
            }

# ── RIGHT: Prediction Card ──────────────────────────────
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Prediction</div>', unsafe_allow_html=True)

    if st.session_state.prediction_data is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🎙️</div>
            <div class="empty-title">No prediction yet</div>
            <div class="empty-hint">Upload a file and click Analyze Emotion.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        data   = st.session_state.prediction_data
        e_meta = emotion_meta.get(data["top_emotion"], {
            "emoji": "🎭", "color": "#5B5CEB",
            "gradient": "linear-gradient(90deg,#5B5CEB,#8B5CF6)", "desc": ""
        })

        st.markdown(f"""
        <div class="result-wrapper">
            <div class="result-emoji">{e_meta['emoji']}</div>
            <div class="result-name">{data['top_emotion']}</div>
            <div class="result-meta">
                <strong>{data['top_conf']}%</strong> confidence ·
                <strong>{data['latency']} ms</strong> inference ·
                <strong>{data['duration']}s</strong> audio
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="prob-section-title">Probability Distribution</div>', unsafe_allow_html=True)

        sorted_emotions = sorted(
            emotion_mapping.items(),
            key=lambda x: float(data["probs"][x[0]]) if x[0] < len(data["probs"]) else 0,
            reverse=True
        )

        for idx, em_name in sorted_emotions:
            if idx >= len(data["probs"]):
                continue
            prob     = float(data["probs"][idx]) * 100
            meta     = emotion_meta.get(em_name, {"color": "#5B5CEB", "emoji": "🎭", "gradient": "linear-gradient(90deg,#5B5CEB,#8B5CF6)"})
            is_top   = (em_name == data["top_emotion"])
            dim_cls  = "active" if (is_top or prob >= 5.0) else "dimmed"
            track_cls = "top" if is_top else "normal"

            st.markdown(f"""
            <div class="prob-row {dim_cls}">
                <div class="prob-label-row">
                    <span>{meta['emoji']} {em_name}</span>
                    <span style="color:{meta['color']};font-weight:700;">{prob:.1f}%</span>
                </div>
                <div class="prob-track {track_cls}">
                    <div class="prob-fill" style="width:{prob}%;background:{meta['gradient']};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Emotion Reference — Collapsible
# ---------------------------------------------------------
with st.expander("Supported Emotion Classes", expanded=False):
    g_col1, g_col2, g_col3 = st.columns(3)
    grid_cols = [g_col1, g_col2, g_col3]

    for i, (em_idx, em_name) in enumerate(emotion_mapping.items()):
        meta = emotion_meta[em_name]
        with grid_cols[i % 3]:
            st.markdown(f"""
            <div class="emotion-card">
                <div class="emotion-header">
                    <span class="emotion-emoji">{meta['emoji']}</span>
                    <span class="emotion-name">{em_name}</span>
                </div>
                <div class="emotion-desc">{meta['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
        if i == 2:
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tech Stack Footer
# ---------------------------------------------------------
st.markdown("""
<div class="tech-footer">
    <div class="tech-label">Built With</div>
    <div>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Keras</span>
        <span class="tech-badge">TensorFlow</span>
        <span class="tech-badge">PyTorch Backend</span>
        <span class="tech-badge">Librosa</span>
        <span class="tech-badge">Streamlit</span>
    </div>
    <div class="copyright">Speech Emotion Recognition &copy; 2026</div>
</div>
""", unsafe_allow_html=True)
