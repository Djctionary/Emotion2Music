import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
import re
import base64
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from rapidfuzz import fuzz, process as rf_process

# Page Configuration
st.set_page_config(
    page_title="Emotion2Music 🎵",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load and encode background SVG
def get_base64_svg(svg_path):
    """Convert SVG to base64 for CSS embedding"""
    try:
        with open(svg_path, "r") as f:
            svg_content = f.read()
        b64 = base64.b64encode(svg_content.encode()).decode()
        return f"data:image/svg+xml;base64,{b64}"
    except:
        return ""

background_svg = get_base64_svg("background.svg")

# Custom CSS
st.markdown(f"""
<style>
    .main {{
        background: linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%);
        background-image: url('{background_svg}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp {{
        background: linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%);
        background-image: url('{background_svg}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(246, 247, 251, 0.65);
        z-index: -1;
        pointer-events: none;
    }}
    
    /* Narrow content layout */
    .main .block-container {{
        max-width: 800px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }}
    
    /* Sidebar adjustments for narrow layout */
    [data-testid="stSidebar"] {{
        width: 280px !important;
    }}
    
    /* Ensure content sections are properly sized */
    .content-section {{
        max-width: 100% !important;
    }}
    
    /* Optimize columns for narrow layout */
    .stColumns {{
        gap: 1rem !important;
    }}
    
    /* Optimize metrics display for narrow layout */
    .metric-card {{
        margin: 8px 4px !important;
    }}
    
    /* Optimize emotion bubbles for narrow layout */
    .bubble {{
        margin: 6px 4px !important;
        padding: 8px 14px !important;
    }}
    
    /* Optimize preset buttons for narrow layout */
    .stButton {{
        margin: 2px !important;
    }}
    
    /* Optimize input area for narrow layout */
    .content-section [data-testid="stTextInput"] > div > div > input {{
        padding: 8px 12px !important;
        font-size: 0.95em !important;
    }}
    
    /* Optimize buttons for narrow layout - removed duplicate */
    .prediction-section {{
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 5px 10px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.12);
        backdrop-filter: blur(16px);
        border: 2px solid rgba(102, 126, 234, 0.15);
        margin: 10px 0;
        color: #0f1115 !important;
    }}
    .prediction-section h3, .prediction-section h2 {{
        color: #667eea !important;
        font-weight: 700;
        font-size: 1.2em;
        padding: 0.6rem 0px 0.6rem
    }}
    .prediction-section h3 strong, .prediction-section h2 strong {{
        color: #764ba2 !important;
        font-weight: 800;
    }}
    .prediction-section p {{
        color: #374151 !important;
    }}
    .prediction-section p strong {{
        color: #667eea !important;
    }}
    .emotion-area {{
        background: rgba(248, 250, 252, 0.6) !important;
        backdrop-filter: blur(14px);
        border: 2px dashed rgba(102, 126, 234, 0.25);
        border-radius: 16px;
        padding: 16px;
        min-height: 120px;
        margin: 12px 0 6px 0;
        transition: all 0.25s ease;
    }}
    .emotion-area:hover {{
        border-color: rgba(102, 126, 234, 0.4);
        background: rgba(248, 250, 252, 0.8) !important;
    }}
    .emotion-area p {{
        color: #6b7280 !important;
    }}
    .bubble {{
        display: inline-block;
        padding: 10px 18px;
        margin: 8px;
        background: rgba(255, 255, 255, 0.5);
        color: #667eea;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95em;
        border: 2px solid rgba(102, 126, 234, 0.25);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        animation: bubbleIn 0.35s cubic-bezier(0.22, 1, 0.36, 1);
        cursor: move;
        position: relative;
        transition: all 0.25s ease;
        user-select: none;
    }}
    .bubble:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        background: rgba(255, 255, 255, 0.7);
        border-color: rgba(102, 126, 234, 0.4);
    }}
    .bubble-close {{
        margin-left: 10px;
        cursor: pointer;
        font-weight: bold;
        opacity: 0.8;
    }}
    .bubble-close:hover {{
        opacity: 1;
        color: #ff6b6b;
    }}
    @keyframes bubbleIn {{
        0% {{
            opacity: 0;
            transform: scale(0.5);
        }}
        100% {{
            opacity: 1;
            transform: scale(1);
        }}
    }}
    .input-area {{
        background: rgba(248, 250, 252, 0.6) !important;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(16, 24, 40, 0.04);
        border: 1px solid rgba(16, 24, 40, 0.06);
        margin: 12px 0 8px 0;
    }}
    .content-section [data-testid="stTextInput"] > div > div > input {{
        background-color: rgba(255, 255, 255, 0.98) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        transition: all 0.3s ease !important;
    }}
    .content-section [data-testid="stTextInput"] > div > div > input:focus {{
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }}
    .stButton {{
        display: inline-block !important;
        margin: 4px !important;
    }}
    h1 {{
        color: #667eea !important;
        text-align: center;
        font-size: 3.0em;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
        font-weight: 800;
    }}
    h2, h3 {{
        color: #667eea !important;
    }}
    .subtitle {{
        color: #764ba2 !important;
        text-align: center;
        font-size: 1.05em;
        margin-bottom: 24px;
        font-weight: 500;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.7) 0%, rgba(118, 75, 162, 0.7) 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        border-radius: 12px !important;
        border: 2px solid rgba(102, 126, 234, 0.6) !important;
        font-size: 1.0em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        cursor: pointer !important;
        display: inline-block !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
        background: linear-gradient(135deg, rgba(118, 75, 162, 0.5) 0%, rgba(102, 126, 234, 0.5) 100%) !important;
        border-color: rgba(102, 126, 234, 0.8) !important;
    }}
    .stButton>button:active {{
        transform: translateY(0);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }}
    .stButton>button[kind="secondary"] {{
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.7) 0%, rgba(245, 87, 108, 0.7) 100%) !important;
        color: #ffffff !important;
        border: 2px solid rgba(245, 87, 108, 0.6) !important;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3) !important;
    }}
    .stButton>button[kind="secondary"]:hover {{
        background: linear-gradient(135deg, rgba(245, 87, 108, 0.5) 0%, rgba(240, 147, 251, 0.5) 100%) !important;
        border-color: rgba(245, 87, 108, 0.8) !important;
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.5) !important;
    }}
    .metric-card {{
        background: rgba(255, 255, 255, 0.85) !important;
        padding: 14px;
        border-radius: 12px;
        margin: 8px 0;
        border: 1px solid rgba(16, 24, 40, 0.06);
        backdrop-filter: blur(12px);
        color: #0f1115 !important;
    }}
    
    /* Metric styling for better hierarchy */
    [data-testid="metric-container"] {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }}
    
    [data-testid="metric-container"] label {{
        font-size: 0.9em !important;
        font-weight: 600 !important;
        color: #667eea !important;
    }}
    
    [data-testid="metric-container"] [data-testid="metric-value"] {{
        font-size: 1.2em !important;
        font-weight: 700 !important;
        color: #0f1115 !important;
    }}
    
    /* Special styling for Match Quality */
    [data-testid="metric-container"]:has(label:contains("Match Quality")) label {{
        font-size: 1.0em !important;
        font-weight: 700 !important;
    }}
    
    [data-testid="metric-container"]:has(label:contains("Match Quality")) [data-testid="metric-value"] {{
        font-size: 1.0em !important;
        font-weight: 600 !important;
    }}
    .section-title {{
        color: #667eea !important;
        font-size: 1.2em;
        font-weight: 700;
        margin: 12px 0 6px 0;
        letter-spacing: -0.01em;
    }}
    .helper-text {{
        color: #6b7280;
        font-size: 0.95em;
        margin: 4px 0 12px 0;
    }}
    .content-section {{
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 13px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.12);
        backdrop-filter: blur(16px);
        border: 2px solid rgba(102, 126, 234, 0.15);
        margin: 20px 0;
        transition: all 0.3s ease;
    }}
    .content-section:hover {{
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.18);
        border-color: rgba(102, 126, 234, 0.25);
    }}
    .content-section .section-title {{
        margin-top: 0;
    }}

    /* Sidebar background - ensure light background */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%) !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%) !important;
    }}
    
    .sidebar-card {{
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.15);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(102, 126, 234, 0.1);
        color: #111827 !important;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }}
    .sidebar-card:hover {{
        border-color: rgba(102, 126, 234, 0.25);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    }}
    .sidebar-title {{
        font-weight: 700;
        color: #667eea !important;
        margin-bottom: 6px;
    }}
    
    /* Ensure list items are visible */
    .sidebar-card ul li, .sidebar-card ol li {{
        color: #374151 !important;
    }}
    .sidebar-card ul li b, .sidebar-card ol li b {{
        color: #111827 !important;
    }}
    
    /* Streamlit warning and error messages styling */
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
    }}
    [data-testid="stNotificationContentWarning"] {{
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%) !important;
        color: #D97706 !important;
        font-weight: 600;
        border-left: 4px solid #F59E0B !important;
    }}
    [data-testid="stNotificationContentError"] {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%) !important;
        color: #DC2626 !important;
        font-weight: 600;
        border-left: 4px solid #EF4444 !important;
    }}
    [data-testid="stNotificationContentSuccess"] {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%) !important;
        color: #059669 !important;
        font-weight: 600;
        border-left: 4px solid #10B981 !important;
    }}
    
    /* Ensure Streamlit text input has proper background */
    [data-testid="stTextInput"] > div > div > input {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #0f1115 !important;
    }}
</style>
""", unsafe_allow_html=True)

# String2VAD Class
class String2VAD:
    def __init__(self, lexicon_url: str):
        self.lexicon_url = lexicon_url
        self._lex = self._read_nrc_vad()
        self._words = self._lex["word"].tolist()

    def _read_nrc_vad(self) -> pd.DataFrame:
        temp_lexicon_path = "NRC-VAD-Lexicon-v2.1.txt"
        if not st.session_state.get('lexicon_downloaded', False):
            with st.spinner("Loading emotion lexicon..."):
                self._download_from_gdrive(self.lexicon_url, temp_lexicon_path)
                st.session_state.lexicon_downloaded = True
        
        df = pd.read_csv(
            temp_lexicon_path,
            sep="\t",
            engine="python",
            header=None,
            names=["word", "valence", "arousal", "dominance"],
        )
        df["word"] = df["word"].astype(str).str.strip().str.lower()
        return df

    def _download_from_gdrive(self, link: str, destination: str):
        fid = self._file_id_from_link(link)
        url = "https://docs.google.com/uc?export=download&id=" + fid
        session = requests.Session()
        response = session.get(url, stream=True)
        
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        if token:
            response = session.get(url, params={"id": fid, "confirm": token}, stream=True)
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    def _file_id_from_link(self, link: str) -> str:
        m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", link)
        if m: return m.group(1)
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", link)
        if m: return m.group(1)
        raise ValueError("Cannot parse Google Drive file id from link.")

    def _tokenize_keywords(self, s: str) -> List[str]:
        parts = re.split(r"[,\|;]+|\s+", s.strip())
        return [p.lower() for p in parts if re.search(r"[a-zA-Z0-9]", p)]

    def _best_fuzzy_match(self, query: str, candidates: List[str]) -> Tuple[Optional[str], float]:
        q = query.strip().lower()
        if not q:
            return None, 0.0
        res = rf_process.extractOne(q, candidates, scorer=fuzz.token_set_ratio)
        return (res[0], float(res[1])) if res else (None, 0.0)

    def convert_to_vad(self, keywords: str, fuzzy_threshold: int = 75) -> Dict:
        tokens = self._tokenize_keywords(keywords)
        rows = []
        for tok in tokens:
            best, score = self._best_fuzzy_match(tok, self._words)
            if best is None or score < float(fuzzy_threshold):
                continue
            row = self._lex.loc[self._lex["word"] == best].iloc[0]
            v_mapped = float(row["valence"]) * 4 + 5
            a_mapped = float(row["arousal"]) * 4 + 5
            d_mapped = float(row["dominance"]) * 4 + 5
            rows.append({
                "input": tok,
                "matched": best,
                "score": float(score),
                "vad": [v_mapped, a_mapped, d_mapped],
            })

        if rows:
            v = sum(r["vad"][0] for r in rows) / len(rows)
            a = sum(r["vad"][1] for r in rows) / len(rows)
            d = sum(r["vad"][2] for r in rows) / len(rows)
            avg = [v, a, d]
        else:
            avg = [float("nan"), float("nan"), float("nan")]

        return {"tokens": rows, "avg_vad": avg, "used_tokens": len(rows)}

# VAD Model
class VADModel(nn.Module):
    def __init__(self, in_dim=2, num_classes=10):
        super().__init__()
        self.in_dim = in_dim
        self.trunk = nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.cls_head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,num_classes))
        self.reg_head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    
    def forward(self, x):
        h = self.trunk(x)
        return self.cls_head(h), self.reg_head(h)

# Load Model and Data
@st.cache_resource
def load_model_and_data():
    # Load model
    model_path = "model/va_2d_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = VADModel(
        in_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    data_path = "data/top3_themes_with_vad_mood_900.tsv"
    df = pd.read_csv(data_path, sep="\t")
    
    return model, checkpoint, df

@st.cache_resource
def load_vad_converter():
    VAD_URL = "https://drive.google.com/file/d/1JOWIqfD3zd9p4oiWgIxajdEUzL1wNEZb/view?usp=sharing"
    return String2VAD(VAD_URL)

def find_matching_song(predicted_mood, predicted_bpm, dataframe):
    """Find the best matching song"""
    mood_df = dataframe[dataframe['TAGS'].str.contains(f'mood/theme---{predicted_mood}', na=False)]
    
    if len(mood_df) == 0:
        return None
    
    mood_df = mood_df.dropna(subset=['BPM'])
    if len(mood_df) == 0:
        return None
    
    mood_df['BPM_diff'] = abs(mood_df['BPM'] - predicted_bpm)
    closest_song = mood_df.loc[mood_df['BPM_diff'].idxmin()]
    
    return {
        'track_id': closest_song['TRACK_ID'],
        'bpm': closest_song['BPM'],
        'bpm_diff': abs(closest_song['BPM'] - predicted_bpm)
    }

def get_audio_url(track_id, client_id="5dd0e1c2", max_retries=3):
    """Get audio URL from Jamendo API with retry mechanism"""
    numeric_id = track_id.replace('track_', '')
    
    api_url = "https://api.jamendo.com/v3.0/tracks/"
    params = {
        "client_id": client_id,
        "format": "json",
        "id": numeric_id,
        "limit": 1,
        "audioformat": "mp32"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                track_data = data['results'][0]
                audio_url = track_data.get('audio') or track_data.get('audiodownload')
                if audio_url:
                    return audio_url
            
            # If no audio URL found, try different audio formats
            if attempt < max_retries - 1:
                # Try different audio formats
                formats = ["mp3", "ogg", "flac"]
                for fmt in formats:
                    params["audioformat"] = fmt
                    try:
                        response = requests.get(api_url, params=params, timeout=10)
                        response.raise_for_status()
                        data = response.json()
                        if data.get('results') and len(data['results']) > 0:
                            track_data = data['results'][0]
                            audio_url = track_data.get('audio') or track_data.get('audiodownload')
                            if audio_url:
                                return audio_url
                    except:
                        continue
                        
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed, return API link for manual access
                api_link = f"{api_url}?client_id={client_id}&format=json&id={numeric_id}&limit=1"
                return {"error": f"Failed to load audio after {max_retries} attempts: {e}", "api_link": api_link}
            continue
    
    # If all attempts failed, return API link
    api_link = f"{api_url}?client_id={client_id}&format=json&id={numeric_id}&limit=1"
    return {"error": f"Failed to load audio after {max_retries} attempts", "api_link": api_link}

# Initialize session state
if 'lexicon_downloaded' not in st.session_state:
    st.session_state.lexicon_downloaded = False
if 'emotion_bubbles' not in st.session_state:
    st.session_state.emotion_bubbles = []
if 'keyword_input' not in st.session_state:
    st.session_state.keyword_input = ""

# Main Interface
st.markdown("<h1>🎵 Emotion2Music</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Transform Your Emotions Into Music 🎼</p>", unsafe_allow_html=True)

# Load Resources
try:
    model, checkpoint, df = load_model_and_data()
    vad_converter = load_vad_converter()
    use_vad = checkpoint['input_dim'] == 3
    moods = checkpoint['moods']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-card'>
        <div class='sidebar-title'>🎨 Model Information</div>
        <ul style='margin:0 0 8px 16px; padding:0; color:#374151;'>
            <li><b>Mode</b>: {'3D (VAD)' if use_vad else '2D (VA)'}</li>
            <li><b>Input Dimensions</b>: {checkpoint['input_dim']}</li>
            <li><b>Emotion Categories</b>: {checkpoint['num_classes']}</li>
            <li><b>Dataset Size</b>: {len(df)} songs</li>
        </ul>
    </div>

    <div class='sidebar-card'>
        <div class='sidebar-title'>💡 How to Use</div>
        <ol style='margin:0 0 0 16px; padding:0; color:#374151;'>
            <li>Type or click presets to add emotion bubbles</li>
            <li>Use Clear All to reset when needed</li>
            <li>Click 🎵 Generate to get music</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Main Content Area
st.markdown("""
<div class='content-section'>
    <p class='section-title'>✍️ Add Emotion Keywords</p>
    <p class='helper-text'>Type an emotion word and press Enter or click the button to add it</p>
""", unsafe_allow_html=True)

def on_add_keyword():
    clean_keyword = str(st.session_state.keyword_input).strip().lower()
    if clean_keyword and clean_keyword not in st.session_state.emotion_bubbles:
        st.session_state.emotion_bubbles.append(clean_keyword)
    st.session_state.keyword_input = ""

col1, col2 = st.columns([4, 1])
with col1:
    st.text_input(
        "Emotion keyword",
        placeholder="e.g., happy, sad, calm, energetic...",
        key="keyword_input",
        label_visibility="collapsed",
        on_change=on_add_keyword
    )
with col2:
    st.button("➕ Add", type="primary", use_container_width=True, on_click=on_add_keyword)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class='content-section'>
    <p class='section-title'>🎭 Emotion Bubble Area</p>
    <p class='helper-text'>Your emotion keywords will appear here as bubbles. Use Clear All to reset.</p>
""", unsafe_allow_html=True)

# Emotion Area with Bubbles
emotion_area_html = '<div class="emotion-area">'
if st.session_state.emotion_bubbles:
    for i, bubble in enumerate(st.session_state.emotion_bubbles):
        emotion_area_html += f'<span class="bubble">{bubble}</span>'
else:
    emotion_area_html += '<p style="color: #6b7280; text-align: center; font-size: 0.95em; margin-top: 30px; line-height: 1.4;">✨ Add emotion keywords below to see them appear here as bubbles ✨</p>'
emotion_area_html += '</div>'
st.markdown(emotion_area_html, unsafe_allow_html=True)

# Bubble management buttons
if st.session_state.emotion_bubbles:
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear All", key="clear_all"):
            st.session_state.emotion_bubbles = []
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

## removed old input area (moved to top) and direct state mutation to avoid StreamlitAPIException

# Preset Keywords (click-only)
st.markdown("""
<div class='content-section'>
    <p class='section-title'>🎯 Quick Preset Emotions</p>
    <p class='helper-text'>Click to quickly add preset emotion keywords</p>
""", unsafe_allow_html=True)

reference_presets = ["Matcha", "Coffee", "Beach", "Night", "Rain"]

# Display all preset buttons in a single row using columns
cols = st.columns(5)
for i, preset in enumerate(reference_presets):
    with cols[i]:
        if st.button(preset, key=f"preset_chip_{i}", use_container_width=True):
            if preset not in st.session_state.emotion_bubbles:
                st.session_state.emotion_bubbles.append(preset)
                st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

## removed stray divider to avoid thin horizontal lines

# Generate Button (compact, centered)
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    generate_button = st.button("🎵 Generate", type="primary", use_container_width=True)

# Handle Prediction
if generate_button:
    if not st.session_state.emotion_bubbles:
        st.warning("⚠️ Please add emotion keywords to the bubble area first!")
    else:
        # Combine all bubbles into a single string
        combined_keywords = ", ".join(st.session_state.emotion_bubbles)
        
        with st.spinner("🎨 Analyzing your emotions..."):
            # Convert to VAD
            res = vad_converter.convert_to_vad(combined_keywords, fuzzy_threshold=75)
            valence, arousal, dominance = res["avg_vad"]
            
            # Check if we have valid VAD values
            if np.isnan(valence) or np.isnan(arousal):
                st.error("⚠️ Unable to recognize keywords. Please try different emotion words!")
                st.stop()
            
            # Prepare input
            if use_vad:
                model_input = torch.tensor([[valence, arousal, dominance]], dtype=torch.float32)
            else:
                model_input = torch.tensor([[valence, arousal]], dtype=torch.float32)
            
            # Predict
            with torch.no_grad():
                logits, bpm_pred = model(model_input)
                _, pred_id = torch.max(logits, 1)
            
            predicted_mood = moods[pred_id.item()]
            predicted_bpm = bpm_pred.item()
            
        # Display prediction results
        st.markdown(f"""
        <div class='prediction-section'>
            <h3>🎭 Emotion Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Use 2 columns for better narrow layout
        if use_vad:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("😊 Valence (Pleasure)", f"{valence:.2f}")
            with col2:
                st.metric("⚡ Arousal (Energy)", f"{arousal:.2f}")
            with col3:
                st.metric("💪 Dominance (Control)", f"{dominance:.2f}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("😊 Valence (Pleasure)", f"{valence:.2f}")
            with col2:
                st.metric("⚡ Arousal (Energy)", f"{arousal:.2f}")
        
        st.markdown(f"""
        <div class='prediction-section'>
            <h3>🎭 Predicted Mood: <strong>{predicted_mood.upper()}</strong> | 🥁 Predicted BPM: <strong>{predicted_bpm:.1f}</strong></h3>
        </div>
        """, unsafe_allow_html=True)

        # Find matching song
        with st.spinner("🔍 Searching for the perfect music match..."):
            song = find_matching_song(predicted_mood, predicted_bpm, df)
            
            if song:
                st.markdown(f"""
                <div class='prediction-section'>
                    <h3>🎵 Your Music Match</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Use 2 columns for better narrow layout
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎵 Track ID", song['track_id'])
                with col2:
                    st.metric("🥁 Actual BPM", f"{song['bpm']:.1f}")
                
                # Get and play audio with retry mechanism
                with st.spinner("🎵 Loading audio (attempting multiple formats)..."):
                    audio_result = get_audio_url(song['track_id'])
                
                st.markdown("""
                <div class='prediction-section'>
                    <h3>🎧 Now Playing</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if isinstance(audio_result, str):
                    # Success: got audio URL
                    st.audio(audio_result)
                    st.success("✅ Music loaded successfully! Enjoy your music 🎶")
                elif isinstance(audio_result, dict) and "error" in audio_result:
                    # Failed after retries, show error and API link
                    st.error(f"❌ {audio_result['error']}")
                    st.markdown("""
                    <div class='prediction-section'>
                        <h3>🔗 Manual Access</h3>
                        <p>You can try accessing the track manually using this API link:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code(audio_result['api_link'], language="text")
                    st.markdown(f"[🔗 Open API Link]({audio_result['api_link']})")
                else:
                    st.warning("⚠️ Unable to load audio, but found a matching track")
            else:
                st.error(f"😢 No music found matching {predicted_mood} mood")

# Footer
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 16px;'>
    <p>🎵 Emotion2Music — Powered by Deep Learning & Music Intelligence</p>
    <p>Built with Streamlit, PyTorch & Jamendo API</p>
</div>
""", unsafe_allow_html=True)

