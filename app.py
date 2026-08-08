import streamlit as st
import speech_recognition as sr
import gtts
import io
import pygame
import time
from langdetect import detect
from PIL import Image
from rag import (
    load_model, load_database, search_products,
    get_all_categories, get_price_range,
    search_by_image
)
import os
import json
import uuid
from datetime import datetime
import numpy as np
import pandas as pd

# =====================================================
# Chat History Management
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(history):
    # Write to a temp file first, then atomically replace the real file.
    # This avoids ending up with a half-written/corrupted JSON file (which
    # load_chat_history would then silently treat as "no history") if the
    # app is interrupted mid-save.
    tmp_file = CHAT_HISTORY_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, CHAT_HISTORY_FILE)

def add_chat_session(query, results, filters=None):
    if not query or not results:
        return

    serializable_results = []
    for product in results:
        p = {}
        for key, value in product.items():
            if isinstance(value, (np.integer, np.int64)):
                p[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                p[key] = float(value)
            elif isinstance(value, (np.bool_)):
                p[key] = bool(value)
            else:
                p[key] = value
        serializable_results.append(p)

    if filters:
        serializable_filters = {}
        for key, value in filters.items():
            if isinstance(value, (np.integer, np.int64)):
                serializable_filters[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                serializable_filters[key] = float(value)
            elif isinstance(value, (np.bool_)):
                serializable_filters[key] = bool(value)
            else:
                serializable_filters[key] = value
    else:
        serializable_filters = {}

    history = load_chat_history()
    session = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results": serializable_results,
        "filters": serializable_filters
    }
    history.insert(0, session)
    if len(history) > 50:
        history = history[:50]
    save_chat_history(history)

def delete_chat_session(session_id):
    history = load_chat_history()
    history = [s for s in history if s["id"] != session_id]
    save_chat_history(history)

def get_intro_message(query, count):
    if not query.strip():
        return f"🛍️ Here are some products we picked for you."
    query_lower = query.lower()
    if "sunglasses" in query_lower:
        intro = "☀️ Here are some sunglasses we found for you!"
    elif "shoes" in query_lower or "footwear" in query_lower:
        intro = "👟 Here are some great shoes matching your style."
    elif "laptop" in query_lower or "computer" in query_lower:
        intro = "💻 Check out these laptops we think you'll love."
    elif "phone" in query_lower or "smartphone" in query_lower:
        intro = "📱 Here are some smartphones we recommend."
    elif "headphones" in query_lower or "earbuds" in query_lower:
        intro = "🎧 Here are some top-rated headphones for you."
    elif "watch" in query_lower:
        intro = "⌚ Discover these stylish watches."
    elif "bag" in query_lower:
        intro = "👜 Here are some bags we think you'll like."
    elif "shirt" in query_lower or "t-shirt" in query_lower:
        intro = "👕 Here are some shirts we've picked for you."
    elif "jeans" in query_lower:
        intro = "👖 Here are some jeans we recommend."
    else:
        intro = f"🔍 Showing results for <strong>{query}</strong>"
    if count > 0:
        intro += f" – found <strong>{count}</strong> product{'s' if count > 1 else ''}."
    return intro

# =====================================================
# About Us Page (now inside app.py)
# =====================================================
def show_about_page():
    # Live stats pulled from the actual loaded catalog (fast: no translation call)
    try:
        num_products = len(products)
        num_categories = len(get_all_categories(products, auto_translate=False))
    except Exception:
        num_products, num_categories = 0, 0

    st.markdown("""
    <style>
    .about-container {
        max-width: 900px;
        margin: 30px auto;
        padding: 40px;
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(16px);
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .avatar-circle {
        width: 92px;
        height: 92px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px auto;
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        background: linear-gradient(135deg, #00d4ff, #7c3aed, #e94560);
        border: 4px solid rgba(255,255,255,0.3);
        box-shadow: 0 0 24px rgba(124,58,237,0.4);
    }
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 14px;
        flex-wrap: wrap;
        margin: 18px 0 6px 0;
    }
    .stat-box {
        border-radius: 14px;
        padding: 12px 22px;
        text-align: center;
        min-width: 130px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .stat-box:nth-child(1) { background: linear-gradient(135deg, #2563eb, #1d4ed8); }
    .stat-box:nth-child(2) { background: linear-gradient(135deg, #7c3aed, #6d28d9); }
    .stat-box:nth-child(3) { background: linear-gradient(135deg, #e94560, #d1264a); }
    .stat-box .num {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
    }
    .stat-box .lbl {
        font-size: 12px;
        color: rgba(255,255,255,0.85);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .about-container h1 {
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        margin: 0 0 4px;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #e94560);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .about-container .subhead {
        text-align: center;
        font-size: 18px;
        color: #a5b4fc;
        margin-bottom: 25px;
        font-weight: 500;
    }
    .about-container .bio-text {
        font-size: 16.5px;
        line-height: 1.8;
        color: #d1d5db;
        text-align: left;
        margin: 15px 0;
    }
    .about-container .bio-text strong {
        color: #c4b5fd;
    }
    .about-container .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #f3f4f6;
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(124,58,237,0.3);
    }
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 15px 0 10px 0;
    }
    .tech-item {
        padding: 14px 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
        transition: 0.3s;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .tech-item:nth-child(8n+1) { background: linear-gradient(135deg, #3b82f6, #2563eb); }
    .tech-item:nth-child(8n+2) { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
    .tech-item:nth-child(8n+3) { background: linear-gradient(135deg, #f43f5e, #e94560); }
    .tech-item:nth-child(8n+4) { background: linear-gradient(135deg, #22d3ee, #0891b2); }
    .tech-item:nth-child(8n+5) { background: #ffffff; color: #1f2937; }
    .tech-item:nth-child(8n+6) { background: linear-gradient(135deg, #34d399, #059669); }
    .tech-item:nth-child(8n+7) { background: linear-gradient(135deg, #a78bfa, #7c3aed); }
    .tech-item:nth-child(8n+8) { background: linear-gradient(135deg, #fb7185, #e11d48); }
    .tech-item:hover {
        transform: translateY(-3px);
        filter: brightness(1.12);
        box-shadow: 0 8px 22px rgba(0,0,0,0.35);
    }
    .tech-item .icon {
        font-size: 28px;
        display: block;
        margin-bottom: 6px;
    }
    .step-list {
        padding-left: 20px;
        color: #d1d5db;
        font-size: 16px;
        line-height: 1.9;
    }
    .step-list li {
        margin-bottom: 6px;
    }
    .step-list li strong {
        color: #a5b4fc;
    }
    .social-links {
        text-align: center;
        margin: 20px 0 10px 0;
    }
    .social-links a {
        display: inline-block;
        margin: 6px 10px;
        color: #93c5fd;
        font-size: 16px;
        text-decoration: none;
        font-weight: 600;
        padding: 8px 18px;
        border-radius: 20px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        transition: 0.3s;
    }
    .social-links a:hover {
        color: #c4b5fd;
        background: rgba(124,58,237,0.2);
        border-color: #7c3aed;
        transform: scale(1.05);
    }
    .feature-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 700;
        margin: 3px 4px;
        color: #ffffff;
        box-shadow: 0 3px 10px rgba(0,0,0,0.25);
    }
    .feature-badge:nth-child(8n+1) { background: linear-gradient(120deg, #3b82f6, #2563eb); }
    .feature-badge:nth-child(8n+2) { background: linear-gradient(120deg, #8b5cf6, #7c3aed); }
    .feature-badge:nth-child(8n+3) { background: linear-gradient(120deg, #f43f5e, #e94560); }
    .feature-badge:nth-child(8n+4) { background: linear-gradient(120deg, #22d3ee, #0891b2); }
    .feature-badge:nth-child(8n+5) { background: #ffffff; color: #1f2937; }
    .feature-badge:nth-child(8n+6) { background: linear-gradient(120deg, #34d399, #059669); }
    .feature-badge:nth-child(8n+7) { background: linear-gradient(120deg, #a78bfa, #7c3aed); }
    .feature-badge:nth-child(8n+8) { background: linear-gradient(120deg, #fb7185, #e11d48); }
    .back-btn-container {
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    about_html = f"""
    <div class="about-container">
        <div class="avatar-circle">DT</div>
        <h1>Dimpal Tamta</h1>
        <div class="subhead">🌟 Creator & Developer • MCA Student • AI Enthusiast</div>

        <div class="stats-row">
            <div class="stat-box"><div class="num">{num_products:,}</div><div class="lbl">Products Indexed</div></div>
            <div class="stat-box"><div class="num">{num_categories}</div><div class="lbl">Categories</div></div>
            <div class="stat-box"><div class="num">3</div><div class="lbl">Search Modes</div></div>
        </div>

        <div class="bio-text">
            <p>Hi there! 👋 I'm Dimpal, a passionate MCA student and AI enthusiast. I built <strong>ShopWise AI</strong> 
            to make online shopping smarter, faster, and more fun. I believe technology should feel intuitive — 
            whether you're typing, speaking, or just snapping a photo.</p>
            <p>This project is my vision of a next‑generation shopping assistant — combining the power of 
            <strong>Sentence Transformers</strong>, <strong>FAISS</strong>, and <strong>CLIP</strong> to deliver 
            accurate product searches via text, voice, and even images.</p>
            <p>When I'm not coding, you'll find me exploring new tech, reading, or sipping coffee ☕.</p>
        </div>

        <div class="section-title">⚙️ How ShopWise AI Works</div>
        <ol class="step-list">
            <li><strong>📝 Your Query</strong> — You type, speak, or upload an image.</li>
            <li><strong>🧠 AI Embeddings</strong> — Sentence Transformers convert your query into a numerical vector (embedding).</li>
            <li><strong>🔍 FAISS Search</strong> — We compare your embedding against 5,000+ product embeddings in milliseconds.</li>
            <li><strong>📦 Results</strong> — The most relevant products are fetched from Amazon's catalog and displayed with prices, ratings, and discounts.</li>
            <li><strong>📸 Image Search</strong> — CLIP model finds visually similar products from your photo.</li>
        </ol>

        <div class="section-title">🛠️ Technologies Used</div>
        <div class="tech-grid">
            <div class="tech-item"><span class="icon">🧠</span>Sentence Transformers</div>
            <div class="tech-item"><span class="icon">⚡</span>FAISS Vector DB</div>
            <div class="tech-item"><span class="icon">🖼️</span>CLIP (Image Search)</div>
            <div class="tech-item"><span class="icon">🎤</span>Speech Recognition</div>
            <div class="tech-item"><span class="icon">🗣️</span>Google TTS</div>
            <div class="tech-item"><span class="icon">🐍</span>Python + Streamlit</div>
            <div class="tech-item"><span class="icon">📊</span>Pandas / NumPy</div>
            <div class="tech-item"><span class="icon">☁️</span>Deep Translator</div>
        </div>

        <div class="section-title">✨ Key Features</div>
        <div style="margin: 10px 0 5px 0;">
            <span class="feature-badge">🔍 Smart Text Search</span>
            <span class="feature-badge">🎤 Voice Search</span>
            <span class="feature-badge">📸 Image Search</span>
            <span class="feature-badge">🌐 Multi‑lingual</span>
            <span class="feature-badge">📜 Search History</span>
            <span class="feature-badge">🗣️ Text‑to‑Speech</span>
            <span class="feature-badge">📊 Analytics</span>
            <span class="feature-badge">🎯 Smart Filters</span>
        </div>

        <div class="section-title">📬 Get in Touch</div>
        <div class="social-links">
            <a href="https://www.linkedin.com/in/dimpal-tamta-5b62402b4/" target="_blank">🔗 LinkedIn</a>
            <a href="#" target="_blank">💼 Portfolio</a>
            <a href="#" target="_blank">📧 Email</a>
            <a href="#" target="_blank">🐙 GitHub</a>
        </div>

        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.08); text-align: center; font-size: 13px; color: #9ca3af;">
            Made with ❤️ in India • Built with Streamlit & AI
        </div>
    </div>
    """
    # Markdown treats lines indented 4+ spaces (after a blank line) as a
    # literal code block. Stripping each line's leading whitespace keeps
    # this as one continuous raw-HTML block so it renders as styled HTML
    # instead of showing up as a scrollable code box.
    about_html = "\n".join(line.strip() for line in about_html.strip().split("\n"))
    st.markdown(about_html, unsafe_allow_html=True)

    if st.button("🏠 Back to Home", key="about_back", use_container_width=False):
        st.session_state.nav_page = "main"
        st.rerun()

# =====================================================
# Initialize Pygame Mixer
# =====================================================
pygame.mixer.init()

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(page_title="ShopWise AI", page_icon="🛒", layout="wide")

# =====================================================
# Cache heavy resources (load BEFORE sidebar)
# =====================================================
@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_database():
    return load_database()

model = get_model()
index, products = get_database()
min_price_all, max_price_all = get_price_range(products)
min_price_all = max(0, int(min_price_all) - 50)
max_price_all = int(max_price_all) + 100

# =====================================================
# Session state (navigation and pagination are separate)
# =====================================================
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "main"   # "main", "about", "analytics"
if "page" not in st.session_state:       # pagination for results
    st.session_state.page = 1
if "query" not in st.session_state:
    st.session_state.query = ""
if "results" not in st.session_state:
    st.session_state.results = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "mic_device_index" not in st.session_state:
    st.session_state.mic_device_index = 0
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""
if "clear_search" not in st.session_state:
    st.session_state.clear_search = False
if "tts_playing_asin" not in st.session_state:
    st.session_state.tts_playing_asin = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# =====================================================
# CSS – with original slider style (two sliders) but red track
# =====================================================
st.markdown("""
<style>

/* ---------- PAGE SPACING ---------- */
div[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2.2rem !important;
}

/* ---------- ANIMATED BACKGROUND ---------- */
.stApp {
    background: linear-gradient(135deg, 
        #0f0c29, #302b63, #24243e, 
        #1a1a2e, #16213e, #0f3460, 
        #533483, #e94560, #f5a623,
        #0f0c29);
    background-size: 600% 600%;
    animation: gradientBG 20s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    25% { background-position: 100% 0%; }
    50% { background-position: 100% 100%; }
    75% { background-position: 0% 100%; }
    100% { background-position: 0% 50%; }
}

/* ---------- HEADER ---------- */
.main-title {
    text-align: left;
    font-size: 50px;
    font-weight: 900;
    margin-top: 0px;
    margin-bottom: 2px;
    letter-spacing: 0.5px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #e94560, #f5a623, #00d4ff);
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleShine 6s linear infinite, titleGlow 2.4s ease-in-out infinite alternate;
}
@keyframes titleShine {
    to { background-position: 300% center; }
}
@keyframes titleGlow {
    from { filter: drop-shadow(0 0 12px rgba(0,212,255,0.55)) drop-shadow(0 0 22px rgba(124,58,237,0.35)); }
    to   { filter: drop-shadow(0 0 26px rgba(233,69,96,0.65)) drop-shadow(0 0 42px rgba(245,166,35,0.5)); }
}
.subtitle {
    text-align: left;
    font-size: 18px;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 4px;
    margin-top: 2px;
    animation: fadeIn 2s;
}
.tagline {
    text-align: left;
    font-size: 16px;
    color: #a5b4fc;
    margin-top: 2px;
    font-weight: 400;
    font-style: italic;
    letter-spacing: 0.5px;
    animation: fadeIn 2.5s;
}
.badge-row {
    margin-top: 12px;
    margin-bottom: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    animation: fadeIn 3s;
}
.badge-chip {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 12.5px;
    font-weight: 700;
    backdrop-filter: blur(6px);
    color: #f8fafc;
    cursor: default;
    transition: 0.3s ease;
    border: 1px solid transparent;
}
.badge-1 { background: rgba(59,130,246,0.18); border-color: rgba(96,165,250,0.5); }
.badge-1:hover { background: linear-gradient(120deg,#3b82f6,#2563eb); color:#fff; transform: translateY(-3px) scale(1.06); box-shadow: 0 8px 18px rgba(59,130,246,0.45); }
.badge-2 { background: rgba(124,58,237,0.18); border-color: rgba(167,139,250,0.5); }
.badge-2:hover { background: linear-gradient(120deg,#8b5cf6,#7c3aed); color:#fff; transform: translateY(-3px) scale(1.06); box-shadow: 0 8px 18px rgba(139,92,246,0.45); }
.badge-3 { background: rgba(233,69,96,0.18); border-color: rgba(251,113,133,0.5); }
.badge-3:hover { background: linear-gradient(120deg,#f43f5e,#e94560); color:#fff; transform: translateY(-3px) scale(1.06); box-shadow: 0 8px 18px rgba(233,69,96,0.45); }
.badge-4 { background: rgba(6,182,212,0.18); border-color: rgba(34,211,238,0.5); }
.badge-4:hover { background: linear-gradient(120deg,#22d3ee,#0891b2); color:#fff; transform: translateY(-3px) scale(1.06); box-shadow: 0 8px 18px rgba(6,182,212,0.45); }
.badge-5 { background: rgba(245,166,35,0.18); border-color: rgba(251,191,36,0.5); }
.badge-5:hover { background: linear-gradient(120deg,#fbbf24,#f5a623); color:#111827; transform: translateY(-3px) scale(1.06); box-shadow: 0 8px 18px rgba(245,166,35,0.45); }

div[data-testid="column"]:has(div[data-testid="stImage"]) {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}
div[data-testid="stImage"] img {
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.4), 0 0 0 3px rgba(255,255,255,0.08);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.bubble {
    position: fixed;
    width: 20px;
    height: 20px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
    animation: float 8s infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes float {
    0% { transform: translateY(100vh) scale(0) rotate(0deg); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { transform: translateY(-10vh) scale(1) rotate(720deg); opacity: 0; }
}

/* ---------- PRODUCT CARDS ---------- */
.product-card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
    transition: 0.4s;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.product-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 12px 48px rgba(0,191,255,0.3);
    border-color: rgba(255,255,255,0.3);
}
.price {
    font-size: 26px;
    font-weight: bold;
    color: #00ff99;
    text-shadow: 0 0 10px rgba(0,255,153,0.3);
}
.old-price {
    color: #cbd5e1;
    text-decoration: line-through;
}
.discount {
    color: #ffd54f;
    font-weight: bold;
}
div[data-testid="stTextInput"] input {
    background: rgba(22,27,34,0.8);
    color: white;
    border-radius: 15px;
    border: 2px solid #3b82f6;
    padding: 12px;
    font-size: 17px;
    backdrop-filter: blur(4px);
}
div[data-testid="stTextInput"] input:focus {
    border: 2px solid #8b5cf6;
    box-shadow: 0 0 20px rgba(139,92,246,0.5);
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed, #e94560);
    color: white !important;
    font-size: 18px;
    font-weight: bold;
    border: none;
    border-radius: 15px;
    padding: 12px;
    transition: 0.3s;
    background-size: 200% 200%;
    animation: buttonGradient 3s ease infinite;
}
@keyframes buttonGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 30px rgba(124,58,237,0.6);
}

/* ---------- SIDEBAR – DARK GLASS ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0e14, #12141f) !important;
    background-image: linear-gradient(180deg, #0b0e14, #12141f) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    padding: 0px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 18px 16px 24px 16px !important;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Brand block at top of sidebar */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    border-radius: 16px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #2563eb, #7c3aed 55%, #e94560);
    box-shadow: 0 6px 18px rgba(124,58,237,0.35);
}
.sidebar-brand .brand-icon {
    font-size: 26px;
    line-height: 1;
}
.sidebar-brand .brand-text h3 {
    margin: 0 !important;
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
}
.sidebar-brand .brand-text p {
    margin: 2px 0 0 0 !important;
    color: #e5e7ff !important;
    font-size: 12px !important;
}

/* ---- Section Headers – unified gradient (blue-purple-pink) ---- */
.section-header {
    display: block;
    width: 100%;
    padding: 10px 14px;
    margin: 20px 0 12px 0;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #ffffff !important;
    background: linear-gradient(90deg, #3b82f6, #7c3aed, #e94560);
    box-shadow: 0 3px 12px rgba(124,58,237,0.3);
}
.section-header.blue,
.section-header.purple,
.section-header.pink,
.section-header.cyan,
.section-header.gold,
.section-header.green {
    background: linear-gradient(90deg, #3b82f6, #7c3aed, #e94560);
}

/* Sidebar controls */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
    font-weight: 500 !important;
}

/* Dropdowns – white cards */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 2px solid transparent !important;
    background-image: linear-gradient(#ffffff, #ffffff),
                       linear-gradient(120deg, #3b82f6, #7c3aed, #e94560) !important;
    background-origin: border-box !important;
    background-clip: padding-box, border-box !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.3) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #374151 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #7c3aed !important;
}
section[data-testid="stSidebar"] div[data-baseweb="popover"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div[role="listbox"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div[role="option"] {
    color: #374151 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div[role="option"][aria-selected="true"] {
    background-color: #ede9fe !important;
    color: #4c1d95 !important;
}

/* ---------- PRICE / RATING SLIDER – SINGLE CLEAN RED TRACK, NO GREY BOX ---------- */
/* Streamlit wraps every slider in its own widget container that carries a
   default grey pill background — that's the thick grey box you were
   seeing behind the thin track. Strip it (and any nested wrapper divs)
   down to fully transparent, and give the whole widget some breathing
   room so it doesn't crowd/overlap the label text underneath it. */
section[data-testid="stSidebar"] div[data-testid="stSlider"] {
    background: transparent !important;
    padding: 2px 0 18px 0 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] > div,
section[data-testid="stSidebar"] div[data-baseweb="slider"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] div[data-baseweb="slider"] {
    height: 6px !important;
    margin: 8px 0 !important;
    padding: 0 !important;
}
/* The full track is one flat, plain color end-to-end — no red "fill"
   from 0 up to the current value, so the unselected side stays white/plain
   and only the thumb (handle) is red. */
section[data-testid="stSidebar"] div[data-baseweb="slider"] > div:first-child {
    background: rgba(255,255,255,0.35) !important;
    height: 6px !important;
    border-radius: 6px !important;
}
/* Neutralize the inner progress/fill div so it blends into the same plain
   track color instead of showing as a red bar from 0 to the thumb. */
section[data-testid="stSidebar"] div[data-baseweb="slider"] > div:first-child > div {
    background: rgba(255,255,255,0.35) !important;
    height: 6px !important;
    border-radius: 6px !important;
}
/* The thumb (handle) – red with glow. The transition is scoped to ONLY
   box-shadow/filter (not "all"/position) — transitioning the thumb's
   position made it visibly lag half a beat behind the mouse while
   dragging, which is what felt "not smooth". */
section[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"] {
    background: #dc2626 !important;
    border: 3px solid #ffffff !important;
    width: 18px !important;
    height: 18px !important;
    box-shadow: 0 0 14px rgba(239,68,68,0.6) !important;
    border-radius: 50% !important;
    transition: box-shadow 0.15s ease, filter 0.15s ease !important;
}
section[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"]:hover,
section[data-testid="stSidebar"] div[data-baseweb="slider"] [role="slider"]:active {
    box-shadow: 0 0 20px rgba(239,68,68,0.85) !important;
    filter: brightness(1.1);
}
/* Hide the min/max endpoint labels ("0" ... "268000") that Streamlit shows
   at the two ends of the track. Only the current value (shown just above
   the thumb) should be visible — not a redundant pair of range numbers.
   We target this a few different ways since the exact internal testid
   can vary by Streamlit version — the structural rule (last child of the
   slider widget that ISN'T the slider control itself) is the reliable
   catch-all if the named testids below don't match. */
section[data-testid="stSidebar"] div[data-testid="stTickBar"],
section[data-testid="stSidebar"] div[data-testid="stTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSliderTickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] > div:last-child:not([data-baseweb="slider"]) {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* Style the current-value label that floats above the thumb so it's easy
   to read (this is the ONLY number that should show on a slider). */
section[data-testid="stSidebar"] div[data-testid="stThumbValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    background: rgba(220,38,38,0.9) !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
}

section[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.10) !important;
    margin: 14px 0 !important;
}
section[data-testid="stSidebar"] .stAlert {
    background-color: rgba(255,255,255,0.06) !important;
    color: #e5e7eb !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
section[data-testid="stSidebar"] .stAlert div {
    color: #e5e7eb !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f9fafb !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #e5e7eb !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: #9ca3af !important;
}

/* Sidebar buttons – clean dark cards */
section[data-testid="stSidebar"] .stButton>button {
    background: rgba(255,255,255,0.05) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 12px !important;
    box-shadow: none !important;
    animation: none !important;
    text-align: left !important;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(124,58,237,0.20) !important;
    border-color: #a78bfa !important;
    transform: none !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.25) !important;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    padding: 8px 6px !important;
    font-size: 15px !important;
    line-height: 1 !important;
    white-space: nowrap !important;
    text-align: center !important;
    color: #f87171 !important;
    background: rgba(248,113,113,0.10) !important;
    border: 1px solid rgba(248,113,113,0.3) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: rgba(248,113,113,0.22) !important;
    border-color: #f87171 !important;
}

/* Chat history cards – fixed size, no date */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-left-width: 5px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
    transition: 0.25s;
    overflow: hidden;
    background: rgba(255,255,255,0.04) !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateX(2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    filter: brightness(1.12);
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] .stButton>button {
    background: transparent !important;
    border: none !important;
    color: #f9fafb !important;
    font-weight: 600 !important;
    padding: 8px 8px 0 8px !important;
    font-size: 14px !important;
    text-align: left !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] .stButton>button:hover {
    background: rgba(255,255,255,0.10) !important;
}
.history-time {
    display: none !important;
}

.tech-pill {
    display: inline-block;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px 4px 3px 0;
    border: 1px solid transparent;
}
.tech-pill.blue   { background: rgba(59,130,246,0.16);  border-color: rgba(59,130,246,0.4);  color: #93c5fd !important; }
.tech-pill.purple { background: rgba(124,58,237,0.16);  border-color: rgba(124,58,237,0.4);  color: #c4b5fd !important; }
.tech-pill.pink   { background: rgba(233,69,96,0.16);   border-color: rgba(233,69,96,0.4);   color: #fca5a5 !important; }
.tech-pill.gold   { background: rgba(245,166,35,0.16);  border-color: rgba(245,166,35,0.4);  color: #fcd34d !important; }

/* Analytics container (inline) */
.analytics-container {
    max-width:900px; 
    margin:40px auto; 
    padding:30px; 
    background:rgba(255,255,255,0.06); 
    backdrop-filter:blur(16px); 
    border-radius:30px; 
    border:1px solid rgba(255,255,255,0.15); 
    box-shadow:0 20px 60px rgba(0,0,0,0.5);
}
.analytics-container h1 {
    font-size:36px; 
    text-align:center; 
    margin-bottom:10px; 
    background:linear-gradient(90deg, #fbbf24, #f59e0b); 
    -webkit-background-clip:text; 
    background-clip:text; 
    -webkit-text-fill-color:transparent;
}
.analytics-container .sub {
    text-align:center; 
    color:#9ca3af; 
    margin-bottom:30px;
}

/* Analytics metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 14px 10px;
}
div[data-testid="stMetricLabel"] { color: #a5b4fc !important; }
div[data-testid="stMetricValue"] { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# Floating Bubbles (visible on all pages)
# =====================================================
bubbles_html = ""
for i in range(12):
    left = (i * 9) % 100
    delay = (i * 0.7) % 6
    size = 15 + (i % 4) * 8
    bubbles_html += f"""
    <div class="bubble" style="left:{left}%; animation-delay:{delay}s; width:{size}px; height:{size}px;"></div>
    """
st.markdown(bubbles_html, unsafe_allow_html=True)

# =====================================================
# Sidebar (common to all pages)
# =====================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🛍️</div>
        <div class="brand-text">
            <h3>ShopWise AI</h3>
            <p>Search smarter. Shop faster.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.nav_page = "main"
            st.rerun()
    with col2:
        if st.button("👤 About", use_container_width=True):
            st.session_state.nav_page = "about"
            st.rerun()

    if st.button("📊 Analytics", use_container_width=True):
        st.session_state.nav_page = "analytics"
        st.rerun()

    st.markdown("---")

    # Only show filters & history on the main page
    if st.session_state.nav_page == "main":
        st.markdown('<div class="section-header">🌐 Language</div>', unsafe_allow_html=True)
        voice_lang = st.selectbox("Voice Language", ["English", "Hindi"], index=0)
        display_lang = st.selectbox("Display Language", ["English", "Hindi"], index=0)
        auto_detect = st.checkbox("Auto-detect query language", value=True)

        st.markdown('<div class="section-header">🔍 Filters</div>', unsafe_allow_html=True)
        translated_cats = get_all_categories(products, auto_translate=True)
        cat_display = ["All"] + translated_cats
        selected_category = st.selectbox("Category", cat_display)

        # ----- ORIGINAL STYLE PRICE SLIDERS (two separate sliders) -----
        st.markdown("**Price Range**")
        min_price = st.slider("Min (₹)", min_price_all, max_price_all, min_price_all, step=50)
        max_price = st.slider("Max (₹)", min_price_all, max_price_all, max_price_all, step=50)

        min_rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 0.0, step=0.5)
        sort_by = st.selectbox("Sort by", [
            "Relevance",
            "Price: Low to High",
            "Price: High to Low",
            "Discount: High to Low",
            "Rating: High to Low"
        ])

        with st.expander("🎤 Microphone Setup"):
            try:
                mic_names = sr.Microphone.list_microphone_names()
                if mic_names:
                    mic_options = [f"{i}: {name}" for i, name in enumerate(mic_names)]
                    current_idx = st.session_state.mic_device_index
                    if current_idx >= len(mic_options):
                        current_idx = 0
                    selected_label = st.selectbox(
                        "Select your microphone",
                        mic_options,
                        index=current_idx,
                        key="mic_selector"
                    )
                    st.session_state.mic_device_index = int(selected_label.split(":")[0])
                    st.caption(f"Selected device: `{selected_label}`")
                else:
                    st.error("No microphones found.")
                    st.stop()
            except Exception as e:
                st.error(f"Could not list microphones: {e}")
                st.stop()
            st.caption("After selecting your mic, use the Speak button next to the search box.")

        with st.expander("⚙️ Tech Stack"):
            st.markdown("""
                <span class="tech-pill blue">MiniLM Embeddings</span>
                <span class="tech-pill purple">FAISS Vector DB</span>
                <span class="tech-pill pink">CLIP Image Search</span>
                <span class="tech-pill gold">5,000 Products</span>
            """, unsafe_allow_html=True)

        # Chat History
        history = load_chat_history()
        st.markdown('<div class="section-header">📜 Recent Searches</div>', unsafe_allow_html=True)

        if not history:
            st.info("No previous searches yet — your searches will show up here.")
        else:
            for session in history[:10]:
                query_short = session['query'][:34] + "…" if len(session['query']) > 34 else session['query']
                with st.container(border=True):
                    item_col1, item_col2 = st.columns([6, 1])
                    with item_col1:
                        if st.button(query_short, key=f"hist_{session['id']}", use_container_width=True):
                            st.session_state.query = session['query']
                            st.session_state.search_input = session['query']
                            st.session_state.results = session['results']
                            st.session_state.page = 1
                            st.session_state.voice_text = ""
                            st.session_state.current_session_id = session['id']
                            st.rerun()
                    with item_col2:
                        if st.button("❌", key=f"del_{session['id']}", help="Remove this search", type="primary"):
                            delete_chat_session(session['id'])
                            if st.session_state.current_session_id == session['id']:
                                st.session_state.results = []
                                st.session_state.current_session_id = None
                            st.rerun()

            if len(history) > 10:
                st.caption(f"+ {len(history)-10} more searches in your history")

            if st.button("Clear all searches", key="clear_all_history", help="Clear all history", type="primary", use_container_width=True):
                if os.path.exists(CHAT_HISTORY_FILE):
                    os.remove(CHAT_HISTORY_FILE)
                st.session_state.results = []
                st.session_state.current_session_id = None
                st.rerun()

        st.caption("Built with Streamlit · Sentence Transformers · FAISS")

# =====================================================
# PAGE ROUTING
# =====================================================

if st.session_state.nav_page == "about":
    show_about_page()

elif st.session_state.nav_page == "analytics":
    # ----- ANALYTICS PAGE -----
    st.markdown("""
    <div class="analytics-container">
        <h1>📊 Search Analytics</h1>
        <div class="sub">Real‑time insights from your search history</div>
    </div>
    """, unsafe_allow_html=True)

    history = load_chat_history()
    if not history:
        st.info("No search data yet. Start searching to see analytics!")
    else:
        data = []
        for s in history:
            data.append({
                "query": s["query"],
                "timestamp": s["timestamp"],
                "category": s.get("filters", {}).get("category", "All") if isinstance(s.get("filters"), dict) else "All",
                "results_count": len(s.get("results", []))
            })
        df = pd.DataFrame(data)

        if not df.empty:
            df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date

            total_searches = len(df)
            unique_queries = df["query"].nunique()
            top_category = df["category"].value_counts().idxmax() if not df["category"].empty else "—"
            avg_results = round(df["results_count"].mean(), 1) if "results_count" in df.columns else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Searches", total_searches)
            m2.metric("Unique Queries", unique_queries)
            m3.metric("Top Category", top_category)
            m4.metric("Avg Results / Search", avg_results)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("🔍 Popular Search Categories")
                cat_counts = df["category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Searches"]
                st.bar_chart(cat_counts.set_index("Category"))
            with col_b:
                st.subheader("📅 Searches Over Time")
                daily_counts = df.groupby("date").size().reset_index(name="Searches")
                if not daily_counts.empty:
                    st.line_chart(daily_counts.set_index("date"))
                else:
                    st.info("Not enough dated data yet.")

            st.subheader("📈 Recent Activity")
            st.dataframe(df[["query", "timestamp", "category"]].head(10), use_container_width=True)
        else:
            st.info("No data to display.")

    if st.button("🏠 Back to Home", key="analytics_back"):
        st.session_state.nav_page = "main"
        st.rerun()

else:
    # ----- MAIN PAGE (default) -----
    # Header
    header_col1, header_col2 = st.columns([1, 5])
    with header_col1:
        try:
            st.image("logo.png", width=118)
        except:
            st.write("🛍️")
    with header_col2:
        st.markdown("<div class='main-title'>ShopWise AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Your Fashion & Shopping Buddy, Powered by AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='tagline'>Find it faster — just type it, say it, or snap a photo of it.</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class="badge-row">
                <span class="badge-chip badge-1">🔍 Smart Search</span>
                <span class="badge-chip badge-2">🎤 Voice Search</span>
                <span class="badge-chip badge-3">📸 Image Search</span>
                <span class="badge-chip badge-4">🌐 Multi‑lingual</span>
                <span class="badge-chip badge-5">📜 Search History</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ---------- Main Page logic (search, results, etc.) ----------
    def on_query_change():
        if "search_input" in st.session_state:
            st.session_state.query = st.session_state.search_input

    def speak_text(text, lang='en', asin=None):
        if asin is not None and st.session_state.tts_playing_asin == asin:
            pygame.mixer.music.stop()
            st.session_state.tts_playing_asin = None
            return
        pygame.mixer.music.stop()
        try:
            tts = gtts.gTTS(text, lang=lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            if asin is not None:
                st.session_state.tts_playing_asin = asin
        except Exception as e:
            st.error(f"TTS error: {e}")

    # Pre‑render logic: handle voice text and clear flag
    if st.session_state.clear_search:
        st.session_state.search_input = ""
        st.session_state.query = ""
        st.session_state.clear_search = False

    if st.session_state.voice_text:
        st.session_state.search_input = st.session_state.voice_text
        st.session_state.query = st.session_state.voice_text
        st.session_state.voice_text = ""

    # Search bar
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input(
            label="Search query",
            label_visibility="collapsed",
            placeholder="🔍 Search products... Example: black cap under ₹500",
            key="search_input",
            on_change=on_query_change
        )
    with col2:
        if st.button("🎤 Speak"):
            with st.spinner("Listening..."):
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 4000
                recognizer.dynamic_energy_threshold = True
                device_index = st.session_state.mic_device_index
                try:
                    with sr.Microphone(device_index=device_index) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=1.0)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                except Exception as e:
                    st.error(f"Microphone error: {e}. Please check your microphone selection.")
                    st.stop()

                lang_code = "hi-IN" if voice_lang == "Hindi" else "en-IN"
                try:
                    query_text = recognizer.recognize_google(audio, language=lang_code)
                except sr.UnknownValueError:
                    fallback = "en-US" if lang_code == "hi-IN" else "hi-IN"
                    try:
                        query_text = recognizer.recognize_google(audio, language=fallback)
                    except:
                        st.error("Sorry, could not understand audio in either language. Please speak clearly.")
                        st.stop()
                except sr.RequestError:
                    st.error("Speech service failed. Check internet connection.")
                    st.stop()

                st.success(f"🎤 You said: **{query_text}**")
                st.session_state.voice_text = query_text
                st.rerun()

    # Search & Clear Buttons
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    col_search, col_clear = st.columns([4, 1])
    with col_search:
        if st.button("🚀 Search Products"):
            query_to_use = st.session_state.query
            if query_to_use.strip() != "":
                with st.spinner("Searching products..."):
                    translate_titles = (display_lang == "English")
                    results = search_products(
                        query_to_use, model, index, products,
                        top_k=20, translate_titles=translate_titles
                    )
                    st.session_state.results = results
                    st.session_state.page = 1
                    if results:
                        filters = {
                            "category": selected_category,
                            "min_price": min_price,
                            "max_price": max_price,
                            "min_rating": min_rating,
                            "sort_by": sort_by
                        }
                        add_chat_session(query_to_use, results, filters)
                        st.session_state.current_session_id = None
                    st.rerun()
            elif selected_category != "All":
                with st.spinner(f"Fetching products in {selected_category}..."):
                    from rag import get_reverse_category_mapping
                    rev_map = get_reverse_category_mapping(products)
                    original_cat = rev_map.get(selected_category)
                    if original_cat is None:
                        for eng, hin in rev_map.items():
                            if selected_category.lower() in eng.lower() or eng.lower() in selected_category.lower():
                                original_cat = hin
                                break
                    if original_cat is None:
                        st.warning(f"Category '{selected_category}' not found.")
                        st.session_state.results = []
                    else:
                        filtered_df = products[products["categoryName"] == original_cat]
                        sample = filtered_df.head(10)
                        from rag import get_category_mapping
                        cat_mapping = get_category_mapping(products)
                        results = []
                        for idx, row in sample.iterrows():
                            try:
                                price = float(row["price"])
                            except:
                                price = 0
                            try:
                                list_price = float(row["listPrice"])
                            except:
                                list_price = 0
                            if list_price > 0:
                                discount = round(((list_price - price) / list_price) * 100)
                            else:
                                discount = 0
                            category_en = cat_mapping.get(row["categoryName"], row["categoryName"])
                            results.append({
                                "asin": row["asin"],
                                "title": row["title"],
                                "category": category_en,
                                "price": price,
                                "listPrice": list_price,
                                "discount": discount,
                                "stars": row["stars"],
                                "reviews": row["reviews"],
                                "bestSeller": row["isBestSeller"],
                                "image": row["imgUrl"],
                                "url": row["productURL"]
                            })
                        st.session_state.results = results
                        st.session_state.page = 1
                        if results:
                            add_chat_session(f"Category: {selected_category}", results)
                            st.session_state.current_session_id = None
                        st.rerun()
            else:
                st.warning("Please enter a product name or select a category.")

    with col_clear:
        if st.button("🗑️ Clear"):
            st.session_state.clear_search = True
            st.session_state.voice_text = ""
            st.session_state.results = []
            st.session_state.page = 1
            st.session_state.uploaded_image = None
            st.session_state.current_session_id = None
            st.rerun()

    # Image Search
    with st.expander("📸 Search by Image", expanded=False):
        st.markdown("Upload a product photo and find visually similar items.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=200)
            st.session_state.uploaded_image = uploaded_file

            if st.button("🔍 Find Similar Products", key="image_search_button"):
                with st.spinner("Searching by image..."):
                    results = search_by_image(image, products, top_k=20, translate_titles=(display_lang=="English"))
                    if results:
                        st.session_state.results = results
                        st.session_state.page = 1
                        add_chat_session("Image Search", results)
                        st.session_state.current_session_id = None
                        st.rerun()
                    else:
                        st.warning("No similar products found. Try a different image.")
        else:
            st.session_state.uploaded_image = None

    # Display Results
    results = st.session_state.results

    if results:
        filtered = results
        if selected_category != "All":
            filtered = [p for p in filtered if p['category'] == selected_category]
        filtered = [p for p in filtered if min_price <= p['price'] <= max_price]
        filtered = [p for p in filtered if float(p['stars']) >= min_rating]

        if sort_by == "Price: Low to High":
            filtered = sorted(filtered, key=lambda x: x['price'])
        elif sort_by == "Price: High to Low":
            filtered = sorted(filtered, key=lambda x: x['price'], reverse=True)
        elif sort_by == "Discount: High to Low":
            filtered = sorted(filtered, key=lambda x: x['discount'], reverse=True)
        elif sort_by == "Rating: High to Low":
            filtered = sorted(filtered, key=lambda x: float(x['stars']), reverse=True)

        if filtered:
            intro = get_intro_message(st.session_state.query, len(filtered))
            st.markdown(f"<div style='font-size:20px; margin-bottom:20px;'>{intro}</div>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.warning("No products match the current filters.")
            st.session_state.results = []
            st.rerun()

        items_per_page = 5
        total_pages = (len(filtered) - 1) // items_per_page + 1 if filtered else 0
        page = st.session_state.page
        start = (page - 1) * items_per_page
        end = start + items_per_page
        page_results = filtered[start:end]

        for product in page_results:
            with st.container():
                st.markdown("<div class='product-card'>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(product["image"], width=180)
                with col2:
                    st.markdown(f"### {product['title']}")
                    st.write(f"📂 **Category:** {product['category']}")
                    st.write(f"⭐ {product['stars']} &nbsp;&nbsp; 📝 {product['reviews']} Reviews")
                    st.markdown(f"<div class='price'>₹{product['price']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='old-price'>₹{product['listPrice']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='discount'>🎉 {product['discount']}% OFF</div>", unsafe_allow_html=True)
                    if str(product["bestSeller"]).lower() == "true":
                        st.success("🏆 Amazon Best Seller")
                    st.link_button("🛒 View on Amazon", product["url"])

                    asin = product["asin"]
                    button_label = "🔊 Speak this"
                    if st.session_state.tts_playing_asin == asin:
                        button_label = "⏹️ Stop speaking"
                    if st.button(button_label, key=f"speak_{asin}"):
                        details = f"{product['title']}. Price ₹{product['price']}. Discount {product['discount']} percent."
                        lang = 'hi' if display_lang == 'Hindi' else 'en'
                        speak_text(details, lang, asin=asin)
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous") and page > 1:
                    st.session_state.page = page - 1
                    st.rerun()
            with col2:
                st.write(f"Page {page} of {total_pages}")
            with col3:
                if st.button("Next ➡️") and page < total_pages:
                    st.session_state.page = page + 1
                    st.rerun()

        if st.button("🔊 Speak All Results Summary"):
            if filtered:
                summary = f"Found {len(filtered)} products. "
                for i, p in enumerate(filtered[:3]):
                    summary += f"Product {i+1}: {p['title']}, price ₹{p['price']}. "
                speak_text(summary, 'hi' if display_lang == 'Hindi' else 'en')
            else:
                speak_text("No products match the current filters.", 'hi' if display_lang == 'Hindi' else 'en')
    else:
        if st.session_state.query or selected_category != "All":
            st.info("No products to display. Try a different search or category.")
        else:
            st.info("🔍 Start by typing a product name, speaking, or uploading an image.")