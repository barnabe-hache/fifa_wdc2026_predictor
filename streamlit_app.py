"""
streamlit_app.py — CDM 2026 Score Predictor
"""

from pathlib import Path
import numpy as np
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR    = PROJECT_ROOT / "models"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CDM 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Countries ─────────────────────────────────────────────────────────────────
COUNTRIES_FLAGS = {
    "Mexico": "🇲🇽", "South_Africa": "🇿🇦", "South_Korea": "🇰🇷",
    "Czechia": "🇨🇿", "Canada": "🇨🇦", "Bosnia_and_Herzegovina": "🇧🇦",
    "Qatar": "🇶🇦", "Switzerland": "🇨🇭", "Brazil": "🇧🇷", "Morocco": "🇲🇦",
    "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "United_States": "🇺🇸", "Paraguay": "🇵🇾",
    "Australia": "🇦🇺", "Turkey": "🇹🇷", "Germany": "🇩🇪", "Curaçao": "🇨🇼",
    "Ecuador": "🇪🇨", "Ivory_Coast": "🇨🇮", "Netherlands": "🇳🇱", "Japan": "🇯🇵",
    "Sweden": "🇸🇪", "Tunisia": "🇹🇳", "Belgium": "🇧🇪", "Egypt": "🇪🇬",
    "Iran": "🇮🇷", "New_Zealand": "🇳🇿", "Spain": "🇪🇸", "Cape_Verde": "🇨🇻",
    "Saudi_Arabia": "🇸🇦", "Uruguay": "🇺🇾", "France": "🇫🇷", "Senegal": "🇸🇳",
    "Iraq": "🇮🇶", "Norway": "🇳🇴", "Argentina": "🇦🇷", "Algeria": "🇩🇿",
    "Austria": "🇦🇹", "Jordan": "🇯🇴", "Portugal": "🇵🇹", "DR_Congo": "🇨🇩",
    "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷",
    "Ghana": "🇬🇭", "Panama": "🇵🇦",
}

COUNTRY_KEYS   = sorted(COUNTRIES_FLAGS.keys())
DISPLAY_NAMES  = {k: k.replace("_", " ") for k in COUNTRY_KEYS}
DISPLAY_LIST   = [DISPLAY_NAMES[k] for k in COUNTRY_KEYS]
DISPLAY_TO_KEY = {v: k for k, v in DISPLAY_NAMES.items()}

def flag(key: str) -> str:
    return COUNTRIES_FLAGS.get(key, "🏳")

# ── Session state navigation ──────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "predictor"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

/* ─ Base ─ */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F7F9F2 !important;
}
[data-testid="stAppViewContainer"] > .main { background-color: #F7F9F2; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 0 !important; max-width: 1060px !important; }
* { font-family: 'Inter', sans-serif; }

/* ─ Top nav bar ─ */
.topnav {
    background: #FFFFFF;
    border-bottom: 2px solid #E8F5E9;
    padding: .7rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.topnav-brand {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: .08em;
    color: #15803D;
}
.topnav-brand span { color: #F59E0B; }
.nav-btn {
    display: inline-block;
    padding: .4rem 1.1rem;
    border-radius: 999px;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .04em;
    cursor: pointer;
    border: 2px solid transparent;
    transition: all .18s;
    text-decoration: none;
}
.nav-btn.active {
    background: #15803D;
    color: #FFFFFF;
    border-color: #15803D;
}
.nav-btn.inactive {
    background: transparent;
    color: #15803D;
    border-color: #BBF7D0;
}
.nav-btn.inactive:hover { background: #F0FDF4; }

/* ─ Hero ─ */
.hero {
    text-align: center;
    padding: 1.4rem 1rem 1rem;
}
.hero-eyebrow {
    font-size: .7rem; letter-spacing: .16em; color: #16A34A;
    text-transform: uppercase; margin-bottom: .35rem; font-weight: 700;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    letter-spacing: .07em; color: #14532D; line-height: 1; margin: 0;
}
.hero-title span { color: #F59E0B; }

/* ─ Section label ─ */
.section-label {
    font-size: .65rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #6B7280; margin-bottom: .35rem;
}

/* ─ Cards ─ */
.card {
    background: #FFFFFF;
    border: 1.5px solid #E5E7EB;
    border-radius: 14px;
    padding: 1.15rem 1.35rem;
    margin-bottom: .85rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.card-title {
    font-size: .62rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #9CA3AF; margin-bottom: .75rem;
}

/* ─ Matchup hero ─ */
.matchup {
    background: linear-gradient(145deg, #14532D 0%, #15803D 50%, #166534 100%);
    border-radius: 18px; padding: 2rem 1.5rem;
    text-align: center; margin-bottom: 1.3rem;
    box-shadow: 0 8px 32px rgba(21,128,61,.25);
    position: relative; overflow: hidden;
}
.matchup::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 110%, rgba(245,158,11,.18) 0%, transparent 60%);
    pointer-events: none;
}
.matchup-flag  { font-size: 4rem; line-height: 1; filter: drop-shadow(0 2px 8px rgba(0,0,0,.3)); }
.matchup-name  { font-family: 'Bebas Neue',sans-serif; font-size: 1.4rem; color: #DCFCE7; letter-spacing: .06em; margin-top: .2rem; }
.matchup-vs    { font-family: 'Bebas Neue',sans-serif; font-size: 1.9rem; color: #F59E0B; letter-spacing: .15em; text-shadow: 0 0 20px rgba(245,158,11,.5); }
.score-display { font-family: 'Bebas Neue',sans-serif; font-size: 5rem; color: #FFFFFF; letter-spacing: .05em; line-height: 1.05; margin-top: .25rem; text-shadow: 0 2px 12px rgba(0,0,0,.3); }
.score-dash    { color: #F59E0B; margin: 0 .2em; }
.result-badge  { display: inline-block; font-weight: 800; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; padding: .3rem .9rem; border-radius: 999px; margin-top: .4rem; }
.result-badge.home { background: #F59E0B; color: #14532D; }
.result-badge.draw { background: #E5E7EB; color: #374151; }
.result-badge.away { background: #EF4444; color: #FFFFFF; }

/* ─ Confidence bar ─ */
.conf-wrap  { max-width: 300px; margin: .85rem auto 0; }
.conf-label { font-size: .62rem; color: rgba(255,255,255,.6); text-transform: uppercase; letter-spacing: .1em; }
.conf-bar-bg   { background: rgba(255,255,255,.2); border-radius: 999px; height: 5px; margin: .28rem 0; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#FDE68A,#F59E0B); }
.conf-val { font-family: 'JetBrains Mono',monospace; font-size: .7rem; color: rgba(255,255,255,.55); text-align: right; }

/* ─ Metric pill ─ */
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(150px,1fr)); gap: .6rem; }
.metric-pill {
    background: #F9FAFB; border: 1.5px solid #E5E7EB; border-radius: 10px;
    padding: .72rem .9rem; position: relative; cursor: default;
    transition: border-color .18s, box-shadow .18s;
}
.metric-pill:hover { border-color: #6EE7B7; box-shadow: 0 2px 10px rgba(21,128,61,.1); }
.metric-pill:hover .tooltip { display: block; }
.metric-label { font-size: .6rem; color: #6B7280; text-transform: uppercase; letter-spacing: .1em; line-height: 1.3; }
.metric-value { font-family: 'JetBrains Mono',monospace; font-size: 1.15rem; font-weight: 600; color: #111827; margin-top: .18rem; }
.metric-value.green  { color: #15803D; }
.metric-value.amber  { color: #B45309; }
.metric-value.purple { color: #6D28D9; }
.metric-sub { font-size: .6rem; color: #9CA3AF; margin-top: .08rem; }
.tooltip {
    display: none; position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%);
    background: #111827; border-radius: 9px; padding: .6rem .85rem;
    width: 230px; font-size: .7rem; color: #E5E7EB; line-height: 1.5;
    z-index: 999; box-shadow: 0 8px 28px rgba(0,0,0,.25); white-space: normal;
}
.tooltip::after { content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%); border:6px solid transparent; border-top-color:#111827; }

/* ─ Proba bars ─ */
.proba-row { display:flex; align-items:center; gap:.65rem; margin:.38rem 0; }
.proba-name { font-size:.78rem; color:#374151; width:130px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:500; }
.proba-track { flex:1; background:#F3F4F6; border-radius:999px; height:9px; overflow:hidden; }
.proba-fill  { height:100%; border-radius:999px; }
.proba-fill.home { background:linear-gradient(90deg,#16A34A,#4ADE80); }
.proba-fill.draw { background:linear-gradient(90deg,#D97706,#FCD34D); }
.proba-fill.away { background:linear-gradient(90deg,#B91C1C,#F87171); }
.proba-pct { font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#111827; width:44px; text-align:right; font-weight:700; }

/* ─ Score cells ─ */
.score-grid { display:flex; flex-wrap:wrap; gap:.42rem; }
.score-cell { background:#F9FAFB; border:1.5px solid #E5E7EB; border-radius:8px; padding:.42rem .78rem; text-align:center; min-width:66px; }
.score-cell.best { border-color:#16A34A; background:#F0FDF4; }
.score-cell-score { font-family:'JetBrains Mono',monospace; font-size:.92rem; color:#111827; font-weight:700; }
.score-cell-pct   { font-size:.6rem; color:#9CA3AF; margin-top:.08rem; }

/* ─ ELO bars ─ */
.elo-row { display:flex; align-items:center; gap:.85rem; margin:.38rem 0; }
.elo-team { font-size:.78rem; color:#374151; width:120px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:500; }
.elo-track { flex:1; background:#F3F4F6; border-radius:999px; height:9px; overflow:hidden; }
.elo-fill  { height:100%; border-radius:999px; background:linear-gradient(90deg,#7C3AED,#A78BFA); }
.elo-val   { font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#6D28D9; width:50px; text-align:right; font-weight:700; }

/* ─ Compare bars ─ */
.compare-row { display:flex; align-items:center; gap:.5rem; margin:.35rem 0; }
.compare-label { color:#6B7280; text-transform:uppercase; font-size:.6rem; letter-spacing:.08em; width:150px; flex-shrink:0; }
.compare-val-h { font-family:'JetBrains Mono',monospace; font-size:.74rem; color:#15803D; width:46px; text-align:right; font-weight:700; }
.compare-val-a { font-family:'JetBrains Mono',monospace; font-size:.74rem; color:#DC2626; width:46px; text-align:left;  font-weight:700; }
.compare-bar-wrap { flex:1; position:relative; height:6px; background:#F3F4F6; border-radius:999px; overflow:hidden; }
.compare-bar-h { position:absolute; top:0; right:50%; height:100%; background:#16A34A; border-radius:999px 0 0 999px; }
.compare-bar-a { position:absolute; top:0; left:50%; height:100%; background:#EF4444; border-radius:0 999px 999px 0; }

/* ─ Dominance gauge ─ */
.dom-track { background:linear-gradient(90deg,#FCA5A5 0%,#F3F4F6 50%,#6EE7B7 100%); border-radius:999px; height:10px; position:relative; margin:.38rem 0; }
.dom-needle { position:absolute; top:50%; transform:translate(-50%,-50%); width:14px; height:14px; background:#111827; border-radius:50%; border:2.5px solid #FFFFFF; box-shadow:0 2px 6px rgba(0,0,0,.25); }

/* ─ H2H badges ─ */
.h2h-badges { display:flex; flex-wrap:wrap; gap:.28rem; margin-top:.38rem; }
.h2h-badge  { font-size:.72rem; font-weight:800; padding:.2rem .52rem; border-radius:6px; }
.h2h-badge.W { background:#DCFCE7; color:#15803D; }
.h2h-badge.D { background:#FEF9C3; color:#92400E; }
.h2h-badge.L { background:#FEE2E2; color:#B91C1C; }

/* ─ Tabs ─ */
[data-testid="stTabs"] button { color:#6B7280 !important; font-size:.82rem !important; font-weight:600 !important; background:transparent !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#15803D !important; border-bottom:2.5px solid #15803D !important; }

/* ─ Selectbox ─ */
[data-testid="stSelectbox"] > div > div { background:#FFFFFF !important; border-color:#D1D5DB !important; color:#111827 !important; }
label { color:#374151 !important; font-size:.8rem !important; font-weight:600 !important; }

/* ─ Toggle ─ */
[data-testid="stToggle"] label { color:#374151 !important; font-weight:500 !important; }

/* ─ Buttons ─ */
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
    transition: opacity .18s, transform .1s, background .15s !important;
    border: 1.5px solid #D1D5DB !important;
    background: #FFFFFF !important;
    color: #374151 !important;
    padding: .5rem 1rem !important;
}
[data-testid="stButton"] > button:hover {
    border-color: #86EFAC !important;
    background: #F0FDF4 !important;
    color: #15803D !important;
    transform: translateY(-1px) !important;
}
/* Predict button — ciblé par key via data-testid parent */
[data-testid="stButton"]:has(> button[data-testid="btn_predict"]) > button,
div:has(> [data-testid="stButton"] > button#btn_predict) > [data-testid="stButton"] > button,
[data-testid="stButton"][key="btn_predict"] > button {
    background: linear-gradient(135deg,#16A34A,#15803D) !important;
    color: #FFFFFF !important; font-weight: 800 !important; font-size: 1rem !important;
    letter-spacing: .06em !important; border: none !important; border-radius: 12px !important;
    padding: .78rem 2rem !important;
    box-shadow: 0 4px 16px rgba(21,128,61,.3) !important;
}
/* Fallback : dernier bouton dans sa colonne centrale = predict */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#16A34A,#15803D) !important;
    color: #FFFFFF !important; font-weight: 800 !important; font-size: 1rem !important;
    letter-spacing: .06em !important; border: none !important; border-radius: 12px !important;
    padding: .78rem 2rem !important;
    box-shadow: 0 4px 16px rgba(21,128,61,.3) !important;
}

/* ─ Nav buttons ─ */
[data-testid="stButton"][key="nav_pred"] > button,
[data-testid="stButton"][key="nav_expl"] > button {
    background: #F0FDF4 !important;
    color: #15803D !important;
    border: 1.5px solid #BBF7D0 !important;
    border-radius: 999px !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    padding: .38rem 1.1rem !important;
    width: auto !important;
    box-shadow: none !important;
    transition: all .15s !important;
}
[data-testid="stButton"][key="nav_pred"] > button:hover,
[data-testid="stButton"][key="nav_expl"] > button:hover {
    background: #DCFCE7 !important;
    border-color: #86EFAC !important;
    transform: none !important;
    opacity: 1 !important;
}

/* ─ Nav radio (fallback, kept for safety) ─ */
div[data-testid="stRadio"] > div { gap: .4rem !important; }
div[data-testid="stRadio"] label {
    background: #F0FDF4 !important;
    border: 1.5px solid #BBF7D0 !important;
    border-radius: 999px !important;
    padding: .38rem 1.1rem !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    color: #15803D !important;
    cursor: pointer !important;
}
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }

/* ─ Neutral selector card ─ */
.neutral-card {
    background: #FFFFFF; border: 1.5px solid #E5E7EB; border-radius: 12px;
    padding: .85rem 1.1rem; display: flex; align-items: flex-start; gap: .9rem;
    cursor: pointer; transition: border-color .18s, box-shadow .18s;
    position: relative;
}
.neutral-card:hover { border-color: #86EFAC; box-shadow: 0 2px 10px rgba(21,128,61,.1); }
.neutral-card.selected { border-color: #16A34A; background: #F0FDF4; }
.neutral-card-icon { font-size: 1.6rem; flex-shrink: 0; line-height: 1; margin-top: .1rem; }
.neutral-card-title { font-size: .82rem; font-weight: 700; color: #111827; }
.neutral-card-desc  { font-size: .72rem; color: #6B7280; margin-top: .18rem; line-height: 1.4; }
.neutral-card-badge {
    position: absolute; top: .6rem; right: .8rem;
    font-size: .62rem; font-weight: 700; padding: .18rem .55rem;
    border-radius: 999px; letter-spacing: .06em; text-transform: uppercase;
}
.neutral-card.selected .neutral-card-badge { background: #DCFCE7; color: #15803D; }
.neutral-card:not(.selected) .neutral-card-badge { background: #F3F4F6; color: #9CA3AF; }

/* ─ Warn box ─ */
.warn-box { background:#FFFBEB; border:1.5px solid #FCD34D; border-radius:9px; padding:.65rem 1rem; font-size:.78rem; color:#92400E; margin:.5rem 0; font-weight:500; }

/* ─ Stat highlight ─ */
.stat-big { font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:700; color:#111827; }
.stat-big.green  { color:#15803D; }
.stat-big.amber  { color:#D97706; }
.stat-big.red    { color:#DC2626; }

/* ─ Explainer ─ */
.explainer { max-width:760px; margin:0 auto; }
.explainer h2 { font-family:'Bebas Neue',sans-serif; color:#15803D; font-size:1.65rem; letter-spacing:.05em; margin-top:1.6rem; border-bottom:2px solid #DCFCE7; padding-bottom:.35rem; }
.explainer h3 { color:#111827; font-size:.95rem; font-weight:700; margin-top:1.1rem; }
.explainer p, .explainer li { color:#374151; font-size:.87rem; line-height:1.7; }
.explainer code { background:#F0FDF4; color:#15803D; padding:.1em .38em; border-radius:4px; font-size:.8rem; font-weight:600; border:1px solid #BBF7D0; }
.explainer .tag { display:inline-block; background:#EDE9FE; color:#5B21B6; font-size:.68rem; padding:.12rem .5rem; border-radius:5px; font-weight:700; margin-right:.3rem; letter-spacing:.04em; }
.explainer table { width:100%; border-collapse:collapse; font-size:.83rem; margin:.6rem 0 1rem; }
.explainer th { background:#F0FDF4; color:#15803D; font-weight:700; padding:.5rem .8rem; text-align:left; border-bottom:2px solid #BBF7D0; }
.explainer td { padding:.45rem .8rem; border-bottom:1px solid #F3F4F6; color:#374151; }
.explainer tr:hover td { background:#FAFFF6; }

hr { border-color:#E5E7EB !important; }

@media (max-width:640px) {
    .matchup-flag { font-size:2.8rem; }
    .score-display { font-size:3.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor(model_dir: Path):
    import sys
    for d in [SCRIPTS_DIR, PROJECT_ROOT]:
        if (d / "predict.py").exists():
            sys.path.insert(0, str(d))
            break
    from predict import Predictor
    return Predictor(model_dir)


# ── Nav bar ────────────────────────────────────────────────────────────────────
def render_nav():
    col_brand, _, col_nav = st.columns([3, 1, 3])
    with col_brand:
        st.markdown('<div class="topnav-brand" style="padding-top:.4rem;">CDM <span>2026</span> ⚽</div>', unsafe_allow_html=True)
    with col_nav:
        is_pred = st.session_state.page == "predictor"
        nc1, nc2 = st.columns(2)
        with nc1:
            # Style inline pour l'état actif — surcharge le CSS global
            active_style = "background:#15803D!important;color:#fff!important;border-color:#15803D!important;" if is_pred else ""
            st.markdown(f'<style>[data-testid="stButton"][key="nav_pred"]>button{{{active_style}}}</style>', unsafe_allow_html=True)
            if st.button("⚽ Prédicteur", key="nav_pred", use_container_width=True):
                st.session_state.page = "predictor"
                st.rerun()
        with nc2:
            active_style = "background:#15803D!important;color:#fff!important;border-color:#15803D!important;" if not is_pred else ""
            st.markdown(f'<style>[data-testid="stButton"][key="nav_expl"]>button{{{active_style}}}</style>', unsafe_allow_html=True)
            if st.button("📖 Comment ça marche ?", key="nav_expl", use_container_width=True):
                st.session_state.page = "explainer"
                st.rerun()


# ── HTML helpers ───────────────────────────────────────────────────────────────
def fmt_pct(v) -> str:
    return f"{v:.1%}" if v is not None else "—"

def fmt_f(v, d=2) -> str:
    return f"{v:.{d}f}" if v is not None else "—"

def metric_pill(label: str, value: str, tooltip: str, cls: str = "") -> str:
    return f"""
<div class="metric-pill">
  <div class="tooltip">{tooltip}</div>
  <div class="metric-label">{label}</div>
  <div class="metric-value {cls}">{value}</div>
</div>"""

def proba_bar(name: str, pct: float, cls: str) -> str:
    w = int((pct or 0) * 100)
    return f"""
<div class="proba-row">
  <div class="proba-name">{name}</div>
  <div class="proba-track"><div class="proba-fill {cls}" style="width:{w}%"></div></div>
  <div class="proba-pct">{fmt_pct(pct)}</div>
</div>"""

def compare_bar(label: str, val_h, val_a, low_good=False) -> str:
    if val_h is None or val_a is None:
        return ""
    total = (val_h + val_a) or 1
    wh = val_h / total * 50
    wa = val_a / total * 50
    if low_good:
        wh, wa = wa, wh
    return f"""
<div class="compare-row">
  <div class="compare-label">{label}</div>
  <div class="compare-val-h">{fmt_f(val_h,1)}</div>
  <div class="compare-bar-wrap">
    <div class="compare-bar-h" style="width:{wh:.1f}%"></div>
    <div class="compare-bar-a" style="width:{wa:.1f}%"></div>
  </div>
  <div class="compare-val-a">{fmt_f(val_a,1)}</div>
</div>"""


# ── PAGE PREDICTOR ─────────────────────────────────────────────────────────────
def page_predictor():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">🏆 FIFA World Cup 2026 · Analyse & Prédiction</div>
      <h1 class="hero-title">QUI VA <span>GAGNER</span> ?</h1>
    </div>
    """, unsafe_allow_html=True)

    col_h, col_v, col_a = st.columns([5, 1, 5])
    with col_h:
        st.markdown('<div class="section-label">🏠 Équipe 1</div>', unsafe_allow_html=True)
        home_display = st.selectbox("home_sel", DISPLAY_LIST,
                                    index=DISPLAY_LIST.index("France"),
                                    label_visibility="collapsed", key="sel_home")
    with col_v:
        st.markdown('<div style="text-align:center;padding-top:1rem;font-family:\'Bebas Neue\',sans-serif;font-size:2rem;color:#F59E0B;letter-spacing:.12em;">VS</div>', unsafe_allow_html=True)
    with col_a:
        st.markdown('<div class="section-label">✈️ Équipe 2</div>', unsafe_allow_html=True)
        away_display = st.selectbox("away_sel", DISPLAY_LIST,
                                    index=DISPLAY_LIST.index("Brazil"),
                                    label_visibility="collapsed", key="sel_away")

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    # ── Terrain neutre — sélecteur explicite centré ────────────────────────
    _, neutral_col, _ = st.columns([1, 4, 1])
    with neutral_col:
        st.markdown('<div class="section-label" style="text-align:center;margin-bottom:.5rem;">📍 Type de terrain</div>', unsafe_allow_html=True)
        neu_c1, neu_c2 = st.columns(2)

        neutral_sel = st.session_state.get("neutral_choice", "neutre")

        with neu_c1:
            sel_cls = "selected" if neutral_sel == "neutre" else ""
            st.markdown(f"""
            <div class="neutral-card {sel_cls}">
              <div class="neutral-card-icon">🌍</div>
              <div>
                <div class="neutral-card-title">Terrain neutre</div>
                <div class="neutral-card-desc">Aucune équipe joue à domicile.<br>Cas typique CDM 2026 (USA / Canada / Mexique).</div>
              </div>
              <span class="neutral-card-badge">{'✓ Sélectionné' if neutral_sel == 'neutre' else 'Cliquer'}</span>
            </div>""", unsafe_allow_html=True)
            if st.button("🌍 Terrain neutre", key="btn_neutre", use_container_width=True):
                st.session_state["neutral_choice"] = "neutre"
                st.rerun()

        with neu_c2:
            sel_cls = "selected" if neutral_sel == "domicile" else ""
            st.markdown(f"""
            <div class="neutral-card {sel_cls}">
              <div class="neutral-card-icon">🏠</div>
              <div>
                <div class="neutral-card-title">Équipe 1 à domicile</div>
                <div class="neutral-card-desc">L'équipe 1 joue chez elle.<br>Avantage terrain pris en compte dans le modèle.</div>
              </div>
              <span class="neutral-card-badge">{'✓ Sélectionné' if neutral_sel == 'domicile' else 'Cliquer'}</span>
            </div>""", unsafe_allow_html=True)
            if st.button("🏠 Équipe 1 à domicile", key="btn_domicile", use_container_width=True):
                st.session_state["neutral_choice"] = "domicile"
                st.rerun()

    neutral = (st.session_state.get("neutral_choice", "neutre") == "neutre")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # ── Bouton centré ─────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run = st.button("⚽  PRÉDIRE LE SCORE", key="btn_predict", use_container_width=True)

    home_key = DISPLAY_TO_KEY[home_display]
    away_key = DISPLAY_TO_KEY[away_display]

    cache_key = (home_key, away_key, neutral)

    if run:
        if home_key == away_key:
            st.markdown('<div class="warn-box">⚠️ Sélectionne deux équipes différentes.</div>', unsafe_allow_html=True)
            return
        # Lancer le calcul avec un spinner bien visible
        placeholder = st.empty()
        with placeholder.container():
            with st.spinner(""):
                st.markdown("""
                <div style="text-align:center;padding:2rem 0;">
                  <div style="font-size:2.5rem;margin-bottom:.6rem;">⚽</div>
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:#15803D;letter-spacing:.08em;">Analyse en cours…</div>
                  <div style="font-size:.8rem;color:#6B7280;margin-top:.35rem;">Le modèle calcule les probabilités</div>
                </div>
                """, unsafe_allow_html=True)
                try:
                    predictor = load_predictor(MODEL_DIR)
                    r = predictor.predict_score(home_key, away_key,
                                                neutral=neutral,
                                                tournament="FIFA World Cup")
                    st.session_state["pred_result"]    = r
                    st.session_state["pred_cache_key"] = cache_key
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Erreur : {e}")
                    st.exception(e)
                    return
        placeholder.empty()

    # Afficher le résultat stocké
    if "pred_result" not in st.session_state or st.session_state.get("pred_cache_key") != cache_key:
        return
    r = st.session_state["pred_result"]

    pred    = r["prediction"]
    res     = r["predicted_result"]
    res_cls = "home" if res == "home_win" else ("draw" if res == "draw" else "away")
    res_labels = {
        "home_win": f"VICTOIRE {home_display.upper()}",
        "draw":     "MATCH NUL",
        "away_win": f"VICTOIRE {away_display.upper()}",
    }
    conf = r["confidence_score"]

    # ── Matchup hero ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="matchup">
      <div style="display:flex;align-items:center;justify-content:center;gap:1.5rem;flex-wrap:wrap;">
        <div>
          <div class="matchup-flag">{flag(home_key)}</div>
          <div class="matchup-name">{home_display}</div>
        </div>
        <div style="text-align:center;">
          <div class="matchup-vs">VS</div>
          <div class="score-display">{pred['home']}<span class="score-dash">–</span>{pred['away']}</div>
          <span class="result-badge {res_cls}">{res_labels[res]}</span>
        </div>
        <div>
          <div class="matchup-flag">{flag(away_key)}</div>
          <div class="matchup-name">{away_display}</div>
        </div>
      </div>
      <div class="conf-wrap">
        <div class="conf-label">Indice de confiance du modèle</div>
        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{int(conf*100)}%"></div></div>
        <div class="conf-val">{conf:.0%}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    rp  = r["result_probs"]
    xg  = r["xg_analysis"]
    cs  = r["clean_sheet_probs"]
    h2h = r["h2h"]
    hi  = r["home_team_info"]
    ai  = r["away_team_info"]
    dom = r["dominance_index"]

    t1, t2, t3, t4 = st.tabs(["📊  Probabilités", "⚽  xG & Buts", "📈  ELO & Force", "🤝  H2H & Forme"])

    # ─ Tab 1 : Probabilités ───────────────────────────────────────────────
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            needle = (dom + 1) / 2 * 100
            upset  = r["upset"]
            fav_name = home_display if upset["favorite"] == "home" else away_display
            fav_flag = flag(home_key if upset["favorite"] == "home" else away_key)
            bars = (
                proba_bar(f"{flag(home_key)} {home_display}", rp["home_win"], "home") +
                proba_bar("⚖️ Match nul", rp["draw"], "draw") +
                proba_bar(f"{flag(away_key)} {away_display}", rp["away_win"], "away")
            )
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Résultat du match</div>
              {bars}
              <div style="margin-top:1.05rem;padding-top:.8rem;border-top:1.5px solid #F3F4F6;">
                <div class="card-title">Indice de dominance</div>
                <div class="dom-track"><div class="dom-needle" style="left:{needle:.1f}%"></div></div>
                <div style="display:flex;justify-content:space-between;font-size:.6rem;color:#9CA3AF;margin-top:.2rem;">
                  <span>← {away_display}</span><span>{home_display} →</span>
                </div>
                <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#111827;margin-top:.22rem;font-weight:700;">{dom:+.3f}</div>
              </div>
              <div style="margin-top:1rem;padding-top:.8rem;border-top:1.5px solid #F3F4F6;">
                <div class="card-title">Probabilité d'upset</div>
                <div style="font-size:.8rem;color:#374151;font-weight:500;">Favori ELO : <strong style="color:#111827;">{fav_flag} {fav_name}</strong></div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;color:#D97706;margin-top:.22rem;font-weight:700;">
                  {upset['upset_prob']:.1%} <span style="font-size:.65rem;color:#9CA3AF;font-weight:400;">de chances pour l'outsider</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        with c2:
            sc_html = '<div class="card"><div class="card-title">Top 5 scores les plus probables</div><div class="score-grid">'
            for i, s in enumerate(r["top5_scores"]):
                sc_html += f'<div class="score-cell{"  best" if i==0 else ""}"><div class="score-cell-score">{s["home"]}–{s["away"]}</div><div class="score-cell-pct">{s["proba"]:.1%}</div></div>'
            sc_html += "</div>"

            p_conf = (
                metric_pill("Confiance modèle", f"{conf:.0%}",
                    "Indice [0–1] combinant : clarté du résultat (50%), écart ELO (35%), richesse H2H (15%). Élevé = signal clair.",
                    "green" if conf > 0.6 else "amber") +
                metric_pill("Dominance",  f"{dom:+.3f}",
                    "Score composite [−1, +1] : 40% avantage xG + 35% avantage ELO + 25% win prob. +1 = domination totale équipe 1.") +
                metric_pill("Mode inférence",
                    "Symétrique ✓" if r["match_context"]["symmetric_inference"] else "Standard",
                    "En terrain neutre : double passe (A vs B + B vs A) puis moyenne des λ. Garantit predict(A,B) = predict(B,A).",
                    "green" if r["match_context"]["symmetric_inference"] else "")
            )
            sc_html += f'<div style="margin-top:.85rem;"><div class="card-title">Indicateurs clés</div><div class="metrics-grid">{p_conf}</div></div></div>'
            st.markdown(sc_html, unsafe_allow_html=True)

    # ─ Tab 2 : xG & Buts ─────────────────────────────────────────────────
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            pxg = (
                metric_pill(f"xG {home_display[:14]}", f"{xg['xg_home']}", f"Buts attendus pour {home_display} selon Poisson calibré. λ pilote toute la distribution.", "green") +
                metric_pill(f"xG {away_display[:14]}", f"{xg['xg_away']}", f"Buts attendus pour {away_display}.", "green") +
                metric_pill("xG Total",  f"{xg['xg_total']}", "Somme des deux λ = buts totaux attendus dans le match.") +
                metric_pill("xG Diff",   f"{xg['xg_diff']:+.3f}", f"λ_home − λ_away. Positif = avantage offensif {home_display}.")
            )
            pm = (
                metric_pill("Poisson GLM home", f"{r['poisson_lambda_home']:.3f}", "λ du GLM Poisson. Modèle linéaire très interprétable.") +
                metric_pill("Poisson GLM away", f"{r['poisson_lambda_away']:.3f}", "λ GLM pour l'équipe away.") +
                metric_pill("LightGBM home", f"{r['lgbm_lambda_home']:.3f}", "λ LightGBM (gradient boosting). Capture les non-linéarités.", "purple") +
                metric_pill("LightGBM away", f"{r['lgbm_lambda_away']:.3f}", "λ LGBM pour l'équipe away.", "purple") +
                metric_pill("Calibré home", f"{r['lambda_home']:.3f}", "Ensemble (60% LGBM + 40% GLM) puis calibrage Isotonic. Valeur finale utilisée.", "green") +
                metric_pill("Calibré away", f"{r['lambda_away']:.3f}", "Idem pour l'équipe away. Ces λ pilotent la matrice de probabilité.", "green")
            )
            st.markdown(f'<div class="card"><div class="card-title">Expected Goals (λ Poisson)</div><div class="metrics-grid">{pxg}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="card-title">Détail par modèle</div><div class="metrics-grid">{pm}</div></div>', unsafe_allow_html=True)

        with c2:
            pb = (
                metric_pill("BTTS",     fmt_pct(xg["btts_prob"]),     "Both Teams To Score : P(λ_home ≥ 1) × P(λ_away ≥ 1). Les deux équipes marquent.") +
                metric_pill("Over 0.5", fmt_pct(xg["over_0_5_prob"]), "P(total buts ≥ 1).") +
                metric_pill("Over 1.5", fmt_pct(xg["over_1_5_prob"]), "P(total buts ≥ 2).") +
                metric_pill("Over 2.5", fmt_pct(xg["over_2_5_prob"]), "P(total buts ≥ 3). Seuil le plus populaire chez les bookmakers.", "amber" if xg["over_2_5_prob"] > 0.5 else "") +
                metric_pill("Over 3.5", fmt_pct(xg["over_3_5_prob"]), "P(total buts ≥ 4). Match ouvert.") +
                metric_pill("Over 4.5", fmt_pct(xg["over_4_5_prob"]), "P(total buts ≥ 5). Festival offensif.")
            )
            pcs = (
                metric_pill(f"CS {home_display[:14]}", fmt_pct(cs["clean_sheet_home_prob"]), f"P({away_display} ne marque pas) = e^(−λ_away).") +
                metric_pill(f"CS {away_display[:14]}", fmt_pct(cs["clean_sheet_away_prob"]), f"P({home_display} ne marque pas) = e^(−λ_home).")
            )
            st.markdown(f'<div class="card"><div class="card-title">Marchés buts</div><div class="metrics-grid">{pb}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="card-title">Clean sheets</div><div class="metrics-grid">{pcs}</div></div>', unsafe_allow_html=True)

    # ─ Tab 3 : ELO & Force ───────────────────────────────────────────────
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            elo_max  = max(r["elo_home"], r["elo_away"], 1800)
            span     = max(elo_max - 1200, 1)
            wh = (r["elo_home"] - 1200) / span * 100
            wa = (r["elo_away"] - 1200) / span * 100
            ep = r["elo_win_prob_home"]
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Ratings ELO (recalculés sur tout l'historique)</div>
              <div class="elo-row"><div class="elo-team">{flag(home_key)} {home_display[:16]}</div><div class="elo-track"><div class="elo-fill" style="width:{wh:.1f}%"></div></div><div class="elo-val">{r['elo_home']:.0f}</div></div>
              <div class="elo-row"><div class="elo-team">{flag(away_key)} {away_display[:16]}</div><div class="elo-track"><div class="elo-fill" style="width:{wa:.1f}%"></div></div><div class="elo-val">{r['elo_away']:.0f}</div></div>
              <div style="margin-top:.9rem;padding-top:.7rem;border-top:1.5px solid #F3F4F6;">
                <div class="card-title">Probabilité ELO pure (sans ML)</div>
                {proba_bar(f"{flag(home_key)} {home_display}", ep, "home")}
                {proba_bar(f"{flag(away_key)} {away_display}", 1-ep, "away")}
                <div style="font-size:.66rem;color:#9CA3AF;margin-top:.35rem;">Formule ELO classique — baseline théorique indépendante des modèles</div>
              </div>
            </div>""", unsafe_allow_html=True)

            lbl = f'<div style="display:flex;justify-content:space-between;font-size:.62rem;font-weight:700;margin-bottom:.35rem;"><span style="color:#15803D;">{flag(home_key)} {home_display}</span><span style="color:#DC2626;">{flag(away_key)} {away_display}</span></div>'
            cmp = (
                compare_bar("Overall top11",     hi.get("top11_overall_mean"),   ai.get("top11_overall_mean")) +
                compare_bar("Attaque (shoot)",   hi.get("top11_shooting_mean"),  ai.get("top11_shooting_mean")) +
                compare_bar("Défense",           hi.get("top11_defending_mean"), ai.get("top11_defending_mean")) +
                compare_bar("Passe",             hi.get("top11_passing_mean"),   ai.get("top11_passing_mean")) +
                compare_bar("Vitesse (pace)",    hi.get("top11_pace_mean"),      ai.get("top11_pace_mean")) +
                compare_bar("Physique",          hi.get("top11_physic_mean"),    ai.get("top11_physic_mean"))
            )
            st.markdown(f'<div class="card"><div class="card-title">Comparaison Strength FIFA (top 11)</div>{lbl}{cmp}</div>', unsafe_allow_html=True)

        with c2:
            ps = (
                metric_pill(f"Strength {home_display[:13]}", fmt_f(hi.get("team_strength_score"),1), "Score composite FIFA (overall, potentiel, valeur marchande).", "green") +
                metric_pill(f"Strength {away_display[:13]}", fmt_f(ai.get("team_strength_score"),1), "Idem pour l'équipe adverse.", "green") +
                metric_pill(f"Valeur top11 {home_display[:9]}", f"{(hi.get('top11_value_sum_eur') or 0)/1e6:.0f}M€", "Valeur marchande cumulée des 11 meilleurs joueurs (données FIFA).") +
                metric_pill(f"Valeur top11 {away_display[:9]}", f"{(ai.get('top11_value_sum_eur') or 0)/1e6:.0f}M€", "Idem.") +
                metric_pill(f"≥85 top23 {home_display[:9]}", str(hi.get("count_85_plus_top23") or 0), "Joueurs top 23 avec note FIFA ≥ 85. Densité de talent élite.") +
                metric_pill(f"≥85 top23 {away_display[:9]}", str(ai.get("count_85_plus_top23") or 0), "Idem pour l'équipe adverse.")
            )
            pa = (
                metric_pill(f"Adv. buts {home_display[:12]}", f"{hi.get('home_adv_goals', 0) or 0:+.2f}", f"Delta buts/match {home_display} : domicile vs neutre. Positif = meilleur chez soi.") +
                metric_pill(f"Adv. buts {away_display[:12]}", f"{ai.get('home_adv_goals', 0) or 0:+.2f}", f"Idem pour {away_display}.") +
                metric_pill(f"Adv. WR {home_display[:13]}", fmt_pct(hi.get("home_adv_winrate")), f"Delta win rate domicile vs neutre pour {home_display}.") +
                metric_pill(f"Adv. WR {away_display[:13]}", fmt_pct(ai.get("home_adv_winrate")), "Idem pour l'équipe adverse.")
            )
            st.markdown(f'<div class="card"><div class="card-title">Score de force global</div><div class="metrics-grid">{ps}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="card-title">Avantage terrain historique</div><div class="metrics-grid">{pa}</div></div>', unsafe_allow_html=True)

    # ─ Tab 4 : H2H & Forme ───────────────────────────────────────────────
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            if h2h["h2h_n"] > 0:
                last   = h2h.get("h2h_last_results", [])
                badges = "".join(f'<span class="h2h-badge {b}">{b}</span>' for b in last[-10:])
                h2h_html = f"""
                <div class="card">
                  <div class="card-title">Head-to-Head (derniers {h2h['h2h_n']} matchs)</div>
                  <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:.8rem;">
                    <div><div class="metric-label">Matchs</div><div class="stat-big">{h2h['h2h_n']}</div></div>
                    <div><div class="metric-label">Victoires {flag(home_key)}</div><div class="stat-big green">{h2h['h2h_wins']}</div></div>
                    <div><div class="metric-label">Nuls</div><div class="stat-big amber">{h2h['h2h_draws']}</div></div>
                    <div><div class="metric-label">Défaites {flag(home_key)}</div><div class="stat-big red">{h2h['h2h_losses']}</div></div>
                  </div>
                  {proba_bar(f"{flag(home_key)} {home_display}", h2h['h2h_home_winrate'], "home")}
                  {proba_bar("⚖️ Nul", h2h['h2h_draw_rate'], "draw")}
                  {proba_bar(f"{flag(away_key)} {away_display}", h2h['h2h_away_winrate'], "away")}
                  <div style="margin-top:.8rem;padding-top:.7rem;border-top:1.5px solid #F3F4F6;">
                    <div class="card-title">Buts moyens par match</div>
                    <div style="display:flex;align-items:center;gap:1rem;font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;">
                      <span style="color:#15803D;">{flag(home_key)} {h2h['h2h_avg_goals_home']:.1f}</span>
                      <span style="color:#9CA3AF;">—</span>
                      <span style="color:#DC2626;">{h2h['h2h_avg_goals_away']:.1f} {flag(away_key)}</span>
                    </div>
                  </div>
                  <div style="margin-top:.75rem;">
                    <div class="card-title">Derniers résultats (point de vue {home_display})</div>
                    <div class="h2h-badges">{badges}</div>
                  </div>
                </div>"""
            else:
                h2h_html = '<div class="card"><div class="card-title">Head-to-Head</div><div style="color:#6B7280;font-size:.82rem;padding:.3rem 0;">Aucun historique de confrontation directe disponible.</div></div>'
            st.markdown(h2h_html, unsafe_allow_html=True)

        with c2:
            lbl = f'<div style="display:flex;justify-content:space-between;font-size:.62rem;font-weight:700;margin-bottom:.38rem;"><span style="color:#15803D;">{flag(home_key)} {home_display}</span><span style="color:#DC2626;">{flag(away_key)} {away_display}</span></div>'
            form = (
                compare_bar("Win rate (5 matchs)",       hi.get("win_last5"),           ai.get("win_last5")) +
                compare_bar("Win rate (10 matchs)",      hi.get("win_last10"),          ai.get("win_last10")) +
                compare_bar("Win rate (20 matchs)",      hi.get("win_last20"),          ai.get("win_last20")) +
                compare_bar("Buts marqués/match (5)",    hi.get("goals_scored_last5"),  ai.get("goals_scored_last5")) +
                compare_bar("Buts encaissés/match (5)",  hi.get("goals_conceded_last5"),ai.get("goals_conceded_last5"), low_good=True) +
                compare_bar("xG récents (EWM-10)",       hi.get("goals_scored_ewm10"),  ai.get("goals_scored_ewm10")) +
                compare_bar("Clean sheets (10 matchs)",  hi.get("clean_sheet_last10"),  ai.get("clean_sheet_last10"))
            )
            st.markdown(f'<div class="card"><div class="card-title">Forme récente</div>{lbl}{form}</div>', unsafe_allow_html=True)

    ctx   = r["match_context"]
    warns = []
    if not ctx["home_known_in_strength"]: warns.append(f"{home_display} absent du fichier strength")
    if not ctx["away_known_in_strength"]: warns.append(f"{away_display} absent du fichier strength")
    if warns:
        st.markdown(f'<div class="warn-box">⚠️ {" · ".join(warns)} — prédiction dégradée (features strength = 0).</div>', unsafe_allow_html=True)


# ── PAGE EXPLICATIONS ──────────────────────────────────────────────────────────
def page_explainer():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">📖 Documentation</div>
      <h1 class="hero-title">COMMENT ÇA <span>MARCHE</span> ?</h1>
    </div>
    <div class="explainer">

    <h2>Architecture du modèle</h2>
    <p>Le prédicteur combine deux familles de modèles entraînés indépendamment pour prédire
    les buts de chaque équipe — un pour les buts home, un pour les buts away :</p>
    <table>
      <tr><th>Modèle</th><th>Type</th><th>Rôle</th></tr>
      <tr><td><code>Poisson GLM</code></td><td>Régression linéaire généralisée</td><td>Interprétable, robuste sur petits effectifs</td></tr>
      <tr><td><code>LightGBM</code></td><td>Gradient boosting (objective=poisson)</td><td>Capture les interactions non-linéaires</td></tr>
      <tr><td><code>IsotonicRegression</code></td><td>Calibrage</td><td>Corrige le biais de surestimation des buts</td></tr>
    </table>
    <p>Les deux modèles sont combinés en <strong>ensemble pondéré (60% LGBM + 40% GLM)</strong>,
    puis calibrés. Les lambdas calibrés (λ_home, λ_away) alimentent deux lois de Poisson indépendantes
    pour générer la <strong>matrice P(home=i, away=j)</strong> 9×9 d'où toutes les probabilités sont dérivées.</p>

    <h2>Symétrie en terrain neutre</h2>
    <p>En terrain neutre, predict(France, Brésil) doit être identique à predict(Brésil, France). Deux mécanismes garantissent cela :</p>
    <ul>
      <li><strong>Entraînement</strong> : data augmentation — chaque match neutre est dupliqué avec les équipes inversées.</li>
      <li><strong>Inférence</strong> : double passe (A vs B + B vs A) puis λ_A = moyenne(λ_home(A,B), λ_away(B,A)). Symétrie mathématiquement garantie.</li>
    </ul>

    <h2>Features utilisées</h2>
    <h3>📡 ELO dynamique</h3>
    <p>Recalculé depuis 1872 sur tout l'historique. K-factor adaptatif (32 amical → 60 CDM) + multiplicateur goal-diff. Signal #1 en prédiction foot.</p>

    <h3>📊 Rolling stats (5 / 10 / 20 matchs)</h3>
    <p>Buts marqués/encaissés, win rate, clean sheet rate en fenêtre glissante. Plus une moyenne pondérée exponentielle (<code>EWM span=10</code>) pour la forme récente.</p>

    <h3>🎮 Strength FIFA</h3>
    <p>Notes FIFA des joueurs (top 11/23/5), valeurs marchandes, statistiques par poste (shooting, defending, passing, pace, physic, dribbling).</p>

    <h3>🤝 Head-to-Head</h3>
    <p>Bilan des 10 dernières confrontations directes : win rate, buts moyens, draw rate.</p>

    <h3>🏠 Home advantage factor</h3>
    <p>Delta de performance (buts et win rate) domicile vs terrain neutre, calculé individuellement par équipe.</p>

    <h2>Glossaire des métriques</h2>
    <h3>Score & résultat</h3>
    <p><span class="tag">prediction</span> Score exact correspondant au maximum de la matrice Poisson.</p>
    <p><span class="tag">predicted_result</span> home_win / draw / away_win selon la proba la plus haute parmi les 3.</p>
    <p><span class="tag">confidence_score</span> [0–1] : clarté du résultat (50%) + écart ELO (35%) + richesse H2H (15%).</p>
    <p><span class="tag">dominance_index</span> [−1, +1] : 40% xG + 35% ELO + 25% win prob. +1 = domination totale équipe 1.</p>
    <p><span class="tag">upset_prob</span> P(le favori ELO perd). Mesure la surprise potentielle du match.</p>

    <h3>Expected Goals (λ)</h3>
    <p><span class="tag">lambda_home/away</span> Buts attendus après calibrage. Paramètre λ de la loi de Poisson.</p>
    <p><span class="tag">btts_prob</span> Both Teams To Score : P(λ_home ≥ 1) × P(λ_away ≥ 1).</p>
    <p><span class="tag">over_X_5_prob</span> P(total buts > X). Over 2.5 = seuil bookmakers standard.</p>
    <p><span class="tag">clean_sheet</span> P(équipe adverse marque 0) = e^(−λ). Dérivé direct de Poisson.</p>

    <h3>ELO</h3>
    <p><span class="tag">elo_home/away</span> Rating actuel. 1500 = moyenne mondiale. Top nations ≈ 1900–2100.</p>
    <p><span class="tag">elo_win_prob_home</span> Probabilité ELO pure : <code>1 / (1 + 10^((elo_away − elo_home) / 400))</code>.</p>

    <h2>Performances</h2>
    <table>
      <tr><th>Métrique</th><th>Résultat</th><th>Référence naïve</th></tr>
      <tr><td>MAE buts home</td><td>≈ 1.05–1.10</td><td>—</td></tr>
      <tr><td>Résultat correct (H/N/D)</td><td>≈ 64%</td><td>~50% (modèle naïf)</td></tr>
      <tr><td>Score exact correct</td><td>≈ 12%</td><td>~7% (aléatoire)</td></tr>
    </table>
    <p>Évaluation sur les 15% de matchs les plus récents (split temporel strict, pas de data leakage).</p>

    </div>
    """, unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    render_nav()
    if st.session_state.page == "predictor":
        page_predictor()
    else:
        page_explainer()

if __name__ == "__main__" or True:
    main()