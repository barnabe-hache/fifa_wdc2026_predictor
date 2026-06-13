import time
from pathlib import Path

import numpy as np
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
# ou parents[0]

MODEL_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

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


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] { background-color: #0B0E17 !important; }
[data-testid="stAppViewContainer"] > .main { background-color: #0B0E17; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1080px !important; }
* { font-family: 'Inter', sans-serif; }

/* ─ Hero ─ */
.hero { text-align: center; padding: 1.5rem 1rem 1rem; }
.hero-eyebrow { font-size: .7rem; letter-spacing: .18em; color: #00E676; text-transform: uppercase; margin-bottom: .4rem; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: clamp(2.6rem,5.5vw,4.5rem); letter-spacing: .07em; color: #FFFFFF; line-height: 1; margin: 0; }
.hero-title span { color: #00E676; }

/* ─ Section label ─ */
.section-label { font-size: .65rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #5A6175; margin-bottom: .4rem; }

/* ─ Cards ─ */
.card { background: #161B27; border: 1px solid #222840; border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: .9rem; }
.card-title { font-size: .65rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: #5A6175; margin-bottom: .8rem; }

/* ─ Matchup hero ─ */
.matchup {
    background: linear-gradient(160deg, #111827 0%, #161B27 60%, #111827 100%);
    border: 1px solid #222840;
    border-radius: 18px; padding: 2rem 1.5rem;
    text-align: center; margin-bottom: 1.4rem; position: relative; overflow: hidden;
}
.matchup::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% -10%, rgba(0,230,118,.09) 0%, transparent 60%);
    pointer-events: none;
}
.matchup-flag { font-size: 4.2rem; line-height: 1; }
.matchup-name { font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; color: #FFFFFF; letter-spacing: .05em; margin-top: .25rem; }
.matchup-vs { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: #00E676; letter-spacing: .15em; text-shadow: 0 0 24px rgba(0,230,118,.45); }
.score-display { font-family: 'Bebas Neue', sans-serif; font-size: 4.8rem; color: #FFFFFF; letter-spacing: .05em; line-height: 1.05; margin-top: .3rem; }
.score-dash { color: #00E676; margin: 0 .25em; }
.result-badge { display: inline-block; font-weight: 700; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; padding: .28rem .85rem; border-radius: 999px; margin-top: .4rem; }
.result-badge.home { background: #00E676; color: #0B0E17; }
.result-badge.draw { background: #F59E0B; color: #0B0E17; }
.result-badge.away { background: #EF4444; color: #FFFFFF; }

/* ─ Confidence bar ─ */
.conf-wrap { max-width: 320px; margin: .9rem auto 0; }
.conf-label { font-size: .65rem; color: #5A6175; text-transform: uppercase; letter-spacing: .1em; }
.conf-bar-bg { background: #222840; border-radius: 999px; height: 5px; margin: .3rem 0; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #00C853, #00E676); }
.conf-val { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #5A6175; text-align: right; }

/* ─ Metric pill (tooltip on hover) ─ */
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(155px,1fr)); gap: .65rem; }
.metric-pill { background: #1E2435; border: 1px solid #2A3050; border-radius: 10px; padding: .75rem .95rem; position: relative; cursor: default; transition: border-color .2s, background .2s; }
.metric-pill:hover { background: #232B42; border-color: #3A4570; }
.metric-pill:hover .tooltip { display: block; }
.metric-label { font-size: .62rem; color: #6B7899; text-transform: uppercase; letter-spacing: .1em; line-height: 1.3; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.18rem; font-weight: 600; color: #E8EAF2; margin-top: .2rem; }
.metric-value.green { color: #00E676; }
.metric-value.gold  { color: #FBBF24; }
.metric-sub { font-size: .62rem; color: #8892B0; margin-top: .1rem; }

.tooltip {
    display: none; position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%);
    background: #0D1120; border: 1px solid #2A3050; border-radius: 9px;
    padding: .6rem .85rem; width: 230px; font-size: .71rem; color: #C5CAE0;
    line-height: 1.5; z-index: 999; box-shadow: 0 8px 28px rgba(0,0,0,.7); white-space: normal;
}
.tooltip::after { content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%); border: 6px solid transparent; border-top-color: #2A3050; }

/* ─ Proba bars ─ */
.proba-row { display: flex; align-items: center; gap: .7rem; margin: .4rem 0; }
.proba-name { font-size: .78rem; color: #C5CAE0; width: 130px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.proba-track { flex: 1; background: #1E2435; border-radius: 999px; height: 8px; overflow: hidden; }
.proba-fill { height: 100%; border-radius: 999px; }
.proba-fill.home { background: linear-gradient(90deg,#00C853,#00E676); }
.proba-fill.draw { background: linear-gradient(90deg,#D97706,#FBBF24); }
.proba-fill.away { background: linear-gradient(90deg,#B91C1C,#EF4444); }
.proba-pct { font-family: 'JetBrains Mono',monospace; font-size: .78rem; color: #E8EAF2; width: 44px; text-align: right; font-weight: 600; }

/* ─ Score cells ─ */
.score-grid { display: flex; flex-wrap: wrap; gap: .45rem; }
.score-cell { background: #1E2435; border: 1px solid #2A3050; border-radius: 8px; padding: .45rem .8rem; text-align: center; min-width: 68px; transition: border-color .2s; }
.score-cell.best { border-color: #00E676; background: rgba(0,230,118,.07); }
.score-cell-score { font-family: 'JetBrains Mono',monospace; font-size: .95rem; color: #E8EAF2; font-weight: 600; }
.score-cell-pct   { font-size: .62rem; color: #6B7899; margin-top: .1rem; }

/* ─ ELO bars ─ */
.elo-row { display: flex; align-items: center; gap: .9rem; margin: .4rem 0; }
.elo-team { font-size: .78rem; color: #C5CAE0; width: 120px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.elo-track { flex: 1; background: #1E2435; border-radius: 999px; height: 9px; overflow: hidden; }
.elo-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#6366F1,#818CF8); }
.elo-val   { font-family: 'JetBrains Mono',monospace; font-size: .78rem; color: #818CF8; width: 52px; text-align: right; font-weight: 600; }

/* ─ Compare bars ─ */
.compare-row { display: flex; align-items: center; gap: .5rem; margin: .38rem 0; }
.compare-label { color: #6B7899; text-transform: uppercase; font-size: .62rem; letter-spacing: .08em; width: 155px; flex-shrink: 0; }
.compare-val-h { font-family: 'JetBrains Mono',monospace; font-size: .75rem; color: #00E676; width: 48px; text-align: right; font-weight: 600; }
.compare-val-a { font-family: 'JetBrains Mono',monospace; font-size: .75rem; color: #EF4444; width: 48px; text-align: left;  font-weight: 600; }
.compare-bar-wrap { flex: 1; position: relative; height: 6px; background: #1E2435; border-radius: 999px; overflow: hidden; }
.compare-bar-h { position: absolute; top: 0; right: 50%; height: 100%; background: #00C853; border-radius: 999px 0 0 999px; }
.compare-bar-a { position: absolute; top: 0; left:  50%; height: 100%; background: #EF4444; border-radius: 0 999px 999px 0; }

/* ─ Dominance gauge ─ */
.dom-track { background: linear-gradient(90deg,#EF4444 0%,#1E2435 50%,#00E676 100%); border-radius: 999px; height: 10px; position: relative; margin: .4rem 0; }
.dom-needle { position: absolute; top: 50%; transform: translate(-50%,-50%); width: 14px; height: 14px; background: #FFFFFF; border-radius: 50%; border: 2px solid #0B0E17; box-shadow: 0 0 8px rgba(0,0,0,.8); }

/* ─ H2H badges ─ */
.h2h-badges { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; }
.h2h-badge { font-size: .72rem; font-weight: 700; padding: .22rem .55rem; border-radius: 6px; }
.h2h-badge.W { background: rgba(0,230,118,.15); color: #00E676; }
.h2h-badge.D { background: rgba(251,191,36,.15); color: #FBBF24; }
.h2h-badge.L { background: rgba(239,68,68,.15); color: #EF4444; }

/* ─ Tabs ─ */
[data-testid="stTabs"] button { color: #6B7899 !important; font-size: .82rem !important; font-weight: 500 !important; background: transparent !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #00E676 !important; border-bottom: 2px solid #00E676 !important; }

/* ─ Selectbox ─ */
[data-testid="stSelectbox"] > div > div { background: #161B27 !important; border-color: #2A3050 !important; color: #E8EAF2 !important; }
[data-baseweb="select"] { background: #161B27 !important; }
label { color: #8892B0 !important; font-size: .78rem !important; }

/* ─ Toggle ─ */
[data-testid="stToggle"] label { color: #C5CAE0 !important; }

/* ─ Button ─ */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#00C853,#00A846) !important;
    color: #0B0E17 !important; font-weight: 800 !important; font-size: .95rem !important;
    letter-spacing: .06em !important; border: none !important; border-radius: 12px !important;
    padding: .7rem 2rem !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(0,200,83,.3) !important;
    transition: opacity .2s, transform .1s !important;
}
[data-testid="stButton"] > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

/* ─ Radio nav ─ */
[data-testid="stRadio"] label { font-size: .8rem !important; color: #8892B0 !important; }
[data-testid="stRadio"] div[data-checked="true"] label { color: #00E676 !important; font-weight: 600 !important; }

/* ─ Warn box ─ */
.warn-box { background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.25); border-radius: 9px; padding: .7rem 1rem; font-size: .78rem; color: #FCD34D; margin: .5rem 0; }

/* ─ Explainer ─ */
.explainer h2 { font-family: 'Bebas Neue',sans-serif; color: #00E676; font-size: 1.7rem; letter-spacing: .05em; margin-top: 1.5rem; border-bottom: 1px solid #222840; padding-bottom: .4rem; }
.explainer h3 { color: #E8EAF2; font-size: .95rem; font-weight: 600; margin-top: 1.1rem; }
.explainer p, .explainer li { color: #8892B0; font-size: .86rem; line-height: 1.65; }
.explainer code { background: #1E2435; color: #00E676; padding: .1em .35em; border-radius: 4px; font-size: .8rem; }
.explainer .tag { display: inline-block; background: #1E2435; color: #818CF8; font-size: .68rem; padding: .12rem .5rem; border-radius: 5px; font-weight: 700; margin-right: .3rem; letter-spacing: .05em; }

hr { border-color: #222840 !important; }

@media (max-width: 640px) {
    .matchup-flag { font-size: 3rem; }
    .score-display { font-size: 3.2rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor(model_dir: Path):
    import sys
    # Cherche predict.py d'abord dans scripts/, puis à la racine
    for d in [SCRIPTS_DIR, PROJECT_ROOT]:
        if (d / "predict.py").exists():
            sys.path.insert(0, str(d))
            break
    from predict import Predictor
    return Predictor(model_dir)


# ── Helpers HTML ──────────────────────────────────────────────────────────────
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
  <div class="compare-val-h">{fmt_f(val_h, 1)}</div>
  <div class="compare-bar-wrap">
    <div class="compare-bar-h" style="width:{wh:.1f}%"></div>
    <div class="compare-bar-a" style="width:{wa:.1f}%"></div>
  </div>
  <div class="compare-val-a">{fmt_f(val_a, 1)}</div>
</div>"""


# ── PAGE PREDICTOR ────────────────────────────────────────────────────────────
def page_predictor():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">⚽ Coupe du Monde 2026 · Analyse & Prédiction</div>
      <h1 class="hero-title">CDM <span>2026</span> PREDICTOR</h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Sélection équipes ──────────────────────────────────────────────────
    col_h, col_v, col_a = st.columns([5, 1, 5])
    with col_h:
        st.markdown('<div class="section-label">🏠 Équipe 1</div>', unsafe_allow_html=True)
        home_display = st.selectbox("home_sel", DISPLAY_LIST,
                                    index=DISPLAY_LIST.index("France"),
                                    label_visibility="collapsed", key="sel_home")
    with col_v:
        st.markdown('<div style="text-align:center;padding-top:1.1rem;font-family:Bebas Neue;font-size:1.8rem;color:#00E676;letter-spacing:.1em;">VS</div>', unsafe_allow_html=True)
    with col_a:
        st.markdown('<div class="section-label">✈️ Équipe 2</div>', unsafe_allow_html=True)
        away_display = st.selectbox("away_sel", DISPLAY_LIST,
                                    index=DISPLAY_LIST.index("Brazil"),
                                    label_visibility="collapsed", key="sel_away")

    # ── Options & bouton centré ───────────────────────────────────────────
    st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
    opt_l, opt_c, opt_r = st.columns([2, 3, 2])
    with opt_c:
        neutral = st.toggle("⚖️ Terrain neutre", value=True, key="tog_neutral",
                            help="Activé = aucune équipe joue à domicile (cas CDM 2026 : USA/Canada/Mexique)")

    st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run = st.button("⚽  PRÉDIRE LE SCORE", key="btn_predict")

    # ── Prédiction ────────────────────────────────────────────────────────
    if run:
        home_key = DISPLAY_TO_KEY[home_display]
        away_key = DISPLAY_TO_KEY[away_display]

        if home_key == away_key:
            st.markdown('<div class="warn-box">⚠️ Sélectionne deux équipes différentes.</div>', unsafe_allow_html=True)
            return

        with st.spinner("Analyse en cours…"):
            try:
                predictor = load_predictor(MODEL_DIR)
                r = predictor.predict_score(home_key, away_key,
                                            neutral=neutral,
                                            tournament="FIFA World Cup")
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.exception(e)
                return

        pred   = r["prediction"]
        res    = r["predicted_result"]
        res_cls = "home" if res == "home_win" else ("draw" if res == "draw" else "away")
        res_labels = {"home_win": f"VICTOIRE · {home_display}", "draw": "MATCH NUL", "away_win": f"VICTOIRE · {away_display}"}
        conf   = r["confidence_score"]

        # ── Matchup hero ────────────────────────────────────────────────
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
            <div class="conf-label">Indice de confiance</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{int(conf*100)}%"></div></div>
            <div class="conf-val">{conf:.0%}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs ─────────────────────────────────────────────────────────
        t1, t2, t3, t4 = st.tabs(["📊  Probabilités", "⚽  xG & Buts", "📈  ELO & Force", "🤝  H2H & Forme"])

        rp  = r["result_probs"]
        xg  = r["xg_analysis"]
        cs  = r["clean_sheet_probs"]
        h2h = r["h2h"]
        hi  = r["home_team_info"]
        ai  = r["away_team_info"]
        dom = r["dominance_index"]

        # ─ Tab 1 ──────────────────────────────────────────────────────────
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                needle = (dom + 1) / 2 * 100
                bars_html = (
                    proba_bar(f"{flag(home_key)} {home_display}", rp["home_win"], "home") +
                    proba_bar("⚖️ Match nul", rp["draw"], "draw") +
                    proba_bar(f"{flag(away_key)} {away_display}", rp["away_win"], "away")
                )
                upset = r["upset"]
                fav_name = home_display if upset["favorite"] == "home" else away_display
                fav_flag = flag(home_key if upset["favorite"] == "home" else away_key)
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">Résultat du match</div>
                  {bars_html}
                  <div style="margin-top:1.1rem;">
                    <div class="card-title">Indice de dominance</div>
                    <div class="dom-track"><div class="dom-needle" style="left:{needle:.1f}%"></div></div>
                    <div style="display:flex;justify-content:space-between;font-size:.62rem;color:#5A6175;margin-top:.2rem;">
                      <span>← {away_display}</span><span>{home_display} →</span>
                    </div>
                    <div style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#E8EAF2;margin-top:.25rem;font-weight:600;">{dom:+.3f}</div>
                  </div>
                  <div style="margin-top:1rem;padding-top:.8rem;border-top:1px solid #222840;">
                    <div class="card-title">Probabilité d'upset</div>
                    <div style="font-size:.8rem;color:#8892B0;">Favori ELO : <strong style="color:#E8EAF2;">{fav_flag} {fav_name}</strong></div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;color:#FBBF24;margin-top:.25rem;font-weight:600;">
                      {upset['upset_prob']:.1%} <span style="font-size:.65rem;color:#6B7899;">de chances pour l'outsider</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            with c2:
                sc_html = '<div class="card"><div class="card-title">Top 5 scores les plus probables</div><div class="score-grid">'
                for i, s in enumerate(r["top5_scores"]):
                    best = " best" if i == 0 else ""
                    sc_html += f'<div class="score-cell{best}"><div class="score-cell-score">{s["home"]}–{s["away"]}</div><div class="score-cell-pct">{s["proba"]:.1%}</div></div>'
                sc_html += "</div>"

                pills_conf = ""
                pills_conf += metric_pill("Confiance modèle", f"{conf:.0%}",
                    "Indice [0–1] : élevé si signal ELO fort, H2H disponible et résultat net. Combine clarté du résultat (50%), force ELO (35%) et richesse H2H (15%).",
                    "green" if conf > 0.6 else "gold")
                pills_conf += metric_pill("Dominance", f"{dom:+.3f}",
                    "Score composite [−1, +1] combinant xG (40%), ELO (35%) et win prob (25%). +1 = domination totale du côté gauche.")
                pills_conf += metric_pill("Inférence", "Symétrique ✓" if r["match_context"]["symmetric_inference"] else "Standard",
                    "En terrain neutre : double passe forward (A vs B + B vs A) puis moyenne des λ. Garantit predict(A,B) == predict(B,A).")
                sc_html += f'<div style="margin-top:.9rem;"><div class="card-title">Indicateurs clés</div><div class="metrics-grid">{pills_conf}</div></div></div>'
                st.markdown(sc_html, unsafe_allow_html=True)

        # ─ Tab 2 ──────────────────────────────────────────────────────────
        with t2:
            c1, c2 = st.columns(2)
            with c1:
                pills_xg = ""
                pills_xg += metric_pill(f"xG {home_display[:13]}", f"{xg['xg_home']}", f"Buts attendus pour {home_display} selon le modèle Poisson calibré. λ = paramètre de la loi de Poisson.", "green")
                pills_xg += metric_pill(f"xG {away_display[:13]}", f"{xg['xg_away']}", f"Buts attendus pour {away_display}.", "green")
                pills_xg += metric_pill("xG Total", f"{xg['xg_total']}", "Somme des deux λ = nombre total de buts attendus dans la rencontre.")
                pills_xg += metric_pill("xG Diff", f"{xg['xg_diff']:+.3f}", f"λ_home − λ_away. Positif = avantage offensif {home_display}.")
                pills_m = ""
                pills_m += metric_pill("Poisson GLM home", f"{r['poisson_lambda_home']:.3f}", "λ prédit par la Régression de Poisson (GLM). Modèle linéaire très interprétable, robuste sur petits effectifs.")
                pills_m += metric_pill("Poisson GLM away", f"{r['poisson_lambda_away']:.3f}", "λ du GLM Poisson pour l'équipe away.")
                pills_m += metric_pill("LightGBM home", f"{r['lgbm_lambda_home']:.3f}", "λ prédit par LightGBM (gradient boosting, objective=poisson). Capture les interactions non-linéaires.")
                pills_m += metric_pill("LightGBM away", f"{r['lgbm_lambda_away']:.3f}", "λ LightGBM pour l'équipe away.")
                pills_m += metric_pill("Ensemble calibré home", f"{r['lambda_home']:.3f}", "60% LGBM + 40% GLM, puis calibrage Isotonic pour corriger le biais de surestimation.", "green")
                pills_m += metric_pill("Ensemble calibré away", f"{r['lambda_away']:.3f}", "Idem pour l'équipe away. Ces valeurs pilotent toute la matrice de probabilité.", "green")
                st.markdown(f'<div class="card"><div class="card-title">Expected Goals (λ Poisson)</div><div class="metrics-grid">{pills_xg}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card"><div class="card-title">Détail par modèle</div><div class="metrics-grid">{pills_m}</div></div>', unsafe_allow_html=True)

            with c2:
                pills_b = ""
                pills_b += metric_pill("BTTS", fmt_pct(xg["btts_prob"]),  "Both Teams To Score : P(λ_home ≥ 1) × P(λ_away ≥ 1). Les deux équipes marquent.")
                pills_b += metric_pill("Over 0.5", fmt_pct(xg["over_0_5_prob"]), "P(total buts ≥ 1). Quasi-certain sauf fort hold-out.")
                pills_b += metric_pill("Over 1.5", fmt_pct(xg["over_1_5_prob"]), "P(total buts ≥ 2).")
                pills_b += metric_pill("Over 2.5", fmt_pct(xg["over_2_5_prob"]), "P(total buts ≥ 3). Seuil le plus échangé chez les bookmakers.", "gold" if xg["over_2_5_prob"] > 0.5 else "")
                pills_b += metric_pill("Over 3.5", fmt_pct(xg["over_3_5_prob"]), "P(total buts ≥ 4). Match ouvert.")
                pills_b += metric_pill("Over 4.5", fmt_pct(xg["over_4_5_prob"]), "P(total buts ≥ 5). Festival offensif.")
                pills_cs = ""
                pills_cs += metric_pill(f"CS {home_display[:14]}", fmt_pct(cs["clean_sheet_home_prob"]), f"P({away_display} ne marque pas) = e^(−λ_away). Probabilité de clean sheet pour {home_display}.")
                pills_cs += metric_pill(f"CS {away_display[:14]}", fmt_pct(cs["clean_sheet_away_prob"]), f"P({home_display} ne marque pas) = e^(−λ_home). Clean sheet pour {away_display}.")
                st.markdown(f'<div class="card"><div class="card-title">Marchés buts</div><div class="metrics-grid">{pills_b}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card"><div class="card-title">Clean sheets</div><div class="metrics-grid">{pills_cs}</div></div>', unsafe_allow_html=True)

        # ─ Tab 3 ──────────────────────────────────────────────────────────
        with t3:
            c1, c2 = st.columns(2)
            with c1:
                elo_max = max(r["elo_home"], r["elo_away"], 1800)
                span = max(elo_max - 1200, 1)
                wh = (r["elo_home"] - 1200) / span * 100
                wa = (r["elo_away"] - 1200) / span * 100
                elo_prob_h = r["elo_win_prob_home"]
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">Ratings ELO</div>
                  <div class="elo-row"><div class="elo-team">{flag(home_key)} {home_display[:16]}</div><div class="elo-track"><div class="elo-fill" style="width:{wh:.1f}%"></div></div><div class="elo-val">{r['elo_home']:.0f}</div></div>
                  <div class="elo-row"><div class="elo-team">{flag(away_key)} {away_display[:16]}</div><div class="elo-track"><div class="elo-fill" style="width:{wa:.1f}%"></div></div><div class="elo-val">{r['elo_away']:.0f}</div></div>
                  <div style="margin-top:.9rem;padding-top:.7rem;border-top:1px solid #222840;">
                    <div class="card-title">Probabilité ELO pure (baseline)</div>
                    {proba_bar(f"{flag(home_key)} {home_display}", elo_prob_h, "home")}
                    {proba_bar(f"{flag(away_key)} {away_display}", 1 - elo_prob_h, "away")}
                    <div style="font-size:.7rem;color:#5A6175;margin-top:.4rem;">ELO seul, sans ML ni forme récente</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                lbl_h = f"""<div style="display:flex;justify-content:space-between;font-size:.65rem;margin-bottom:.35rem;">
                  <span style="color:#00E676;">{flag(home_key)} {home_display}</span>
                  <span style="color:#EF4444;">{flag(away_key)} {away_display}</span></div>"""
                cmp_html = (
                    compare_bar("Overall moyen top11", hi.get("top11_overall_mean"), ai.get("top11_overall_mean")) +
                    compare_bar("Attaque (shooting)", hi.get("top11_shooting_mean"), ai.get("top11_shooting_mean")) +
                    compare_bar("Défense", hi.get("top11_defending_mean"), ai.get("top11_defending_mean")) +
                    compare_bar("Passe", hi.get("top11_passing_mean"), ai.get("top11_passing_mean")) +
                    compare_bar("Vitesse (pace)", hi.get("top11_pace_mean"), ai.get("top11_pace_mean")) +
                    compare_bar("Physique", hi.get("top11_physic_mean"), ai.get("top11_physic_mean"))
                )
                st.markdown(f'<div class="card"><div class="card-title">Comparaison Strength FIFA (top 11)</div>{lbl_h}{cmp_html}</div>', unsafe_allow_html=True)

            with c2:
                pills_s = ""
                pills_s += metric_pill(f"Strength {home_display[:13]}", fmt_f(hi.get("team_strength_score"), 1), "Score composite FIFA : agrège overall, potentiel et valeur marchande des meilleurs joueurs.", "green")
                pills_s += metric_pill(f"Strength {away_display[:13]}", fmt_f(ai.get("team_strength_score"), 1), "Idem pour l'équipe adverse.", "green")
                pills_s += metric_pill(f"Valeur top11 {home_display[:9]}", f"{(hi.get('top11_value_sum_eur') or 0)/1e6:.0f}M€", "Valeur marchande cumulée des 11 meilleurs joueurs (données FIFA).")
                pills_s += metric_pill(f"Valeur top11 {away_display[:9]}", f"{(ai.get('top11_value_sum_eur') or 0)/1e6:.0f}M€", "Idem.")
                pills_s += metric_pill(f"≥85 top23 {home_display[:9]}", str(hi.get("count_85_plus_top23") or 0), "Joueurs dans le top 23 avec note FIFA ≥ 85. Mesure la densité de talent élite.")
                pills_s += metric_pill(f"≥85 top23 {away_display[:9]}", str(ai.get("count_85_plus_top23") or 0), "Idem pour l'équipe adverse.")
                pills_adv = ""
                pills_adv += metric_pill(f"Adv. buts {home_display[:11]}", f"{hi.get('home_adv_goals', 0) or 0:+.2f}", f"Delta buts/match de {home_display} à domicile vs terrain neutre. Positif = meilleur chez soi.")
                pills_adv += metric_pill(f"Adv. buts {away_display[:11]}", f"{ai.get('home_adv_goals', 0) or 0:+.2f}", f"Idem pour {away_display}.")
                pills_adv += metric_pill(f"Adv. WR {home_display[:12]}", fmt_pct(hi.get("home_adv_winrate")), f"Delta win rate domicile vs neutre pour {home_display}. Un fort positif = équipe qui souffre sur terrain neutre.")
                pills_adv += metric_pill(f"Adv. WR {away_display[:12]}", fmt_pct(ai.get("home_adv_winrate")), "Idem.")
                st.markdown(f'<div class="card"><div class="card-title">Score de force global</div><div class="metrics-grid">{pills_s}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card"><div class="card-title">Avantage terrain historique</div><div class="metrics-grid">{pills_adv}</div></div>', unsafe_allow_html=True)

        # ─ Tab 4 ──────────────────────────────────────────────────────────
        with t4:
            c1, c2 = st.columns(2)
            with c1:
                if h2h["h2h_n"] > 0:
                    last = h2h.get("h2h_last_results", [])
                    badges = "".join(f'<span class="h2h-badge {b}">{b}</span>' for b in last[-10:])
                    h2h_html = f"""
                    <div class="card">
                      <div class="card-title">Head-to-Head (derniers {h2h['h2h_n']} matchs)</div>
                      <div style="display:flex;gap:1.8rem;flex-wrap:wrap;margin-bottom:.8rem;">
                        <div><div class="metric-label">Matchs</div><div class="metric-value">{h2h['h2h_n']}</div></div>
                        <div><div class="metric-label">V {flag(home_key)}</div><div class="metric-value green">{h2h['h2h_wins']}</div></div>
                        <div><div class="metric-label">Nuls</div><div class="metric-value gold">{h2h['h2h_draws']}</div></div>
                        <div><div class="metric-label">D {flag(home_key)}</div><div style="font-family:'JetBrains Mono',monospace;font-size:1.18rem;font-weight:600;color:#EF4444;">{h2h['h2h_losses']}</div></div>
                      </div>
                      {proba_bar(f"{flag(home_key)} {home_display}", h2h['h2h_home_winrate'], "home")}
                      {proba_bar("⚖️ Nul", h2h['h2h_draw_rate'], "draw")}
                      {proba_bar(f"{flag(away_key)} {away_display}", h2h['h2h_away_winrate'], "away")}
                      <div style="margin-top:.8rem;padding-top:.7rem;border-top:1px solid #222840;">
                        <div class="card-title">Buts moyens par match</div>
                        <div style="display:flex;align-items:center;gap:1rem;font-family:'JetBrains Mono',monospace;font-size:1rem;">
                          <span style="color:#00E676;font-weight:600;">{flag(home_key)} {h2h['h2h_avg_goals_home']:.1f}</span>
                          <span style="color:#5A6175;">—</span>
                          <span style="color:#EF4444;font-weight:600;">{h2h['h2h_avg_goals_away']:.1f} {flag(away_key)}</span>
                        </div>
                      </div>
                      <div style="margin-top:.8rem;">
                        <div class="card-title">Derniers résultats ({home_display})</div>
                        <div class="h2h-badges">{badges}</div>
                      </div>
                    </div>"""
                else:
                    h2h_html = '<div class="card"><div class="card-title">Head-to-Head</div><div style="color:#5A6175;font-size:.82rem;padding:.3rem 0;">Aucun historique de confrontation directe disponible.</div></div>'
                st.markdown(h2h_html, unsafe_allow_html=True)

            with c2:
                lbl_h = f"""<div style="display:flex;justify-content:space-between;font-size:.65rem;margin-bottom:.4rem;">
                  <span style="color:#00E676;">{flag(home_key)} {home_display}</span>
                  <span style="color:#EF4444;">{flag(away_key)} {away_display}</span></div>"""
                form_cmp = (
                    compare_bar("Win rate (5 matchs)", hi.get("win_last5"), ai.get("win_last5")) +
                    compare_bar("Win rate (10 matchs)", hi.get("win_last10"), ai.get("win_last10")) +
                    compare_bar("Win rate (20 matchs)", hi.get("win_last20"), ai.get("win_last20")) +
                    compare_bar("Buts marqués/match (5)", hi.get("goals_scored_last5"), ai.get("goals_scored_last5")) +
                    compare_bar("Buts encaissés/match (5)", hi.get("goals_conceded_last5"), ai.get("goals_conceded_last5"), low_good=True) +
                    compare_bar("xG récents EWM-10", hi.get("goals_scored_ewm10"), ai.get("goals_scored_ewm10")) +
                    compare_bar("Clean sheets (10 matchs)", hi.get("clean_sheet_last10"), ai.get("clean_sheet_last10"))
                )
                st.markdown(f'<div class="card"><div class="card-title">Forme récente (5 / 10 / 20 matchs)</div>{lbl_h}{form_cmp}</div>', unsafe_allow_html=True)

        ctx = r["match_context"]
        warns = []
        if not ctx["home_known_in_strength"]: warns.append(f"{home_display} absent du fichier strength")
        if not ctx["away_known_in_strength"]: warns.append(f"{away_display} absent du fichier strength")
        if warns:
            st.markdown(f'<div class="warn-box">⚠️ {" · ".join(warns)} — prédiction dégradée.</div>', unsafe_allow_html=True)


# ── PAGE EXPLICATIONS ─────────────────────────────────────────────────────────
def page_explainer():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Documentation</div>
      <h1 class="hero-title">COMMENT ÇA <span>MARCHE</span> ?</h1>
    </div>
    <div class="explainer">

    <h2>Architecture du modèle</h2>
    <p>Le prédicteur combine deux familles de modèles entraînés indépendamment pour prédire
    les buts de chaque équipe :</p>
    <ul>
      <li><code>Poisson GLM</code> — régression de Poisson (sklearn <em>PoissonRegressor</em>).
      Modèle linéaire généralisé conçu pour les données de comptage. Très interprétable.</li>
      <li><code>LightGBM</code> avec <code>objective=poisson</code> — gradient boosting qui
      capture les interactions non-linéaires (ex : attaque forte vs défense très faible
      n'est pas simplement additive).</li>
    </ul>
    <p>Les deux sont combinés en <strong>ensemble pondéré (60% LGBM + 40% GLM)</strong>,
    puis calibrés via <code>IsotonicRegression</code> pour corriger le biais de surestimation des buts.</p>
    <p>Les lambdas calibrés (λ_home, λ_away) alimentent deux lois de Poisson indépendantes
    pour générer la <strong>matrice P(home=i, away=j)</strong> 9×9 à partir de laquelle
    toutes les probabilités sont dérivées.</p>

    <h2>Symétrie en terrain neutre</h2>
    <p>En terrain neutre, predict(A, B) doit être identique à predict(B, A).
    Pour garantir cela, deux mécanismes sont combinés :</p>
    <ul>
      <li><strong>Data augmentation à l'entraînement</strong> : chaque match neutre est
      dupliqué avec les équipes inversées. Le modèle apprend que home/away est arbitraire
      sur terrain neutre.</li>
      <li><strong>Double inférence à la prédiction</strong> : on prédit A vs B ET B vs A,
      puis λ_A = moyenne(λ_home(A,B), λ_away(B,A)). Symétrie mathématiquement garantie.</li>
    </ul>

    <h2>Features utilisées</h2>

    <h3>📡 ELO dynamique</h3>
    <p>Recalculé sur tout l'historique depuis 1872. K-factor adaptatif (32 en amical → 60 en CDM)
    avec multiplicateur goal-diff. C'est le signal #1 en prédiction foot.</p>

    <h3>📊 Rolling stats (5 / 10 / 20 matchs)</h3>
    <p>Buts marqués/encaissés, win rate, clean sheet rate, en fenêtre glissante.
    Plus une moyenne pondérée exponentielle (<code>EWM span=10</code>) pour capter la forme récente.</p>

    <h3>🎮 Strength FIFA</h3>
    <p>Notes FIFA des joueurs (top 11/23/5), valeurs marchandes, stats par poste
    (shooting, defending, passing, pace, physic, dribbling).</p>

    <h3>🤝 Head-to-Head</h3>
    <p>Sur les 10 dernières confrontations directes : win rate, buts moyens, draw rate.</p>

    <h3>🏠 Home advantage factor</h3>
    <p>Delta de performance (buts et win rate) domicile vs terrain neutre, par équipe.</p>

    <h2>Glossaire des métriques</h2>

    <h3>Score & résultat</h3>
    <p><span class="tag">prediction</span> Score exact (max de la matrice Poisson).</p>
    <p><span class="tag">predicted_result</span> home_win / draw / away_win selon la proba la plus haute.</p>
    <p><span class="tag">confidence_score</span> [0–1] : clarté du résultat (50%) + écart ELO (35%) + richesse H2H (15%).</p>
    <p><span class="tag">dominance_index</span> [−1, +1] : 40% xG + 35% ELO + 25% win prob. +1 = domination totale.</p>
    <p><span class="tag">upset_prob</span> P(le favori ELO perd). Mesure la surprise potentielle.</p>

    <h3>xG / λ</h3>
    <p><span class="tag">lambda_home/away</span> Buts attendus après calibrage. Pilote toute la distribution.</p>
    <p><span class="tag">btts_prob</span> Both Teams To Score : P(λ_home ≥ 1) × P(λ_away ≥ 1).</p>
    <p><span class="tag">over_X_5_prob</span> P(total buts > X). Over 2.5 = seuil bookmakers standard.</p>
    <p><span class="tag">clean_sheet_*</span> P(équipe adverse marque 0) = e^(−λ). Dérivé de Poisson.</p>

    <h3>ELO</h3>
    <p><span class="tag">elo_home/away</span> Rating actuel. 1500 = moyenne mondiale. Top nations ~1900–2100.</p>
    <p><span class="tag">elo_win_prob_home</span> Probabilité ELO pure : 1/(1+10^((elo_away−elo_home)/400)).</p>

    <h2>Performances du modèle</h2>
    <p>Évaluées sur les 15% de matchs les plus récents (split temporel strict) :</p>
    <ul>
      <li><strong>MAE buts ≈ 1.05–1.10</strong> — erreur absolue moyenne d'environ 1 but.</li>
      <li><strong>Résultat correct ≈ 64%</strong> — home_win / draw / away_win.</li>
      <li><strong>Score exact ≈ 12%</strong> — difficile par nature, le foot reste imprévisible.</li>
    </ul>
    <p>Un modèle naïf (toujours 1–0) donne ~50% de résultats corrects.
    Le modèle apporte ~14 points de gain sur la prédiction du résultat.</p>

    </div>
    """, unsafe_allow_html=True)


# ── NAVIGATION ────────────────────────────────────────────────────────────────
def main():
    _, nav_col, _ = st.columns([5, 2, 1])
    with nav_col:
        page = st.radio("nav", ["⚽ Prédicteur", "📖 Comment ça marche ?"],
                        label_visibility="collapsed", horizontal=False, key="nav")

    if "marche" in page:
        page_explainer()
    else:
        page_predictor()


if __name__ == "__main__" or True:
    main()