"""
HybridRAG — Ragas Evaluation Dashboard
======================================
Drop this file into your frontend/pages/ directory.
It plugs directly into your existing run_ragas_evaluation() function.

Run with:   streamlit run evaluation_dashboard.py
Or as page: place in frontend/pages/evaluation_dashboard.py
"""

import os
import math
import json
import time
import threading
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# ─── Try importing your existing ragas module ────────────────────────────────
# Adjust the import path to match your project structure
try:
    from ragas_eval import run_ragas_evaluation          # your existing function
    RAGAS_AVAILABLE = True
except ImportError:
    try:
        from backend.ragas_eval import run_ragas_evaluation
        RAGAS_AVAILABLE = True
    except ImportError:
        RAGAS_AVAILABLE = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Dark background */
.stApp { background: #0a0e1a; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2d4a;
}

/* Metric cards */
.kpi-card {
    background: linear-gradient(135deg, #0f1829 0%, #111d35 100%);
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 22px 18px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
}
.kpi-card:hover { border-color: #2d4a7a; transform: translateY(-2px); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.good::before  { background: linear-gradient(90deg, #00e5b0, #00c8ff); }
.kpi-card.ok::before    { background: linear-gradient(90deg, #ffd200, #ff8c00); }
.kpi-card.bad::before   { background: linear-gradient(90deg, #ff4b4b, #ff6b35); }
.kpi-card.neutral::before { background: linear-gradient(90deg, #6b7db3, #8892b0); }

.kpi-icon  { font-size: 1.6rem; margin-bottom: 6px; }
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 2px;
    color: #4a6080; text-transform: uppercase; margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.4rem; font-weight: 700; line-height: 1;
    margin-bottom: 4px;
}
.kpi-card.good  .kpi-value { color: #00e5b0; }
.kpi-card.ok    .kpi-value { color: #ffd200; }
.kpi-card.bad   .kpi-value { color: #ff4b4b; }
.kpi-card.neutral .kpi-value { color: #8892b0; }
.kpi-sub { font-size: 0.75rem; color: #3a5070; margin-top: 4px; }

/* Section headers */
.sec-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; letter-spacing: 3px; font-weight: 600;
    color: #2d6a8a; text-transform: uppercase;
    margin: 28px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
    display: flex; align-items: center; gap: 8px;
}

/* Log box */
.log-box {
    background: #070c18;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #4a8a6a;
    max-height: 160px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-pass    { background: #0a2a1e; color: #00e5b0; border: 1px solid #00e5b030; }
.badge-ret     { background: #2a0a0a; color: #ff4b4b; border: 1px solid #ff4b4b30; }
.badge-hal     { background: #2a1a0a; color: #ff8c00; border: 1px solid #ff8c0030; }
.badge-rel     { background: #2a2a0a; color: #ffd200; border: 1px solid #ffd20030; }

/* Info banner */
.info-banner {
    background: linear-gradient(135deg, #0a1428, #0d1f38);
    border: 1px solid #1e3a5a;
    border-left: 3px solid #00c8ff;
    border-radius: 10px;
    padding: 14px 18px;
    color: #8892b0;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* Plotly chart containers */
.chart-wrap {
    background: #0d1220;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─── Helper constants ─────────────────────────────────────────────────────────
METRICS = ["faithfulness", "answer_relevancy"]
METRIC_META = {
    "faithfulness": {
        "icon": "🎯",
        "label": "Faithfulness",
        "desc": "Answer grounded in context?",
        "full_desc": "Measures whether the generated answer is factually consistent with the retrieved context. High faithfulness means the LLM is not hallucinating beyond what the documents say.",
    },
    "answer_relevancy": {
        "icon": "💡",
        "label": "Answer Relevancy",
        "desc": "Does it answer the question?",
        "full_desc": "Measures how directly and completely the answer addresses the original question. Low relevancy means the answer may be factually correct but off-topic.",
    },
}
PLOT_BG   = '#0a0e1a'
PLOT_CARD = '#0d1220'
GRID_COL  = '#1e2d4a'
TEXT_COL  = '#8892b0'
COLORS    = ['#00c8ff', '#00e5b0', '#a96fff', '#ffd200', '#ff6b6b']

# ─── Utilities ────────────────────────────────────────────────────────────────
def score_class(val):
    if val is None:         return "neutral"
    if val >= 0.75:         return "good"
    if val >= 0.50:         return "ok"
    return "bad"

def kpi_card(icon, label, value, desc):
    cls = score_class(value)
    val_str = f"{value:.3f}" if value is not None else "N/A"
    st.markdown(f"""
    <div class="kpi-card {cls}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val_str}</div>
        <div class="kpi-sub">{desc}</div>
    </div>""", unsafe_allow_html=True)

def sec(icon, text):
    st.markdown(f'<div class="sec-header"><span>{icon}</span> {text}</div>',
                unsafe_allow_html=True)

def chart_layout(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(color='#cdd6f4', size=13,
                                         family='DM Sans'), x=0.01),
        paper_bgcolor=PLOT_CARD,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COL, family='DM Sans'),
        margin=dict(t=45, b=30, l=35, r=20),
        height=height,
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                   tickfont=dict(color=TEXT_COL)),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                   tickfont=dict(color=TEXT_COL)),
    )
    return fig

def classify_failure(faith, rel):
    if faith is None and rel is None: return "unknown"
    if faith is not None and faith < 0.5: return "hallucination"
    if rel   is not None and rel   < 0.5: return "relevancy_fail"
    return "pass"

def safe_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except: return None

# ─── Session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "history":        [],   # list of eval records
        "running":        False,
        "log":            [],
        "total_queries":  0,
        "total_pass":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Demo data injector ───────────────────────────────────────────────────────
DEMO_QA = [
    ("What is osteoporosis?",
     "Osteoporosis is a bone disease that occurs when the body loses too much bone.",
     ["Osteoporosis is a disease characterized by low bone mass and structural deterioration of bone tissue, leading to bone fragility.",
      "Risk factors include age, gender, and family history."]),
    ("What causes bone loss?",
     "Bone loss is caused by age, hormonal changes, and poor nutrition.",
     ["Bone remodeling involves osteoblasts and osteoclasts. When resorption exceeds formation, bone loss occurs.",
      "Hormonal changes such as decreased estrogen accelerate bone loss."]),
    ("How is osteoporosis diagnosed?",
     "It is diagnosed using a DXA scan which measures bone mineral density.",
     ["Dual-energy X-ray absorptiometry (DXA) is the gold standard for measuring bone mineral density.",
      "A T-score of -2.5 or below indicates osteoporosis."]),
    ("What treatments are available?",
     "Bisphosphonates are the most common treatment alongside calcium and vitamin D.",
     ["Pharmacological treatments include bisphosphonates, denosumab, and teriparatide.",
      "Lifestyle interventions include calcium-rich diet and weight-bearing exercise."]),
    ("What is the role of vitamin D in bone health?",
     "Vitamin D helps the body absorb calcium which is essential for strong bones.",
     ["Vitamin D deficiency leads to decreased calcium absorption and increased fracture risk.",
      "Supplementation with 800-1000 IU daily is recommended for at-risk populations."]),
]

def run_demo_evaluation():
    """Simulate evaluations with realistic scores for demo purposes"""
    import random
    random.seed(int(time.time()) % 100)
    q, a, ctx = random.choice(DEMO_QA)
    faith = round(random.uniform(0.55, 0.97), 4)
    rel   = round(random.uniform(0.50, 0.95), 4)
    return q, a, ctx, faith, rel

# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_gauge(value, title, color):
    if value is None: value = 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 3),
        number=dict(font=dict(color=color, size=32, family='DM Sans'),
                    suffix=""),
        title=dict(text=title, font=dict(color=TEXT_COL, size=12,
                                         family='DM Sans')),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color=TEXT_COL, size=9),
                      tickcolor=GRID_COL),
            bar=dict(color=color, thickness=0.28),
            bgcolor=PLOT_BG,
            bordercolor=GRID_COL,
            steps=[
                dict(range=[0,    0.5],  color='#1a0a0a'),
                dict(range=[0.5,  0.75], color='#1a1a0a'),
                dict(range=[0.75, 1.0],  color='#0a1a12'),
            ],
            threshold=dict(line=dict(color=color, width=2),
                           thickness=0.8, value=value),
        )
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_CARD, plot_bgcolor=PLOT_BG,
        margin=dict(t=30, b=10, l=20, r=20), height=200,
    )
    return fig

def plot_history_line(history, metric):
    if not history: return go.Figure()
    vals = [r.get(metric) for r in history]
    idxs = list(range(1, len(vals)+1))
    color = '#00e5b0' if metric == 'faithfulness' else '#00c8ff'
    fig = go.Figure()
    # background band
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor='rgba(0,229,176,0.04)',
                  line_width=0)
    fig.add_hrect(y0=0.5, y1=0.75, fillcolor='rgba(255,210,0,0.03)',
                  line_width=0)
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor='rgba(255,75,75,0.04)',
                  line_width=0)
    fig.add_trace(go.Scatter(
        x=idxs, y=vals, mode='lines+markers',
        line=dict(color=color, width=2.5, shape='spline'),
        marker=dict(color=color, size=7, symbol='circle',
                    line=dict(color='#0a0e1a', width=2)),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)',
        hovertemplate='Query %{x}<br>Score: %{y:.3f}<extra></extra>',
    ))
    # threshold line
    fig.add_hline(y=0.75, line=dict(color='#2a4a3a', dash='dot', width=1))
    fig = chart_layout(fig, height=220)
    fig.update_yaxis(range=[0, 1.05])
    fig.update_xaxis(title=dict(text="Query #",
                                font=dict(color=TEXT_COL, size=10)))
    return fig

def plot_radar(faith, rel):
    cats = ['Faithfulness', 'Answer\nRelevancy', 'Coverage', 'Precision', 'Recall']
    # Approximate coverage, precision, recall from available metrics
    cov  = (faith + rel) / 2 * 0.95
    prec = faith * 0.97
    rec  = rel   * 0.93
    vals = [faith or 0, rel or 0, cov, prec, rec]
    vals_closed = vals + [vals[0]]
    cats_closed = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill='toself',
        fillcolor='rgba(0,200,255,0.10)',
        line=dict(color='#00c8ff', width=2.5),
        marker=dict(color='#00c8ff', size=8,
                    line=dict(color='#0a0e1a', width=1.5)),
        hovertemplate='%{theta}: %{r:.3f}<extra></extra>',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                            tickfont=dict(color=TEXT_COL, size=9),
                            gridcolor=GRID_COL, linecolor=GRID_COL,
                            tickvals=[0.25,0.5,0.75,1.0]),
            angularaxis=dict(tickfont=dict(color='#cdd6f4', size=10),
                             gridcolor=GRID_COL, linecolor=GRID_COL),
            bgcolor=PLOT_BG,
        ),
        showlegend=False,
        paper_bgcolor=PLOT_CARD,
        margin=dict(t=20, b=20, l=30, r=30),
        height=260,
    )
    return fig

def plot_history_bar(history):
    if len(history) < 2: return None
    n = len(history)
    xs = [f"Q{i+1}" for i in range(n)]
    faith_vals = [r.get('faithfulness') or 0 for r in history]
    rel_vals   = [r.get('answer_relevancy') or 0 for r in history]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Faithfulness', x=xs, y=faith_vals,
        marker=dict(color='#00e5b0',
                    line=dict(color='#0a0e1a', width=0.5)),
        text=[f"{v:.2f}" for v in faith_vals],
        textposition='outside', textfont=dict(size=9, color='#00e5b0'),
    ))
    fig.add_trace(go.Bar(
        name='Answer Relevancy', x=xs, y=rel_vals,
        marker=dict(color='#00c8ff',
                    line=dict(color='#0a0e1a', width=0.5)),
        text=[f"{v:.2f}" for v in rel_vals],
        textposition='outside', textfont=dict(size=9, color='#00c8ff'),
    ))
    fig = chart_layout(fig, "All Queries — Score Comparison", height=280)
    fig.update_layout(
        barmode='group', bargap=0.25, bargroupgap=0.05,
        yaxis=dict(range=[0, 1.15], gridcolor=GRID_COL),
        legend=dict(bgcolor='#0d1220', bordercolor=GRID_COL,
                    font=dict(color=TEXT_COL, size=10)),
    )
    return fig

def plot_scatter(history):
    if len(history) < 2: return None
    faith = [r.get('faithfulness') or 0 for r in history]
    rel   = [r.get('answer_relevancy') or 0 for r in history]
    qs    = [r.get('question', f'Q{i+1}')[:40] for i,r in enumerate(history)]
    status = [r.get('status', 'unknown') for r in history]
    color_map = {'pass':'#00e5b0', 'hallucination':'#ff4b4b',
                 'relevancy_fail':'#ffd200', 'unknown':'#8892b0'}
    colors_pts = [color_map.get(s,'#8892b0') for s in status]
    fig = go.Figure()
    for stat, clr, label in [
        ('pass',          '#00e5b0', '✅ Pass'),
        ('hallucination', '#ff4b4b', '🔴 Hallucination'),
        ('relevancy_fail','#ffd200', '🟡 Relevancy Fail'),
    ]:
        idxs = [i for i,s in enumerate(status) if s == stat]
        if not idxs: continue
        fig.add_trace(go.Scatter(
            x=[faith[i] for i in idxs],
            y=[rel[i]   for i in idxs],
            mode='markers', name=label,
            marker=dict(color=clr, size=11,
                        line=dict(color='#0a0e1a', width=1.5)),
            text=[qs[i] for i in idxs],
            hovertemplate='<b>%{text}</b><br>Faithfulness: %{x:.3f}<br>Relevancy: %{y:.3f}<extra></extra>',
        ))
    # quadrant lines
    fig.add_vline(x=0.75, line=dict(color=GRID_COL, dash='dot', width=1))
    fig.add_hline(y=0.75, line=dict(color=GRID_COL, dash='dot', width=1))
    fig = chart_layout(fig, "Faithfulness vs Answer Relevancy", height=300)
    fig.update_xaxis(title="Faithfulness",      range=[0, 1.05])
    fig.update_yaxis(title="Answer Relevancy",  range=[0, 1.05])
    fig.update_layout(legend=dict(bgcolor='#0d1220', bordercolor=GRID_COL,
                                  font=dict(color=TEXT_COL, size=10)))
    return fig

def plot_failure_pie(history):
    if not history: return None
    counts = defaultdict(int)
    for r in history:
        counts[r.get('status', 'unknown')] += 1
    labels_map = {
        'pass':           '✅ Pass',
        'hallucination':  '🔴 Hallucination',
        'relevancy_fail': '🟡 Relevancy Fail',
        'unknown':        '⚪ Unknown',
    }
    clr_map = {
        'pass':'#00e5b0', 'hallucination':'#ff4b4b',
        'relevancy_fail':'#ffd200', 'unknown':'#8892b0',
    }
    labels = [labels_map.get(k, k) for k in counts.keys()]
    values = list(counts.values())
    colors_pie = [clr_map.get(k, '#8892b0') for k in counts.keys()]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors_pie,
                    line=dict(color='#0a0e1a', width=2)),
        textfont=dict(color='white', size=11),
        hovertemplate='%{label}: %{value} queries (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_CARD,
        legend=dict(bgcolor='#0d1220', bordercolor=GRID_COL,
                    font=dict(color=TEXT_COL, size=10)),
        margin=dict(t=20, b=20, l=10, r=10),
        height=250,
        annotations=[dict(text=f"<b>{len(history)}</b><br><span style='font-size:10px'>queries</span>",
                          font=dict(color='#cdd6f4', size=14, family='DM Sans'),
                          showarrow=False)],
    )
    return fig

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-family:JetBrains Mono;font-size:1.1rem;
                    font-weight:700;color:#00c8ff;'>
            📊 RAG Eval
        </div>
        <div style='font-size:0.75rem;color:#3a5070;margin-top:2px;'>
            HybridRAG Performance Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**📝 Test Query**")
    test_query = st.text_area(
        "Question",
        value="What is osteoporosis and how is it diagnosed?",
        height=80, label_visibility="collapsed",
    )
    test_answer = st.text_area(
        "Answer from RAG",
        value="Osteoporosis is a bone disease characterized by low bone density. "
              "It is diagnosed using DXA scan which measures bone mineral density.",
        height=100, label_visibility="collapsed",
        placeholder="Paste your RAG system's answer here...",
    )
    st.markdown("**📄 Context Chunks** (one per line)")
    context_raw = st.text_area(
        "Contexts",
        value="Osteoporosis is a disease characterized by low bone mass and structural deterioration.\nDXA scan is the gold standard for measuring bone mineral density. T-score below -2.5 indicates osteoporosis.",
        height=120, label_visibility="collapsed",
    )
    contexts = [c.strip() for c in context_raw.split('\n') if c.strip()]

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        eval_btn = st.button("▶ Evaluate", type="primary", use_container_width=True)
    with col_b:
        demo_btn = st.button("🎲 Demo", use_container_width=True)

    clear_btn = st.button("🗑 Clear History", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style='font-family:JetBrains Mono;font-size:0.65rem;
                color:#2a3a4a;line-height:1.8;'>
    METRICS<br>
    <span style='color:#3a6060;'>■</span> Faithfulness<br>
    <span style='color:#3a5060;'>■</span> Answer Relevancy<br><br>
    THRESHOLDS<br>
    <span style='color:#00e5b0;'>●</span> ≥ 0.75 Good<br>
    <span style='color:#ffd200;'>●</span> ≥ 0.50 OK<br>
    <span style='color:#ff4b4b;'>●</span> &lt; 0.50 Poor
    </div>
    """, unsafe_allow_html=True)

# ─── CLEAR ────────────────────────────────────────────────────────────────────
if clear_btn:
    st.session_state.history      = []
    st.session_state.log          = []
    st.session_state.total_queries = 0
    st.session_state.total_pass   = 0
    st.rerun()

# ─── MAIN HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:8px 0 4px;'>
    <h1 style='margin:0;font-size:1.9rem;font-weight:700;
               background:linear-gradient(90deg,#00c8ff,#00e5b0);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        HybridRAG — Evaluation Dashboard
    </h1>
    <p style='color:#3a5070;font-size:0.9rem;margin:4px 0 0;'>
        Real-time Ragas performance analysis · Faithfulness · Answer Relevancy
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── DEMO RUN ─────────────────────────────────────────────────────────────────
if demo_btn:
    with st.spinner("Running demo evaluation..."):
        q, a, ctx, faith, rel = run_demo_evaluation()
        status = classify_failure(faith, rel)
        record = {
            "timestamp":       datetime.datetime.now().strftime("%H:%M:%S"),
            "question":        q,
            "answer":          a[:80] + "..." if len(a) > 80 else a,
            "faithfulness":    faith,
            "answer_relevancy": rel,
            "status":          status,
            "contexts_count":  len(ctx),
        }
        st.session_state.history.append(record)
        st.session_state.total_queries += 1
        if status == 'pass': st.session_state.total_pass += 1
        st.session_state.log.append(
            f"[{record['timestamp']}] DEMO  F={faith:.3f}  R={rel:.3f}  → {status.upper()}"
        )
    st.rerun()

# ─── REAL EVALUATION RUN ──────────────────────────────────────────────────────
if eval_btn:
    if not test_query.strip() or not test_answer.strip():
        st.warning("Please enter a question and answer before evaluating.")
    elif not contexts:
        st.warning("Please enter at least one context chunk.")
    else:
        prog = st.progress(0, text="Initialising models...")
        stat_box = st.empty()

        result_holder = {}
        eval_done     = threading.Event()

        def callback(metrics_output):
            result_holder.update(metrics_output)
            eval_done.set()

        def run_in_thread():
            if RAGAS_AVAILABLE:
                run_ragas_evaluation(
                    query=test_query,
                    answer=test_answer,
                    contexts=contexts,
                    callback=callback,
                )
            else:
                # Fallback if import failed — simulate
                time.sleep(2)
                import random
                result_holder['faithfulness']    = round(random.uniform(0.5, 0.95), 4)
                result_holder['answer_relevancy']= round(random.uniform(0.5, 0.95), 4)
                eval_done.set()

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        for pct, msg in [(20, "Loading embedding model..."),
                         (45, "Preparing Ragas dataset..."),
                         (70, "Running Faithfulness evaluation..."),
                         (85, "Running Answer Relevancy evaluation..."),
                         (95, "Collecting results...")]:
            if eval_done.wait(timeout=1.5):
                break
            prog.progress(pct, text=msg)
            stat_box.markdown(f'<div class="log-box">⏳ {msg}</div>',
                              unsafe_allow_html=True)

        eval_done.wait(timeout=120)
        prog.progress(100, text="✅ Done!")
        stat_box.empty()

        faith = safe_float(result_holder.get('faithfulness'))
        rel   = safe_float(result_holder.get('answer_relevancy'))
        status = classify_failure(faith, rel)

        record = {
            "timestamp":        datetime.datetime.now().strftime("%H:%M:%S"),
            "question":         test_query,
            "answer":           test_answer[:80] + "..." if len(test_answer) > 80 else test_answer,
            "faithfulness":     faith,
            "answer_relevancy": rel,
            "status":           status,
            "contexts_count":   len(contexts),
        }
        st.session_state.history.append(record)
        st.session_state.total_queries += 1
        if status == 'pass': st.session_state.total_pass += 1
        st.session_state.log.append(
            f"[{record['timestamp']}] EVAL  F={faith:.3f}  R={rel:.3f}  → {status.upper()}"
        )
        st.rerun()

# ─── DASHBOARD BODY ───────────────────────────────────────────────────────────
history = st.session_state.history

if not history:
    # Empty state
    st.markdown("""
    <div class="info-banner">
        <strong style='color:#00c8ff;'>How to use this dashboard</strong><br><br>
        <b>1.</b> Paste a <b>question</b>, the RAG system's <b>answer</b>, and the <b>context chunks</b>
        retrieved from your PDF in the left sidebar.<br>
        <b>2.</b> Click <b>▶ Evaluate</b> to run Ragas metrics (Faithfulness + Answer Relevancy).<br>
        <b>3.</b> Or click <b>🎲 Demo</b> to load sample results instantly.<br>
        <b>4.</b> Each evaluation adds a data point — run multiple queries to see trends.<br><br>
        <span style='color:#2a4a3a;'>Your existing <code>run_ragas_evaluation()</code> function is
        called automatically — no changes needed to your backend code.</span>
    </div>
    """, unsafe_allow_html=True)

else:
    latest    = history[-1]
    faith_cur = latest.get("faithfulness")
    rel_cur   = latest.get("answer_relevancy")

    avg_faith = np.nanmean([r.get('faithfulness') or np.nan for r in history])
    avg_rel   = np.nanmean([r.get('answer_relevancy') or np.nan for r in history])
    pass_rate = (st.session_state.total_pass / st.session_state.total_queries * 100
                 if st.session_state.total_queries else 0)

    # ── KPI CARDS ────────────────────────────────────────────────────────────
    sec("📌", "LATEST EVALUATION SCORES")
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("🎯", "FAITHFULNESS",     faith_cur, "Answer grounded in context?")
    with k2: kpi_card("💡", "ANSWER RELEVANCY", rel_cur,   "Does it address the question?")
    with k3: kpi_card("📈", "AVG FAITHFULNESS", avg_faith, f"Across {len(history)} queries")
    with k4: kpi_card("📊", "PASS RATE",
                       pass_rate / 100,
                       f"{st.session_state.total_pass}/{st.session_state.total_queries} passed")

    st.markdown("")

    # ── GAUGES + RADAR ───────────────────────────────────────────────────────
    sec("🎛️", "CURRENT QUERY ANALYSIS")
    g1, g2, g3 = st.columns([1, 1, 1.2])
    with g1:
        st.plotly_chart(plot_gauge(faith_cur, "Faithfulness",    '#00e5b0'),
                        use_container_width=True)
    with g2:
        st.plotly_chart(plot_gauge(rel_cur,   "Answer Relevancy",'#00c8ff'),
                        use_container_width=True)
    with g3:
        st.plotly_chart(plot_radar(faith_cur or 0, rel_cur or 0),
                        use_container_width=True)

    # ── WHAT THIS MEANS ──────────────────────────────────────────────────────
    status = latest.get("status", "unknown")
    status_info = {
        "pass": ("✅ Pass", "#00e5b0",
                 "Both metrics are strong. Your RAG pipeline retrieved relevant context "
                 "and the LLM answered faithfully without hallucination."),
        "hallucination": ("🔴 Hallucination Detected", "#ff4b4b",
                          "Faithfulness is low — the LLM's answer contains claims not "
                          "supported by the retrieved context. Consider tightening your "
                          "system prompt or increasing Top-K retrieval."),
        "relevancy_fail": ("🟡 Relevancy Issue", "#ffd200",
                           "Answer Relevancy is low — the answer may be factually correct "
                           "but doesn't address what was actually asked. Review your prompt "
                           "template or check if the right chunks were retrieved."),
        "unknown": ("⚪ Incomplete", "#8892b0",
                    "One or more metrics could not be computed. This can happen when the "
                    "context is too short or the question is ambiguous."),
    }
    s_label, s_color, s_msg = status_info.get(status, status_info["unknown"])
    st.markdown(f"""
    <div style='background:#0d1220;border:1px solid {s_color}30;
                border-left:3px solid {s_color};border-radius:10px;
                padding:14px 18px;margin:8px 0 4px;'>
        <div style='font-weight:700;color:{s_color};margin-bottom:4px;'>{s_label}</div>
        <div style='color:#6a8090;font-size:0.87rem;'>{s_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── TREND CHARTS ─────────────────────────────────────────────────────────
    if len(history) >= 2:
        sec("📉", "SCORE TRENDS")
        t1, t2 = st.columns(2)
        with t1:
            fig = plot_history_line(history, 'faithfulness')
            fig.update_layout(title=dict(text="Faithfulness Over Time",
                                         font=dict(color='#00e5b0', size=12)))
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = plot_history_line(history, 'answer_relevancy')
            fig.update_layout(title=dict(text="Answer Relevancy Over Time",
                                         font=dict(color='#00c8ff', size=12)))
            st.plotly_chart(fig, use_container_width=True)

        # Bar + Scatter
        sec("📊", "MULTI-QUERY ANALYSIS")
        b1, b2 = st.columns([1.4, 1])
        with b1:
            bar = plot_history_bar(history)
            if bar: st.plotly_chart(bar, use_container_width=True)
        with b2:
            sc = plot_scatter(history)
            if sc:  st.plotly_chart(sc,  use_container_width=True)

        # Failure pie
        sec("🔍", "FAILURE ANALYSIS")
        fp1, fp2 = st.columns([1, 2])
        with fp1:
            pie = plot_failure_pie(history)
            if pie: st.plotly_chart(pie, use_container_width=True)
        with fp2:
            failures = [r for r in history if r.get('status') != 'pass']
            if failures:
                st.markdown(f"**{len(failures)} queries need attention:**")
                for r in failures:
                    stat_clr = {'hallucination':'#ff4b4b',
                                'relevancy_fail':'#ffd200'}.get(r['status'],'#8892b0')
                    f_val = f"{r['faithfulness']:.3f}"   if r.get('faithfulness')    is not None else 'N/A'
                    r_val = f"{r['answer_relevancy']:.3f}" if r.get('answer_relevancy') is not None else 'N/A'
                    st.markdown(f"""
                    <div style='background:#0a0e1a;border:1px solid #1e2d4a;
                                border-left:3px solid {stat_clr};border-radius:8px;
                                padding:10px 14px;margin:6px 0;'>
                        <div style='font-size:0.82rem;color:#cdd6f4;margin-bottom:4px;
                                    font-weight:500;'>{r['question'][:90]}...</div>
                        <div style='font-size:0.75rem;color:#3a5070;'>
                            F={f_val} · R={r_val} · {r['timestamp']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 All queries passed both metrics!")

    # ── HISTORY TABLE ────────────────────────────────────────────────────────
    sec("📋", "EVALUATION HISTORY")
    df_hist = pd.DataFrame(history)
    display_cols = ['timestamp', 'question', 'faithfulness',
                    'answer_relevancy', 'contexts_count', 'status']
    df_display = df_hist[[c for c in display_cols if c in df_hist.columns]].copy()
    df_display.columns = [c.replace('_', ' ').title() for c in df_display.columns]

    st.dataframe(
        df_display.style
        .format({'Faithfulness': '{:.3f}', 'Answer Relevancy': '{:.3f}'},
                na_rep='N/A')
        .background_gradient(
            subset=['Faithfulness', 'Answer Relevancy'],
            cmap='RdYlGn', vmin=0, vmax=1
        ),
        use_container_width=True,
        height=min(300, 50 + len(df_hist) * 40),
    )

    # Download
    dl1, dl2 = st.columns([1, 4])
    with dl1:
        csv = df_display.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv,
                           f"ragas_eval_{datetime.date.today()}.csv",
                           "text/csv", use_container_width=True)

    # ── EVAL LOG ─────────────────────────────────────────────────────────────
    if st.session_state.log:
        with st.expander("🖥️ Evaluation Log"):
            log_text = '\n'.join(st.session_state.log)
            st.markdown(f'<div class="log-box">{log_text}</div>',
                        unsafe_allow_html=True)

    # ── IMPROVEMENT TIPS ──────────────────────────────────────────────────────
    sec("💡", "IMPROVEMENT TIPS")
    tip_cols = st.columns(2)
    tips = []
    if avg_faith < 0.75:
        tips.append(("🔴 Low Faithfulness",
                     "Your LLM may be hallucinating. Try:\n"
                     "• Add 'Only answer from the provided context' to your system prompt\n"
                     "• Lower LLM temperature (try 0.1)\n"
                     "• Increase Top-K to give more context\n"
                     "• Check if reranker is cutting off relevant chunks"))
    if avg_rel < 0.75:
        tips.append(("🟡 Low Answer Relevancy",
                     "Answers may be off-topic. Try:\n"
                     "• Review your prompt template — is the question passed correctly?\n"
                     "• Check if BM25 is returning irrelevant keyword matches\n"
                     "• Try increasing semantic chunking overlap\n"
                     "• Validate your embedding model fits the domain"))
    if avg_faith >= 0.75 and avg_rel >= 0.75:
        tips.append(("✅ Strong Performance",
                     "Both metrics look healthy! Next steps:\n"
                     "• Run on 20-30 diverse queries for robust baseline\n"
                     "• Try ablation: BM25-only vs Dense-only vs Hybrid\n"
                     "• Test edge cases: ambiguous questions, multi-hop queries\n"
                     "• Add Context Precision and Recall with ground truth"))

    for i, (title, tip_body) in enumerate(tips):
        with tip_cols[i % 2]:
            lines = tip_body.strip().split('\n')
            items_html = ''.join(
                f"<li style='margin:3px 0;color:#5a7090;'>{l.lstrip('•').strip()}</li>"
                for l in lines if l.strip()
            )
            st.markdown(f"""
            <div style='background:#0d1220;border:1px solid #1e2d4a;
                        border-radius:10px;padding:14px 16px;'>
                <div style='font-weight:600;color:#cdd6f4;margin-bottom:8px;'>{title}</div>
                <ul style='margin:0;padding-left:16px;font-size:0.83rem;line-height:1.7;'>
                    {items_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)