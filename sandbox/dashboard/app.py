import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import duckdb
from sandbox.storage import get_store
from sandbox.utils import load_scenario
from sandbox.simulation import Simulation
from sandbox.predictive_model import FailurePredictor, MarkovChainAnalyzer
from sandbox.analytics.sentiment_tracker import SentimentTracker
from sandbox.analytics.report_generator import ReportGenerator
from sandbox.analytics.counterfactual_replay import CounterfactualReplayEngine
from sandbox.analytics.exporter import ConversationExporter
from sandbox.analytics.token_tracker import TokenTracker
from sandbox.analytics.complexity_analyzer import ComplexityAnalyzer

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwarmScope — Agent Safety Sandbox",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global Premium CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Reset & Base */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #050714 !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #1a0533 0%, #050714 40%), 
                radial-gradient(ellipse at 80% 100%, #0a1a3a 0%, #050714 50%) !important;
    background-attachment: fixed !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0a0f25 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
}
[data-testid="stSidebar"] .stSelectbox label, 
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] p {
    color: #e2e8f0 !important;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #1e0a3c 0%, #0f1f5c 40%, #0a2a2a 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -10%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(139,92,246,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50%;
    right: -10%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(6,182,212,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c084fc 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-purple { background: rgba(139,92,246,0.2); color: #c084fc; border: 1px solid rgba(139,92,246,0.4); }
.badge-cyan   { background: rgba(6,182,212,0.2);  color: #67e8f9; border: 1px solid rgba(6,182,212,0.4); }
.badge-emerald{ background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }
.badge-amber  { background: rgba(245,158,11,0.2); color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); }
.badge-rose   { background: rgba(244,63,94,0.2);  color: #fda4af; border: 1px solid rgba(244,63,94,0.4); }

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.mc-blue::before   { background: linear-gradient(90deg, #6366f1, #60a5fa); }
.mc-purple::before { background: linear-gradient(90deg, #8b5cf6, #c084fc); }
.mc-red::before    { background: linear-gradient(90deg, #ef4444, #f97316); }
.mc-green::before  { background: linear-gradient(90deg, #10b981, #34d399); }

.mc-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
    display: block;
}
.mc-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 4px;
}
.mc-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Section Titles */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(139,92,246,0.5), transparent);
    margin: 24px 0;
}

/* Detector Score Cards */
.detector-card {
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.03);
    position: relative;
    overflow: hidden;
}
.detector-name {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.detector-score {
    font-size: 1.9rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 6px;
}
.score-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 8px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}
.detector-expl {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.5;
}

/* Score colors */
.score-safe    { color: #34d399; }
.bar-safe      { background: linear-gradient(90deg, #10b981, #34d399); }
.border-safe   { border-left: 3px solid #10b981; }

.score-warn    { color: #fbbf24; }
.bar-warn      { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.border-warn   { border-left: 3px solid #f59e0b; }

.score-danger  { color: #f87171; }
.bar-danger    { background: linear-gradient(90deg, #ef4444, #f87171); }
.border-danger { border-left: 3px solid #ef4444; }

/* Risk Gauge */
.gauge-container {
    text-align: center;
    padding: 32px 24px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(10px);
}
.gauge-value {
    font-size: 4.5rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -2px;
    line-height: 1;
}
.gauge-label {
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 8px;
}
.gauge-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 12px;
}
.gauge-low    { color: #34d399; border-top: 4px solid #10b981; }
.gauge-medium { color: #fbbf24; border-top: 4px solid #f59e0b; }
.gauge-high   { color: #f87171; border-top: 4px solid #ef4444; }

/* Download buttons area */
.dl-section { margin-top: 8px; }

/* Sidebar nav */
.sidebar-header {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 20px;
    text-align: center;
}
.sidebar-header-text {
    font-size: 1.1rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
    letter-spacing: -0.3px;
}
.sidebar-header-sub {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.7);
    margin: 2px 0 0 0;
}

/* Chart containers */
.chart-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}

/* Info pill */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
}
.pill-blue   { background: rgba(99,102,241,0.2); color: #a5b4fc; }
.pill-purple { background: rgba(139,92,246,0.2); color: #c084fc; }
.pill-green  { background: rgba(16,185,129,0.2); color: #6ee7b7; }

/* Dataframe overrides */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* Spinner and success colors */
div[data-testid="stSpinner"] { color: #8b5cf6 !important; }

/* Headers */
h1, h2, h3, h4 { color: #e2e8f0 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(139,92,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(139,92,246,0.45) !important;
}

/* Selectbox / Slider label colors */
label { color: #94a3b8 !important; }

/* Progress bars */
.stProgress > div > div { background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important; }

/* Expander */
.streamlit-expanderHeader { color: #94a3b8 !important; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────────
def score_class(score: float):
    if score > 0.6:
        return "danger"
    elif score > 0.3:
        return "warn"
    return "safe"

def score_emoji(score: float):
    if score > 0.6:
        return "🔴"
    elif score > 0.3:
        return "🟡"
    return "🟢"

def detector_card_html(name: str, score: float, explanation: str) -> str:
    cls = score_class(score)
    pct = min(int(score * 100), 100)
    return f"""
    <div class="detector-card border-{cls}">
        <div class="detector-name" style="color: #94a3b8;">{score_emoji(score)} {name.upper().replace('_',' ')}</div>
        <div class="detector-score score-{cls}">{score:.2f}</div>
        <div class="score-bar-track">
            <div class="score-bar-fill bar-{cls}" style="width:{pct}%;"></div>
        </div>
        <div class="detector-expl">{explanation}</div>
    </div>
    """

def metric_card_html(icon: str, value: str, label: str, color_cls: str) -> str:
    return f"""
    <div class="metric-card {color_cls}">
        <span class="mc-icon">{icon}</span>
        <div class="mc-value">{value}</div>
        <div class="mc-label">{label}</div>
    </div>
    """

# ─── Database / Storage Init ────────────────────────────────────────────────────
db_path = "simulation_runs.duckdb"
store = get_store(db_path)

def check_and_seed_db():
    runs = store.get_all_runs()
    if not runs:
        scenarios = ["negotiation", "resource_allocation", "debate_consensus", "supply_chain", "hiring_panel", "crisis_response"]
        for sc in scenarios:
            path = f"scenarios/{sc}.yaml"
            if os.path.exists(path):
                cfg = load_scenario(path)
                Simulation(cfg, "dummy", 0.5, db_path).run()
                Simulation(cfg, "dummy", 1.0, db_path).run()

check_and_seed_db()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-header">
    <p class="sidebar-header-text">🛡️ SwarmScope</p>
    <p class="sidebar-header-sub">Agent Safety Sandbox v2.0</p>
</div>
""", unsafe_allow_html=True)

action_choice = st.sidebar.selectbox(
    "Navigate",
    ["📊 Dashboard Overview", "🔍 Run Explorer", "🎛️ Safety Debugger", "🔮 Risk Predictor"],
    label_visibility="collapsed"
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown("**System Info**")
st.sidebar.info("🟢 FastAPI server: `localhost:8000`\n\n🟢 Streamlit: `localhost:8501`")

# ─── Load Data ──────────────────────────────────────────────────────────────────
all_runs_meta = store.get_all_runs()
df_runs = pd.DataFrame([r.model_dump() for r in all_runs_meta]) if all_runs_meta else pd.DataFrame()

# ─── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🛡️ SwarmScope</p>
    <p class="hero-sub">Pre-deployment flight simulator for multi-agent LLM systems.<br/>
    Catch loops, deadlocks, collusion, goal drift, jailbreaks & information leakage <strong style="color:#c084fc">before</strong> they hit production.</p>
    <div class="badge-row">
        <span class="badge badge-purple">🔄 Loop Detection</span>
        <span class="badge badge-cyan">🔒 Deadlock Scanner</span>
        <span class="badge badge-emerald">🤝 Collusion Watch</span>
        <span class="badge badge-amber">📈 Escalation Monitor</span>
        <span class="badge badge-rose">🔓 Info Leakage</span>
        <span class="badge badge-purple">🎯 Goal Drift</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard Overview
# ═══════════════════════════════════════════════════════════════════════════════
if action_choice == "📊 Dashboard Overview":
    st.markdown('<div class="section-title">📊 Systems Reliability Overview</div>', unsafe_allow_html=True)

    if df_runs.empty:
        st.info("🚀 No simulation runs recorded yet. Start a simulation from the CLI or API first.")
    else:
        # ── KPI Cards ──────────────────────────────────────────────────────────
        conn = duckdb.connect(db_path)
        failures = conn.execute("""
            SELECT 
                mean(loop_score), mean(deadlock_score), mean(collusion_score), 
                mean(goal_drift_score), mean(jailbreak_score), mean(escalation_score),
                mean(information_leakage_score)
            FROM runs
        """).fetchone()
        conn.close()

        max_fail_rate = max(f for f in failures if f is not None) if failures else 0.0
        terminated_runs = len(df_runs[df_runs["status"] == "terminated"])
        avg_turns = df_runs["total_turns"].mean()
        n_scenarios = df_runs["scenario_name"].nunique()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(metric_card_html("🧪", str(len(df_runs)), "Total Simulated Runs", "mc-blue"), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card_html("🔀", str(n_scenarios), "Unique Scenarios Tested", "mc-purple"), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card_html("⚠️", f"{max_fail_rate:.1%}", "Peak Avg Failure Risk", "mc-red"), unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card_html("🛑", str(terminated_runs), "Early Terminated Runs", "mc-green"), unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Failure Rate Chart ─────────────────────────────────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="section-title">⚡ Vulnerability Rates by Failure Mode</div>', unsafe_allow_html=True)
            conn = duckdb.connect(db_path)
            df_scores = conn.execute("""
                SELECT loop_score, deadlock_score, collusion_score, goal_drift_score,
                       jailbreak_score, escalation_score, information_leakage_score FROM runs
            """).df()
            conn.close()

            if not df_scores.empty:
                fail_rates = {
                    "🔄 Loop":         (df_scores["loop_score"] > 0.5).mean(),
                    "🔒 Deadlock":     (df_scores["deadlock_score"] > 0.5).mean(),
                    "🤝 Collusion":    (df_scores["collusion_score"] > 0.5).mean(),
                    "🎯 Goal Drift":   (df_scores["goal_drift_score"] > 0.5).mean(),
                    "💣 Jailbreak":    (df_scores["jailbreak_score"] > 0.5).mean(),
                    "📈 Escalation":   (df_scores["escalation_score"] > 0.5).mean(),
                    "🔓 Info Leakage": (df_scores["information_leakage_score"] > 0.5).mean(),
                }
                df_fail = pd.DataFrame(list(fail_rates.items()), columns=["Failure Mode", "Rate"])
                st.bar_chart(df_fail, x="Failure Mode", y="Rate", color="#8b5cf6", height=280)

        with col_right:
            st.markdown('<div class="section-title">📍 Scenario Distribution</div>', unsafe_allow_html=True)
            if not df_runs.empty:
                sc_counts = df_runs["scenario_name"].value_counts().reset_index()
                sc_counts.columns = ["Scenario", "Runs"]
                st.bar_chart(sc_counts, x="Scenario", y="Runs", color="#06b6d4", height=280)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗂️ Recent Simulation Runs</div>', unsafe_allow_html=True)
        st.dataframe(
            df_runs[["simulation_id", "scenario_name", "timestamp", "total_turns", "backend", "temperature", "status"]],
            use_container_width=True,
            height=300
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Run Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif action_choice == "🔍 Run Explorer":
    st.markdown('<div class="section-title">🔍 Conversation Inspector & Analysis</div>', unsafe_allow_html=True)

    if df_runs.empty:
        st.info("No runs available yet.")
    else:
        run_ids = df_runs["simulation_id"].tolist()
        selected_id = st.selectbox("Select Simulation Run", run_ids)
        run = store.get_run(selected_id)

        # Run meta pills
        meta = run.metadata
        st.markdown(f"""
        <div style="margin-bottom:20px; display:flex; flex-wrap:wrap; gap:8px;">
            <span class="info-pill pill-blue">🆔 {meta.simulation_id[:16]}…</span>
            <span class="info-pill pill-purple">📁 {meta.scenario_name}</span>
            <span class="info-pill pill-blue">🤖 {meta.backend}</span>
            <span class="info-pill pill-green">🌡️ temp={meta.temperature}</span>
            <span class="info-pill pill-blue">🔄 {meta.total_turns} turns</span>
            <span class="info-pill {'pill-green' if meta.status=='completed' else 'pill-purple'}">
                {'✅' if meta.status=='completed' else '🛑'} {meta.status.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # ── Dialogue ──────────────────────────────────────────────────────
            st.markdown('<div class="section-title">💬 Dialogue Transcript</div>', unsafe_allow_html=True)
            for msg in run.messages:
                role = "user" if msg.turn % 2 == 0 else "assistant"
                st.chat_message(role).write(f"**{msg.sender}** · Turn {msg.turn}\n\n{msg.content}")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Sentiment Chart ───────────────────────────────────────────────
            st.markdown('<div class="section-title">📈 Sentiment Timeline</div>', unsafe_allow_html=True)
            st.caption("Scale: +1.0 = Cooperative, −1.0 = Hostile / Tense")
            tracker = SentimentTracker()
            sentiment_data = tracker.track_conversation_sentiment(run.messages)
            df_sent = pd.DataFrame(sentiment_data)
            if not df_sent.empty:
                st.line_chart(df_sent, x="turn", y="sentiment_score", color="#8b5cf6", height=220)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Token & Complexity ────────────────────────────────────────────
            st.markdown('<div class="section-title">🪙 Token Usage & Linguistic Complexity</div>', unsafe_allow_html=True)
            token_tracker = TokenTracker()
            token_stats = token_tracker.track_token_usage(run.messages)
            comp_analyzer = ComplexityAnalyzer()
            comp_stats = comp_analyzer.analyze_conversation(run.messages)

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"""
                <div class="detector-card" style="border-left: 3px solid #6366f1;">
                    <div class="detector-name" style="color:#818cf8;">💬 Token Statistics</div>
                    <div style="color:#e2e8f0; font-size:0.9rem; line-height:2;">
                        <b>Total Tokens:</b> <code style="color:#c084fc;">{token_stats.get('total_tokens', 0)}</code><br/>
                        <b>Avg / Turn:</b> <code style="color:#c084fc;">{token_stats.get('avg_tokens_per_turn', 0)}</code><br/>
                        <b>Gini Coeff:</b> <code style="color:#c084fc;">{token_stats.get('gini_coefficient', 0.0):.2f}</code>
                        <span class="info-pill pill-blue">{token_stats.get('fairness_label', '')}</span><br/>
                        <b>Dominant Speaker:</b> <code style="color:#f9a8d4;">{token_stats.get('dominant_speaker', 'N/A')}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with tc2:
                st.markdown("""
                <div class="detector-card" style="border-left: 3px solid #06b6d4;">
                    <div class="detector-name" style="color:#67e8f9;">📚 Linguistic Complexity</div>
                    <div style="color:#e2e8f0; font-size:0.9rem; line-height:2;">
                """, unsafe_allow_html=True)
                for ag, summ in comp_stats.get("agent_summaries", {}).items():
                    st.markdown(f"- **{ag}**: FK Grade `{summ['overall_fk_grade']}`, TTR `{summ['overall_ttr']}`")
                if comp_stats.get("has_degradation"):
                    st.warning("⚠️ Complexity degradation detected!")
                    for sig in comp_stats.get("degradation_signals", []):
                        st.write(f"  — {sig}")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Markov Chain ──────────────────────────────────────────────────
            st.markdown('<div class="section-title">⛓️ Markov State Transition Risk</div>', unsafe_allow_html=True)
            st.caption("Models conversation turns as state paths to compute absorption risk of deadlock / walkaways.")

            markov = MarkovChainAnalyzer()
            try:
                runs_list = []
                for meta_item in all_runs_meta:
                    try:
                        runs_list.append(store.get_run(meta_item.simulation_id))
                    except Exception:
                        pass
                markov.fit_from_runs(runs_list)
            except Exception:
                pass

            run_contents = [m.content for m in run.messages]
            risk_res = markov.predict_absorption_risk(run_contents, steps=5)

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"""
                <div class="detector-card" style="border-left:3px solid #8b5cf6; text-align:center;">
                    <div class="detector-name" style="color:#94a3b8;">Current State</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#c084fc;">{risk_res['starting_state']}</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                trisk = risk_res['termination_risk']
                tc = "f87171" if trisk > 0.5 else "fbbf24" if trisk > 0.3 else "34d399"
                st.markdown(f"""
                <div class="detector-card" style="border-left:3px solid #{tc}; text-align:center;">
                    <div class="detector-name" style="color:#94a3b8;">Termination Risk</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#{tc};">{trisk:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                cp = risk_res['cooperative_prob']
                st.markdown(f"""
                <div class="detector-card" style="border-left:3px solid #34d399; text-align:center;">
                    <div class="detector-name" style="color:#94a3b8;">Cooperative Prob</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#34d399;">{cp:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with mc4:
                tp = risk_res['tense_prob']
                st.markdown(f"""
                <div class="detector-card" style="border-left:3px solid #f87171; text-align:center;">
                    <div class="detector-name" style="color:#94a3b8;">Tense/Hostile Prob</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#f87171;">{tp:.1%}</div>
                </div>""", unsafe_allow_html=True)

            with st.expander("📐 Transition Probability Matrix"):
                matrix_df = pd.DataFrame(markov.get_matrix_dict())
                st.dataframe(matrix_df.style.format("{:.1%}"), use_container_width=True)

        with col_right:
            # ── Detector Scores ───────────────────────────────────────────────
            st.markdown('<div class="section-title">🛡️ Safety Scores</div>', unsafe_allow_html=True)
            for f_mode, score in run.detector_scores.items():
                expl = run.detector_explanations.get(f_mode, "")
                st.markdown(detector_card_html(f_mode, score, expl[:120] + "…" if len(expl) > 120 else expl), unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Interaction Graph ─────────────────────────────────────────────
            st.markdown('<div class="section-title">🕸️ Agent Interaction Graph</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('#0a0a1a')
            ax.set_facecolor('#0a0a1a')
            G = nx.DiGraph()
            for msg in run.messages:
                for r in [x.strip() for x in msg.receiver.split(",")]:
                    if G.has_edge(msg.sender, r):
                        G[msg.sender][r]['weight'] += 1
                    else:
                        G.add_edge(msg.sender, r, weight=1)

            pos = nx.circular_layout(G)
            max_risk = max(run.detector_scores.values()) if run.detector_scores else 0
            node_colors_map = ['#ef4444' if max_risk > 0.6 else '#f59e0b' if max_risk > 0.3 else '#10b981'] * len(G.nodes)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors_map, node_size=1200, ax=ax, alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_color='white', font_size=10, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='#8b5cf6', width=2.5, arrowsize=22,
                                   connectionstyle="arc3,rad=0.12", ax=ax, alpha=0.85)
            plt.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Downloads ─────────────────────────────────────────────────────
            st.markdown('<div class="section-title">📥 Export & Downloads</div>', unsafe_allow_html=True)
            markdown_report = ReportGenerator.generate_markdown_report(run)
            exporter = ConversationExporter()
            st.download_button("📄 Safety Report (Markdown)", markdown_report,
                               f"safety_report_{selected_id}.md", "text/markdown", use_container_width=True)
            st.download_button("📦 Export as JSON", exporter.to_json(run),
                               f"run_{selected_id}.json", "application/json", use_container_width=True)
            st.download_button("📊 Export as CSV", exporter.to_csv(run),
                               f"run_{selected_id}.csv", "text/csv", use_container_width=True)
            st.download_button("🗒️ Export as JSONL", exporter.to_jsonl(run),
                               f"run_{selected_id}.jsonl", "application/x-ndjson", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Safety Debugger
# ═══════════════════════════════════════════════════════════════════════════════
elif action_choice == "🎛️ Safety Debugger":
    st.markdown('<div class="section-title">🎛️ Counterfactual Replay Engine</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="detector-card" style="border-left: 3px solid #8b5cf6; margin-bottom:20px;">
        <div style="color:#c4b5fd; font-size:0.95rem; line-height:1.7;">
        When a simulation fails or exhibits high risk, SwarmScope replays the scenario under counterfactual settings —
        lower temperature, constrained memory, stricter prompts, and trait modifications — to find the most effective mitigation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_runs.empty:
        st.info("No runs available to debug.")
    else:
        run_ids = df_runs["simulation_id"].tolist()
        selected_id = st.selectbox("Select Simulation to Debug", run_ids)
        run = store.get_run(selected_id)
        max_orig_score = max(run.detector_scores.values()) if run.detector_scores else 0.0

        risk_cls = "danger" if max_orig_score > 0.6 else "warn" if max_orig_score > 0.3 else "safe"
        risk_color = "#f87171" if max_orig_score > 0.6 else "#fbbf24" if max_orig_score > 0.3 else "#34d399"
        st.markdown(f"""
        <div class="detector-card" style="border-left: 3px solid {risk_color};">
            <div class="detector-name" style="color:#94a3b8;">Selected Run</div>
            <div style="color:#e2e8f0; font-size:0.9rem; line-height:2;">
                🆔 <code style="color:#c084fc;">{selected_id[:20]}…</code> &nbsp;|&nbsp;
                📁 <b>{run.metadata.scenario_name}</b> &nbsp;|&nbsp;
                ⚠️ Peak Risk: <span style="color:{risk_color}; font-weight:700; font-size:1.1rem;">{max_orig_score:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔧 Execute Counterfactual Replay Diagnostics"):
            with st.spinner("Replaying with mitigations…"):
                engine = CounterfactualReplayEngine(db_path=db_path)
                report = engine.analyze_mitigations(selected_id)

            st.success("🔬 Analysis Complete!")
            st.markdown(f"""
            <div class="detector-card" style="border-left: 3px solid #10b981; margin: 16px 0;">
                <div class="detector-name" style="color:#34d399;">🎯 Recommendation</div>
                <div style="color:#e2e8f0; font-size:0.9rem; line-height:1.7;">{report['recommendation']}</div>
            </div>
            """, unsafe_allow_html=True)

            comparison_data = {"Original Run": max_orig_score}
            for var, res in report["results"].items():
                if "error" not in res:
                    comparison_data[var.replace("_", " ").capitalize()] = res["max_safety_score"]

            df_comp = pd.DataFrame(list(comparison_data.items()), columns=["Configuration", "Peak Safety Risk"])
            st.markdown('<div class="section-title">📉 Risk Comparison Across Counterfactual Replays</div>', unsafe_allow_html=True)
            st.bar_chart(df_comp, x="Configuration", y="Peak Safety Risk", color="#8b5cf6", height=300)
            st.dataframe(df_comp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Risk Predictor
# ═══════════════════════════════════════════════════════════════════════════════
elif action_choice == "🔮 Risk Predictor":
    st.markdown('<div class="section-title">🔮 Pre-Deployment Failure Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="detector-card" style="border-left: 3px solid #6366f1; margin-bottom: 24px;">
        <div style="color:#a5b4fc; font-size:0.95rem; line-height:1.7;">
        A <b>Random Forest classifier</b> trained on historical run metadata estimates the probability of failure modes 
        before execution — <b>no simulation required</b>. Configure your parameters and get instant risk assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-title">⚙️ Configure Parameters</div>', unsafe_allow_html=True)

        scenario = st.selectbox("🗺️ Scenario Type",
            ["negotiation", "resource_allocation", "debate_consensus", "supply_chain", "hiring_panel", "crisis_response"])
        backend = st.selectbox("🤖 LLM Backend", ["dummy", "ollama", "openai", "anthropic"])

        st.markdown("")
        temperature = st.slider("🌡️ Temperature Setting", 0.0, 1.2, 0.7, step=0.05)
        turns = st.slider("🔄 Max Conversation Turns", 5, 30, 15, step=1)

        st.markdown("")
        if st.button("⚡ Predict Failure Probability", use_container_width=True):
            predictor = FailurePredictor()
            prob = predictor.predict_probability(
                scenario_name=scenario, backend=backend,
                temperature=temperature, total_turns=turns
            )
            st.session_state.update({
                "pred_prob": prob, "pred_scenario": scenario,
                "pred_backend": backend, "pred_temp": temperature, "pred_turns": turns
            })

    with col2:
        st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)

        if "pred_prob" in st.session_state:
            prob = st.session_state["pred_prob"]

            if prob > 0.6:
                gauge_cls = "gauge-high"
                risk_label = "HIGH RISK"
                risk_emoji = "🔴"
            elif prob > 0.3:
                gauge_cls = "gauge-medium"
                risk_label = "MEDIUM RISK"
                risk_emoji = "🟡"
            else:
                gauge_cls = "gauge-low"
                risk_label = "LOW RISK"
                risk_emoji = "🟢"

            st.markdown(f"""
            <div class="gauge-container {gauge_cls}">
                <div style="font-size:0.85rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">
                    Predicted Failure Probability
                </div>
                <div class="gauge-value">{prob:.1%}</div>
                <div class="gauge-label">{risk_emoji} {risk_label}</div>
                <div class="gauge-sub">
                    Scenario: <b>{st.session_state['pred_scenario']}</b> &nbsp;|&nbsp;
                    Backend: <b>{st.session_state['pred_backend']}</b><br/>
                    Temp: <b>{st.session_state['pred_temp']}</b> &nbsp;|&nbsp;
                    Turns: <b>{st.session_state['pred_turns']}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            if prob > 0.6:
                st.error("⚠️ **High Risk — Recommendation:** High probability of loops, deadlocks, or collusion. Reduce temperature or lower turn limits before deployment.")
            elif prob > 0.3:
                st.warning("⚠️ **Medium Risk — Recommendation:** Moderate risk. Watch for collusion and semantic goal drift. Run small test batches first.")
            else:
                st.success("✅ **Low Risk — Recommendation:** The configuration represents a stable pathway. Safe to deploy with normal monitoring.")

            # Risk breakdown bar
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            bar_data = pd.DataFrame({
                "Risk Level": ["Safe Zone", "Medium Zone", "High Zone"],
                "Range":      [0.3, 0.3, 0.4]
            })
            st.progress(min(prob, 1.0))
            st.caption(f"Risk probability: {prob:.1%}")

        else:
            st.markdown("""
            <div class="gauge-container" style="border-top: 4px solid #334155;">
                <div style="font-size:3rem; margin-bottom: 16px;">🔮</div>
                <div style="color:#475569; font-size:1rem;">
                    Configure parameters on the left<br/>and click <b style="color:#8b5cf6;">Predict Failure Probability</b><br/>to see your risk assessment.
                </div>
            </div>
            """, unsafe_allow_html=True)
