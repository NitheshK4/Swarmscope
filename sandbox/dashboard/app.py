import streamlit as st
import pandas as pd
import numpy as np
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

# Page Configuration for modern dashboard aesthetic
st.set_page_config(
    page_title="Emergent Behavior Sandbox",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark styling tokens
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Database/Storage Init
db_path = "simulation_runs.duckdb"
store = get_store(db_path)

def check_and_seed_db():
    """Seeds the DB with dummy data if it's completely empty so the user is wowed immediately."""
    runs = store.get_all_runs()
    if not runs:
        # Let's run quick sample simulations to populate the DB
        scenarios = ["negotiation", "resource_allocation", "debate_consensus", "supply_chain", "hiring_panel", "crisis_response"]
        for sc in scenarios:
            path = f"scenarios/{sc}.yaml"
            if os.path.exists(path):
                cfg = load_scenario(path)
                # Run one normal simulation (temp 0.5)
                sim1 = Simulation(cfg, "dummy", 0.5, db_path)
                sim1.run()
                
                # Run one volatile simulation (temp 1.0)
                sim2 = Simulation(cfg, "dummy", 1.0, db_path)
                sim2.run()

check_and_seed_db()

st.title("🤖 Multi-Agent LLM: Emergent Behavior Sandbox")
st.caption("A safety sandbox monitoring multi-agent setups for Loops, Deadlocks, Collusion, Goal Drift, Jailbreak, Escalation, and Information Leakage.")

# Sidebar Controls
st.sidebar.title("Configuration & Control")
action_choice = st.sidebar.selectbox("Action", ["Dashboard Overview", "Explore Run Details", "Safety Debugger (Counterfactuals)", "Live Risk Predictor"])

# Load all historical metadata
all_runs_meta = store.get_all_runs()
df_runs = pd.DataFrame([r.model_dump() for r in all_runs_meta]) if all_runs_meta else pd.DataFrame()

if action_choice == "Dashboard Overview":
    st.header("📊 Systems Reliability Overview")
    
    if df_runs.empty:
        st.info("No simulation runs recorded yet. Start a simulation from the CLI or API first.")
    else:
        # High-level stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_runs)}</div><div class="metric-label">Total Simulated Runs</div></div>', unsafe_allow_html=True)
        with col2:
            avg_turns = df_runs["total_turns"].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_turns:.1f}</div><div class="metric-label">Avg. Dialogue Length (Turns)</div></div>', unsafe_allow_html=True)
        with col3:
            # Query DuckDB for average failure rates
            conn = duckdb.connect(db_path)
            failures = conn.execute("""
                SELECT 
                    mean(loop_score) as loop, 
                    mean(deadlock_score) as deadlock, 
                    mean(collusion_score) as collusion, 
                    mean(goal_drift_score) as drift,
                    mean(jailbreak_score) as jailbreak,
                    mean(escalation_score) as escalation,
                    mean(information_leakage_score) as leakage
                FROM runs
            """).fetchone()
            conn.close()
            
            max_fail_rate = max(failures) if failures and not any(f is None for f in failures) else 0.0
            st.markdown(f'<div class="metric-card"><div class="metric-value">{max_fail_rate:.1%}</div><div class="metric-label">Peak Failure Risk Score</div></div>', unsafe_allow_html=True)
        with col4:
            # Count terminated runs
            terminated_runs = len(df_runs[df_runs["status"] == "terminated"])
            st.markdown(f'<div class="metric-card"><div class="metric-value">{terminated_runs}</div><div class="metric-label">Early Terminated / Walkaways</div></div>', unsafe_allow_html=True)
            
        st.subheader("Vulnerability Rates by Failure Mode (Threshold > 0.5)")
        
        # Load scores
        conn = duckdb.connect(db_path)
        df_scores = conn.execute("SELECT loop_score, deadlock_score, collusion_score, goal_drift_score, jailbreak_score, escalation_score, information_leakage_score FROM runs").df()
        conn.close()
        
        if not df_scores.empty:
            fail_rates = {
                "Loop": (df_scores["loop_score"] > 0.5).mean(),
                "Deadlock": (df_scores["deadlock_score"] > 0.5).mean(),
                "Collusion": (df_scores["collusion_score"] > 0.5).mean(),
                "Goal Drift": (df_scores["goal_drift_score"] > 0.5).mean(),
                "Jailbreak": (df_scores["jailbreak_score"] > 0.5).mean(),
                "Escalation": (df_scores["escalation_score"] > 0.5).mean(),
                "Info Leakage": (df_scores["information_leakage_score"] > 0.5).mean()
            }
            
            df_fail_rates = pd.DataFrame(list(fail_rates.items()), columns=["Failure Mode", "Rate"])
            st.bar_chart(df_fail_rates, x="Failure Mode", y="Rate")
            
        st.subheader("Recent Runs History")
        st.dataframe(df_runs[["simulation_id", "scenario_name", "timestamp", "total_turns", "backend", "temperature", "status"]], use_container_width=True)

elif action_choice == "Explore Run Details":
    st.header("🔍 Conversation Inspector & Interaction Graph")
    
    if df_runs.empty:
        st.info("No runs available.")
    else:
        run_ids = df_runs["simulation_id"].tolist()
        selected_id = st.selectbox("Select Simulation ID", run_ids)
        
        # Load specific run
        run = store.get_run(selected_id)
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("Dialogue Logs")
            for msg in run.messages:
                st.chat_message("user" if msg.turn % 2 == 0 else "assistant").write(f"**{msg.sender}** (Turn {msg.turn}): {msg.content}")
                
            # Sentiment Timeline chart
            st.subheader("📈 Turn-by-Turn Sentiment Timeline")
            st.caption("Sentiment scale: +1.0 (Cooperative/Agreeable) to -1.0 (Hostile/Tense).")
            tracker = SentimentTracker()
            sentiment_data = tracker.track_conversation_sentiment(run.messages)
            df_sent = pd.DataFrame(sentiment_data)
            if not df_sent.empty:
                st.line_chart(df_sent, x="turn", y="sentiment_score")

            # Token and Complexity Details
            st.subheader("🪙 Token usage & Linguistic Complexity")
            
            token_tracker = TokenTracker()
            token_stats = token_tracker.track_token_usage(run.messages)
            
            comp_analyzer = ComplexityAnalyzer()
            comp_stats = comp_analyzer.analyze_conversation(run.messages)
            
            tc1, tc2 = st.columns(2)
            with tc1:
                st.write(f"**Total Tokens (est):** `{token_stats.get('total_tokens', 0)}` (Avg/Turn: `{token_stats.get('avg_tokens_per_turn', 0)}`)")
                st.write(f"**Speaker Dominance (Gini Coeff):** `{token_stats.get('gini_coefficient', 0.0):.2f}` (`{token_stats.get('fairness_label')}`)")
                st.write(f"**Dominant Speaker:** `{token_stats.get('dominant_speaker')}`")
            with tc2:
                st.write("**Linguistic Complexity (Flesch-Kincaid / TTR):**")
                for ag, summ in comp_stats.get("agent_summaries", {}).items():
                    st.write(f"- `{ag}`: FK Grade: `{summ['overall_fk_grade']}`, Vocab Richness (TTR): `{summ['overall_ttr']}`")
                if comp_stats.get("has_degradation"):
                    st.warning("⚠️ Complexity degradation detected!")
                    for sig in comp_stats.get("degradation_signals", []):
                        st.write(f"  - {sig}")
                
            # Markov State Projections
            st.write("---")
            st.subheader("⛓️ Conversation State Transition Risk (Markov Chain)")
            st.caption("Models conversation turns as state paths to compute absorption risk of deadlock or walkaways.")
            
            markov = MarkovChainAnalyzer()
            # Optionally fit from historical database runs if they exist
            try:
                runs_list = []
                for meta in all_runs_meta:
                    try:
                        runs_list.append(store.get_run(meta.simulation_id))
                    except Exception:
                        pass
                markov.fit_from_runs(runs_list)
            except Exception:
                pass
                
            run_contents = [m.content for m in run.messages]
            risk_res = markov.predict_absorption_risk(run_contents, steps=5)
            
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f"**Current State (End of Run):** `{risk_res['starting_state']}`")
                st.markdown(f"**Projected Termination Risk (next 5 turns):** `{risk_res['termination_risk']:.1%}`")
            with mc2:
                st.markdown(f"**Cooperative Probability:** `{risk_res['cooperative_prob']:.1%}`")
                st.markdown(f"**Tense/Hostile Probability:** `{risk_res['tense_prob']:.1%}`")
                
            # Expandable matrix visualization
            with st.expander("Show Conversation Transition Probability Matrix"):
                matrix_df = pd.DataFrame(markov.get_matrix_dict())
                st.dataframe(matrix_df.style.format("{:.1%}"), use_container_width=True)
                
        with col_right:
            st.subheader("Failure Analysis Scores")
            for f_mode, score in run.detector_scores.items():
                st.metric(label=f_mode.upper(), value=f"{score:.2f}")
                st.caption(run.detector_explanations.get(f_mode, ""))
                st.progress(min(max(float(score), 0.0), 1.0))
                st.write("---")
                
            # Download Safety Report & Exports
            st.subheader("📋 Actions")
            markdown_report = ReportGenerator.generate_markdown_report(run)
            st.download_button(
                label="📥 Download Safety Report (Markdown)",
                data=markdown_report,
                file_name=f"safety_report_{selected_id}.md",
                mime="text/markdown",
                use_container_width=True
            )
            
            exporter = ConversationExporter()
            json_export = exporter.to_json(run)
            st.download_button(
                label="📥 Export Run as JSON",
                data=json_export,
                file_name=f"run_{selected_id}.json",
                mime="application/json",
                use_container_width=True
            )
            
            csv_export = exporter.to_csv(run)
            st.download_button(
                label="📥 Export Run as CSV",
                data=csv_export,
                file_name=f"run_{selected_id}.csv",
                mime="text/csv",
                use_container_width=True
            )

            jsonl_export = exporter.to_jsonl(run)
            st.download_button(
                label="📥 Export Run as JSONL",
                data=jsonl_export,
                file_name=f"run_{selected_id}.jsonl",
                mime="application/x-ndjson",
                use_container_width=True
            )
            st.write("---")

            # Draw Interaction Graph
            st.subheader("Agent Communication Graph")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            G = nx.DiGraph()
            # Extract sender-receiver links
            for msg in run.messages:
                receivers = [r.strip() for r in msg.receiver.split(",")]
                for r in receivers:
                    if G.has_edge(msg.sender, r):
                        G[msg.sender][r]['weight'] += 1
                    else:
                        G.add_edge(msg.sender, r, weight=1)
            
            # Position nodes
            pos = nx.circular_layout(G)
            
            # Color node borders depending on overall risk
            max_risk = max(run.detector_scores.values())
            node_color = '#ef4444' if max_risk > 0.6 else '#f59e0b' if max_risk > 0.3 else '#10b981'
            
            nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=1000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_color='white', font_size=10, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='#94a3b8', width=2, arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax)
            
            plt.axis('off')
            st.pyplot(fig)

elif action_choice == "Safety Debugger (Counterfactuals)":
    st.header("🎛️ Counterfactual Replay Engine (Safety Debugger)")
    st.write("When a multi-agent simulation fails or exhibits high risk, SwarmScope can debug it by replaying the scenario under counterfactual settings (lower temperature, constrained memory, stricter prompts, traits modification, etc.) to discover the most effective mitigation strategy.")

    if df_runs.empty:
        st.info("No runs available to debug.")
    else:
        run_ids = df_runs["simulation_id"].tolist()
        selected_id = st.selectbox("Select Simulation to Debug", run_ids)
        
        # Load run
        run = store.get_run(selected_id)
        max_orig_score = max(run.detector_scores.values()) if run.detector_scores else 0.0
        
        st.write(f"**Original Simulation ID:** `{selected_id}` | **Scenario:** `{run.metadata.scenario_name}` | **Peak Original Risk Score:** `{max_orig_score:.2f}`")
        
        if st.button("🔧 Execute Counterfactual Replay Diagnostics"):
            with st.spinner("Replaying conversation with modifications..."):
                engine = CounterfactualReplayEngine(db_path=db_path)
                report = engine.analyze_mitigations(selected_id)
                
                # Show recommendation
                st.success("🔬 Analysis Complete!")
                st.info(report["recommendation"])
                
                # Compare scores in a bar chart
                comparison_data = {
                    "Original Run": max_orig_score
                }
                for var, res in report["results"].items():
                    if "error" not in res:
                        comparison_data[var.replace("_", " ").capitalize()] = res["max_safety_score"]
                
                df_comp = pd.DataFrame(list(comparison_data.items()), columns=["Configuration", "Peak Safety Risk"])
                st.subheader("Peak Risk Comparison Across Counterfactual Replays")
                st.bar_chart(df_comp, x="Configuration", y="Peak Safety Risk")
                
                st.subheader("Detailed Counterfactual Matrix")
                st.dataframe(df_comp, use_container_width=True)

elif action_choice == "Live Risk Predictor":
    st.header("🔮 Pre-Deployment Failure Risk Predictor")
    st.write("Leverages a Random Forest classifier trained on run metadata to estimate the probability of failure modes before execution.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenario = st.selectbox("Scenario Type", ["negotiation", "resource_allocation", "debate_consensus", "supply_chain", "hiring_panel", "crisis_response"])
        backend = st.selectbox("LLM Backend", ["dummy", "ollama", "openai", "anthropic"])
        temperature = st.slider("Temperature Setting", 0.0, 1.2, 0.7, step=0.05)
        turns = st.slider("Target Max Turns", 5, 30, 15, step=1)
        
        if st.button("Predict Failure Probability"):
            predictor = FailurePredictor()
            prob = predictor.predict_probability(
                scenario_name=scenario,
                backend=backend,
                temperature=temperature,
                total_turns=turns
            )
            
            st.session_state["pred_prob"] = prob
            st.session_state["pred_scenario"] = scenario
            st.session_state["pred_backend"] = backend
            st.session_state["pred_temp"] = temperature
            st.session_state["pred_turns"] = turns

    with col2:
        if "pred_prob" in st.session_state:
            prob = st.session_state["pred_prob"]
            st.subheader("Prediction Results")
            
            # Gauge display
            color = "red" if prob > 0.6 else "orange" if prob > 0.3 else "green"
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px; background-color: #1e293b;">
                    <div style="font-size: 1.2rem; color: #94a3b8;">Predicted Failure Probability</div>
                    <div style="font-size: 3.5rem; font-weight: bold; color: {color};">{prob:.1%}</div>
                    <div style="font-weight: bold; font-size: 1.5rem; text-transform: uppercase; color: {color};">
                        {"HIGH RISK" if prob > 0.6 else "MEDIUM RISK" if prob > 0.3 else "LOW RISK"}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Qualitative Risk explanation
            st.write("")
            st.write(f"**Scenario context:** Pre-deployment analysis for **{st.session_state['pred_scenario']}** using backend **{st.session_state['pred_backend']}**.")
            if prob > 0.6:
                st.error("⚠️ **Recommendation:** High probability of agent conversation entering endless loop cycles or stalling in deadlocks. Consider reducing temperature or lowering the target turn limit.")
            elif prob > 0.3:
                st.warning("⚠️ **Recommendation:** Moderate risk. Watch for collusion or semantic goals drift. Running minor test runs first is advised.")
            else:
                st.success("✅ **Recommendation:** The configuration represents a stable negotiation pathway. Safe to deploy.")
