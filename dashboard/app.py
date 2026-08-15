"""
Streamlit dashboard for the Legal RAG system.

Run from the project root:
    streamlit run dashboard/app.py

Query tab talks to the FastAPI service; Metrics and Results read local
files directly, so they work even when the API is down.
"""
import json
import os
import sqlite3

import altair as alt
import pandas as pd
import requests
import streamlit as st

DB_PATH = "results/query_logs.db"
ABLATION_CSV = "results/ablation_table.csv"

# Categorical slots 1-3 from the validated palette (all-pairs CVD safe).
# Fixed order, never cycled.
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
STAGE_LABELS = ["Retrieval", "Rerank", "Generation"]

st.set_page_config(page_title="Legal RAG", layout="wide")

def load_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM query_logs ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

def pct(series, p):
    s = series.dropna().sort_values()
    return None if s.empty else s.iloc[min(int(len(s) * p), len(s) - 1)]

def read_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


st.sidebar.title("Legal RAG")
api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
strategy = st.sidebar.selectbox("Chunking strategy",
                                ["semantic", "fixed", "recursive"], index=0)
use_reranker = st.sidebar.checkbox("Use reranker", value=True)

st.sidebar.caption("Winning config: semantic + bge-reranker-v2-m3")

try:
    h = requests.get(f"{api_url}/health", timeout=5).json()
    ok = h.get("qdrant") == "ok" and h.get("ollama") == "ok"
    if ok:
        st.sidebar.success("API healthy")
    else:
        st.sidebar.warning(f"API degraded: {h}")
except Exception:
    st.sidebar.error("API unreachable - Query tab disabled")

tab_ask, tab_metrics, tab_results = st.tabs(["Ask", "Metrics", "Results"])


with tab_ask:
    st.subheader("Ask a question")
    q = st.text_input("Question", placeholder="What did the Court hold regarding ...?")

    if st.button("Submit", type="primary") and q.strip():
        with st.spinner("Retrieving, reranking, generating..."):
            try:
                r = requests.post(f"{api_url}/query",
                                  json={"question": q, "strategy": strategy,
                                        "use_reranker": use_reranker},
                                  timeout=600)
            except Exception as e:
                st.error(f"Request failed: {e}")
                r = None

        if r is not None and r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.error(f"{r.status_code}: {detail}")
        elif r is not None:
            res = r.json()
            if res.get("cache_hit"):
                st.info("Served from semantic cache")
            if res.get("confidence") == "low":
                st.warning("Guardrail fired - insufficient information in sources")

            st.markdown(res["answer"])

            lat = res.get("latency", {})
            c = st.columns(5)
            c[0].metric("Total", f'{lat.get("total_s", 0):.2f}s')
            c[1].metric("Retrieval", f'{lat.get("retrieval_s", 0):.2f}s')
            c[2].metric("Rerank", f'{lat.get("rerank_s", 0):.2f}s')
            c[3].metric("Generation", f'{lat.get("generation_s", 0):.2f}s')
            c[4].metric("Tokens in/out",
                        f'{res["tokens"]["in"]}/{res["tokens"]["out"]}')

            if res.get("citations"):
                st.caption("Citations: " + " | ".join(map(str, res["citations"])))

            st.markdown("**Retrieved chunks**")
            for i, ch in enumerate(res.get("chunks_used", []), 1):
                score = ch.get("rerank_score")
                label = f'{i}. {ch.get("case_title", "Unknown")}'
                if score is not None:
                    label += f'  (rerank {score:.3f})'
                with st.expander(label):
                    st.caption(f'doc_id: {ch.get("doc_id")} | '
                               f'chunk_id: {ch.get("chunk_id")}')
                    st.write(ch.get("text_preview", ""))

with tab_metrics:
    st.subheader("Observability")
    df = load_logs()

    if df.empty:
        st.info("No queries logged yet. Ask something in the Ask tab.")
    else:
        p50, p95 = pct(df["total_s"], 0.50), pct(df["total_s"], 0.95)
        hits = int(df["cache_hit"].sum())

        c = st.columns(5)
        c[0].metric("Queries", len(df))
        c[1].metric("p50 latency", f"{p50:.2f}s" if p50 else "n/a")
        c[2].metric("p95 latency", f"{p95:.2f}s" if p95 else "n/a")
        c[3].metric("Cache hit rate", f"{hits / len(df):.1%}")
        c[4].metric("Tokens in", f'{int(df["tokens_in"].fillna(0).sum()):,}')

        st.markdown("**Average latency by stage** (cache misses only)")
        miss = df[df["cache_hit"] == 0]
        if miss.empty:
            st.caption("No cache-miss queries yet.")
        else:
            stage = pd.DataFrame({
                "Stage": STAGE_LABELS,
                "Seconds": [miss["retrieval_s"].mean(),
                            miss["rerank_s"].mean(),
                            miss["generation_s"].mean()],
            }).fillna(0)
            chart = (
                alt.Chart(stage)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Seconds:Q", title="Seconds"),
                    y=alt.Y("Stage:N", sort=STAGE_LABELS, title=None),
                    color=alt.Color("Stage:N",
                                    sort=STAGE_LABELS,
                                    scale=alt.Scale(domain=STAGE_LABELS,
                                                    range=STAGE_COLORS),
                                    legend=alt.Legend(title=None,
                                                      orient="bottom")),
                    tooltip=[alt.Tooltip("Stage:N"),
                             alt.Tooltip("Seconds:Q", format=".3f")],
                )
                .properties(height=190)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(stage.style.format({"Seconds": "{:.3f}"}),
                         hide_index=True, use_container_width=True)

        st.markdown("**Recent queries**")
        cols = ["time", "question", "strategy", "confidence", "total_s",
                "tokens_in", "tokens_out", "cache_hit"]
        st.dataframe(df[cols].head(25), hide_index=True,
                     use_container_width=True)


with tab_results:
    st.subheader("Evaluation results")

    if os.path.exists(ABLATION_CSV):
        st.markdown("**Axes 1 & 2 - chunking and reranking (n=250)**")
        st.dataframe(pd.read_csv(ABLATION_CSV), hide_index=True,
                     use_container_width=True)
    else:
        st.caption("ablation_table.csv not found")

    strict = read_json("results/ablation_prompt_strict.json")
    loose = read_json("results/ablation_prompt_loose.json")
    if strict and loose:
        st.markdown("**Axis 5 - strict vs loose prompt (n=100 per variant)**")
        rows = ["guard_decline_rate", "llm_decline_rate",
                "citation_presence_rate", "avg_tokens_in", "answered"]
        st.dataframe(pd.DataFrame({
            "Metric": rows,
            "Strict": [strict.get(k) for k in rows],
            "Loose": [loose.get(k) for k in rows],
        }), hide_index=True, use_container_width=True)

    st.markdown("**Guardrail is targeted, not indiscriminate**")
    st.dataframe(pd.DataFrame({
        "Group": ["Contested (strict refused)", "Matched control"],
        "Questions": [17, 17],
        "Claims": [141, 117],
        "Hallucination rate": ["30.5%", "16.2%"],
        "Per-question median": [0.333, 0.000],
    }), hide_index=True, use_container_width=True)
    st.caption("Mann-Whitney U, one-sided, per-question rates: p = 0.0084 "
               "vs matched control; p = 0.0027 vs the n=199 baseline.")
