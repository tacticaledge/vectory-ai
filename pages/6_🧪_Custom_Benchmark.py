"""Custom Benchmark Page - With Animations"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import init_session_state, EvalTaskType
from components.document_loader import load_file, load_files, infer_columns, ExtractionMode
from components.ui import (
    inject_custom_css,
    animated_metric,
    feature_card,
    display_lottie,
    score_bar,
    section_header,
    get_current_theme,
    sidebar_logo,
)
from components.mteb_leaderboard import get_available_models, AVAILABLE_MODELS_INFO

# Check if MTEB benchmarking is available (requires PyTorch >= 2.1)
MTEB_BENCHMARK_AVAILABLE = False
MTEB_ERROR = None
run_benchmark = None
get_mteb_tasks = None

def _check_torch_version():
    """Check if torch version is compatible (>= 2.1.0)."""
    try:
        import torch
        version = torch.__version__.split('+')[0]
        parts = version.split('.')
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor) >= (2, 1)
    except ImportError:
        return False
    except Exception:
        return False

# Try to import MTEB benchmark functions if PyTorch is available
if _check_torch_version():
    try:
        from components.mteb_evaluator import (
            run_benchmark,
            get_mteb_tasks,
        )
        MTEB_BENCHMARK_AVAILABLE = True
    except Exception as e:
        MTEB_ERROR = str(e)
else:
    MTEB_ERROR = "PyTorch >= 2.1.0 is required for running benchmarks"

# Get available models (doesn't require PyTorch)
AVAILABLE_MODELS = get_available_models()

st.set_page_config(page_title="Custom Benchmark | Vectory", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
init_session_state(st)
inject_custom_css()

# Animated header with developer colors
theme = get_current_theme()
st.markdown(f"""
<h1 style="font-family: 'DM Sans', sans-serif; background: linear-gradient(135deg, {theme.gradient_start}, {theme.gradient_mid}, {theme.gradient_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem;">
    🧪 Custom Benchmark
</h1>
<p style="font-family: 'DM Sans', sans-serif; color: {theme.text_muted}; font-size: 1rem; margin-top: -10px;">
    <span style="color: {theme.terminal};">$</span> run embedding evaluations using the MTEB framework
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Show info about benchmark availability
if not MTEB_BENCHMARK_AVAILABLE:
    st.info(f"""
    ℹ️ **Running benchmarks requires PyTorch >= 2.1.0**

    You can still browse models and upload datasets. To enable actual benchmark execution, run:
    ```
    pip install torch>=2.1.0
    ```
    """)

# Sidebar configuration
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 2rem;">⚙️</span>
        <h3 style="margin: 8px 0; color: {theme.text_primary};">Configuration</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding: 12px; background: {theme.bg_card}; border-radius: 8px; margin-bottom: 16px; border-left: 3px solid {theme.primary};">
        <h4 style="margin: 0 0 8px 0; color: {theme.primary}; font-size: 0.9rem;">📋 Task Types</h4>
    </div>
    """, unsafe_allow_html=True)

    task_types = st.multiselect(
        "Select tasks",
        [t.value for t in EvalTaskType],
        default=["Classification", "STS"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="padding: 12px; background: {theme.bg_card}; border-radius: 8px; margin-bottom: 16px; border-left: 3px solid {theme.warning};">
        <h4 style="margin: 0 0 8px 0; color: {theme.warning}; font-size: 0.9rem;">🤖 Models</h4>
    </div>
    """, unsafe_allow_html=True)

    model_names = [m["name"] for m in AVAILABLE_MODELS]
    selected_models = st.multiselect(
        "Select models",
        model_names,
        default=model_names[:2] if model_names else [],
        label_visibility="collapsed",
    )

    # Show model details
    if selected_models:
        for model_name in selected_models:
            model_info = next((m for m in AVAILABLE_MODELS if m["name"] == model_name), None)
            if model_info:
                st.caption(f"  {model_info['provider']} · {model_info['dimensions']}d")

# Tabs with custom styling
tab1, tab2, tab3 = st.tabs(["📁 Dataset", "⚡ Run", "📊 Results"])

# ==================== TAB 1: DATASET ====================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        section_header("📁 Upload Dataset", style="info", subtitle="Upload your evaluation data in JSON, JSONL, CSV, or PDF format")

    with col2:
        if not display_lottie("data", height=100):
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <span style="font-size: 3rem;">📂</span>
            </div>
            """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload evaluation file",
        type=["json", "jsonl", "csv", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            st.markdown('<div class="animated-progress" style="margin: 20px 0;"></div>', unsafe_allow_html=True)

            if len(uploaded) == 1:
                df = load_file(uploaded[0])
            else:
                df = load_files(uploaded)

            st.session_state.benchmark_dataset = df

            st.markdown(f"""
            <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
                <span style="font-size: 2rem;">✨</span>
                <h3 style="margin: 10px 0 5px 0; color: white;">Successfully Loaded!</h3>
                <p style="margin: 0; opacity: 0.9; color: white;">{len(df)} rows loaded</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Preview
            st.markdown(f"""
            <div class="feature-card" style="margin-bottom: 16px;">
                <h4 style="margin: 0; color: {theme.text_primary};">📋 Preview</h4>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

            # Column mapping
            section_header("🎯 Column Mapping", style="success")

            inferred = infer_columns(df)
            cols = [""] + list(df.columns)

            c1, c2 = st.columns(2)
            with c1:
                text_col = st.selectbox(
                    "Text Column",
                    cols,
                    index=cols.index(inferred.input_col) if inferred.input_col in cols else 0,
                )
            with c2:
                label_col = st.selectbox(
                    "Label Column (for classification)",
                    cols,
                    index=cols.index(inferred.expected_col) if inferred.expected_col in cols else 0,
                )

            st.session_state.benchmark_column_mapping = {
                "text": text_col or None,
                "label": label_col or None,
            }

        except Exception as e:
            error_str = str(e)
            # Check if this is a torch-related error
            if "torch" in error_str.lower() or "pytorch" in error_str.lower():
                st.error(f"""
                ❌ **PyTorch Compatibility Error**

                Your PyTorch version is incompatible. Please upgrade:
                ```
                pip install torch>=2.1.0
                ```

                This error occurred during file loading due to a transitive dependency.
                The file upload itself doesn't require PyTorch - you may need to restart the app after upgrading.

                Technical details: `{error_str}`
                """)
            else:
                st.error(f"❌ Error loading file: {e}")

    elif st.session_state.get("benchmark_dataset") is not None:
        st.markdown(f"""
        <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; border-left: 4px solid {theme.info};">
            <p style="margin: 0; color: {theme.info};">
                📊 Dataset loaded: <strong>{len(st.session_state.benchmark_dataset)}</strong> rows
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Dataset"):
            st.session_state.benchmark_dataset = None
            st.session_state.benchmark_results = None
            st.rerun()

    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if not display_lottie("search", height=150):
                st.markdown("""
                <div style="text-align: center; padding: 30px;">
                    <span style="font-size: 4rem; opacity: 0.5;">📂</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h4 style="color: {theme.text_primary};">No Data Yet</h4>
                <p style="color: {theme.text_muted};">Upload a file above to get started</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 2: RUN ====================
with tab2:
    section_header("⚡ Run Benchmark", style="primary")

    if st.session_state.get("benchmark_dataset") is None:
        st.markdown(f"""
        <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; border-left: 4px solid {theme.error};">
            <p style="margin: 0; color: {theme.error};">
                <strong>⚠️ No Dataset</strong><br>
                <span style="color: {theme.text_secondary};">Upload a dataset first in the Dataset tab.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif not selected_models:
        st.markdown(f"""
        <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; border-left: 4px solid {theme.warning};">
            <p style="margin: 0; color: {theme.warning};">
                <strong>⚠️ No Models Selected</strong><br>
                <span style="color: {theme.text_secondary};">Select at least one model in the sidebar.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif not task_types:
        st.markdown(f"""
        <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; border-left: 4px solid {theme.warning};">
            <p style="margin: 0; color: {theme.warning};">
                <strong>⚠️ No Tasks Selected</strong><br>
                <span style="color: {theme.text_secondary};">Select at least one task type in the sidebar.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.benchmark_dataset
        mapping = st.session_state.get("benchmark_column_mapping", {})

        # Info cards
        c1, c2, c3 = st.columns(3)
        with c1:
            animated_metric("Rows", str(len(df)), "📊", delay=1)
        with c2:
            animated_metric("Models", str(len(selected_models)), "🤖", delay=2)
        with c3:
            animated_metric("Tasks", str(len(task_types)), "📋", delay=3)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid {theme.warning};">
            <p style="margin: 0; color: {theme.warning};">
                <strong>💡 Note:</strong> <span style="color: {theme.text_secondary};">Running MTEB evaluation requires downloading models and datasets.
                This may take several minutes.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not MTEB_BENCHMARK_AVAILABLE:
            st.markdown(f"""
            <div style="padding: 16px; background: {theme.bg_card}; border-radius: 12px; border-left: 4px solid {theme.info};">
                <p style="margin: 0; color: {theme.info};">
                    <strong>ℹ️ Benchmark Execution Disabled</strong><br>
                    <span style="color: {theme.text_secondary};">
                        To run actual MTEB benchmarks, install PyTorch >= 2.1.0:
                        <code>pip install torch>=2.1.0</code>
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.button("🚀 Run Benchmark", type="primary", use_container_width=True, disabled=True)
        elif st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
            # Import EmbeddingModel for creating model objects
            from components.models import EmbeddingModel

            # Convert selected model dicts to EmbeddingModel objects
            models = []
            for model_name in selected_models:
                model_info = next((m for m in AVAILABLE_MODELS if m["name"] == model_name), None)
                if model_info:
                    # Get the model ID (HuggingFace format)
                    provider_prefix = {
                        "Sentence Transformers": "sentence-transformers",
                        "BAAI": "BAAI",
                        "Microsoft": "intfloat",
                        "Jina AI": "jinaai",
                        "Nomic AI": "nomic-ai",
                    }.get(model_info["provider"], model_info["provider"].lower().replace(" ", "-"))

                    model_id = f"{provider_prefix}/{model_info['name']}"

                    models.append(EmbeddingModel(
                        name=model_info["name"],
                        provider=model_info["provider"],
                        model_id=model_id,
                        dimensions=model_info["dimensions"],
                        max_tokens=model_info["max_tokens"],
                    ))

            tasks = [EvalTaskType(t) for t in task_types]

            st.markdown('<div class="animated-progress" style="margin: 20px 0;"></div>', unsafe_allow_html=True)
            progress = st.progress(0)
            status = st.empty()

            def update(p, msg):
                progress.progress(p)
                status.markdown(f"**{msg}**")

            try:
                results = run_benchmark(models, tasks, progress_callback=update)
                st.session_state.benchmark_results = results
                status.empty()

                st.markdown("""
                <div style="background: linear-gradient(-45deg, #6366f1, #8b5cf6, #a855f7, #4f46e5); background-size: 400% 400%; border-radius: 12px; padding: 30px; color: white; text-align: center;">
                    <span style="font-size: 2rem;">✨</span>
                    <h3 style="margin: 10px 0 5px 0; color: white;">Benchmark Complete!</h3>
                    <p style="margin: 0; opacity: 0.9; color: white;">Check the Results tab for details</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                error_str = str(e)
                if "torch" in error_str.lower() or "pytorch" in error_str.lower():
                    st.error(f"""
                    ❌ **PyTorch Compatibility Error**

                    MTEB benchmarking requires PyTorch >= 2.1.0. Please upgrade:
                    ```
                    pip install torch>=2.1.0
                    ```

                    Technical details: `{error_str}`
                    """)
                else:
                    st.error(f"❌ Error running benchmark: {e}")

# ==================== TAB 3: RESULTS ====================
with tab3:
    section_header("📊 Results", style="success")

    results = st.session_state.get("benchmark_results")

    if results is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if not display_lottie("chart", height=150):
                st.markdown("""
                <div style="text-align: center; padding: 30px;">
                    <span style="font-size: 4rem; opacity: 0.5;">📊</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h4 style="color: {theme.text_primary};">No Results Yet</h4>
                <p style="color: {theme.text_muted};">Run a benchmark first to see results</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Rankings
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom: 16px;">
            <h4 style="margin: 0; color: {theme.text_primary};">🏆 Model Rankings</h4>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            results.style.format({"mean_score": "{:.3f}"}).background_gradient(
                subset=["mean_score"], cmap="RdYlGn"
            ),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualization
        section_header("📈 Visualization", style="info")

        fig = px.bar(
            results.sort_values("mean_score"),
            x="mean_score",
            y="model",
            orientation="h",
            color="provider",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Mean Score",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Score bars for each model
        st.markdown(f"""
        <div class="feature-card" style="margin: 20px 0;">
            <h4 style="margin: 0 0 16px 0; color: {theme.text_primary};">📊 Score Comparison</h4>
        </div>
        """, unsafe_allow_html=True)

        for _, row in results.iterrows():
            score_bar(row["mean_score"], label=f"{row['model'][:20]} ({row['provider']})")

        # Export
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("💾 Export", style="primary")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 Download CSV",
                results.to_csv(index=False),
                "benchmark_results.csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "📋 Download JSON",
                results.to_json(orient="records", indent=2),
                "benchmark_results.json",
                use_container_width=True,
            )
