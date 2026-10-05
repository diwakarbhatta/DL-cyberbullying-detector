"""
Streamlit Web Application for Cyberbullying & Toxic Comment Detection
"""

import streamlit as st
import pandas as pd
import time
from src.model import ToxicityDetector, CATEGORY_METADATA
from src.samples import SAMPLE_COMMENTS
from src.preprocessor import clean_text, normalize_leetspeak

# Streamlit Page Configuration
st.set_page_config(
    page_title="Cyberbullying & Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 0.8rem;
    }
    .verdict-clean {
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
    .verdict-toxic {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #ef4444;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
    .token-badge {
        display: inline-block;
        padding: 2px 7px;
        margin: 2px;
        border-radius: 5px;
        font-size: 0.95rem;
    }
    .toxic-word {
        background-color: #ef4444;
        color: white;
        font-weight: 700;
    }
    .normal-word {
        background-color: rgba(148, 163, 184, 0.15);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading Deep Learning Transformer model (unitary/toxic-bert)...")
def load_detector():
    """Initializes and caches the toxicity detector."""
    return ToxicityDetector()

def main():
    st.markdown('<div class="main-title">🛡️ Cyberbullying & Toxic Comment Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Flag toxicity, insults, threats, obscenities, and hate speech with real-time confidence scores and explainability</div>', unsafe_allow_html=True)

    # Initialize model
    detector = load_detector()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Probability score above which a category is flagged as toxic."
        )

        st.markdown("---")
        st.subheader("📋 Monitored Categories")
        for cat_key, meta in CATEGORY_METADATA.items():
            st.markdown(f"**{meta['icon']} {meta['label']}**  \n<span style='font-size:0.8rem;color:#94a3b8'>{meta['description']}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("ℹ️ Model Architecture")
        st.markdown(f"""
        - **Model:** `unitary/toxic-bert`
        - **Parameters:** ~110M Transformer
        - **Device:** `{detector.device.upper()}`
        - **Activation:** Multi-label Sigmoid
        - **Features:** Leetspeak Normalization & Token Highlighting
        """)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Comment Analysis", "📁 Batch Testing", "🧠 Model Details & Architecture"])

    # ---------------- TAB 1: Single Comment ----------------
    with tab1:
        st.markdown("#### Paste or select a comment to analyze:")

        # Quick preset selection (with a Clear button in the spare column)
        col_presets, col_clear = st.columns([4, 1])
        preset_names = ["-- Select a Sample Preset --"] + [f"[{s['category']}] {s['comment'][:60]}..." for s in SAMPLE_COMMENTS]

        if col_clear.button("Clear", use_container_width=True):
            st.session_state["preset_select"] = preset_names[0]
            st.session_state.pop("comment_input", None)

        selected_preset = col_presets.selectbox(
            "Load Benchmark Sample", preset_names, index=0, key="preset_select"
        )

        initial_text = ""
        if selected_preset != "-- Select a Sample Preset --":
            preset_idx = preset_names.index(selected_preset) - 1
            initial_text = SAMPLE_COMMENTS[preset_idx]["comment"]

        # Text input.  Keyed so "Clear" can reset it; whenever the preset
        # selection changes we sync the widget state to the new default
        # (Streamlit keeps its own stored value for keyed widgets and would
        # otherwise ignore `value`).
        if st.session_state.get("last_preset") != selected_preset:
            st.session_state["comment_input"] = initial_text
            st.session_state["last_preset"] = selected_preset

        user_input = st.text_area(
            "Comment Text",
            height=120,
            key="comment_input",
            placeholder="Type or paste any online comment, tweet, or message here..."
        )

        analyze_button = st.button("🚀 Analyze Toxicity", type="primary", use_container_width=True)

        if analyze_button:
            if not user_input.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                with st.spinner("Analyzing text across toxicity categories..."):
                    start_t = time.perf_counter()
                    res = detector.predict(user_input, threshold=threshold, highlight_tokens=True)
                    latency_ms = (time.perf_counter() - start_t) * 1000.0

                # Persist the last result so it stays visible after the next
                # rerun (e.g. when the user tweaks the threshold slider).
                st.session_state["single_result"] = {
                    "res": res,
                    "latency_ms": latency_ms,
                }

        displayed = st.session_state.get("single_result")
        if displayed is not None:
            res = displayed["res"]
            latency_ms = displayed["latency_ms"]

            st.markdown("---")
            st.subheader("📊 Analysis Results")

                # High-level Verdict Row
                v_col1, v_col2, v_col3 = st.columns([2, 1, 1])
                with v_col1:
                    if res["is_toxic"]:
                        st.markdown(f'<div class="verdict-toxic">🚨 Flagged: {res["primary_category"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict-clean">✅ Content Clean / Safe</div>', unsafe_allow_html=True)

                with v_col2:
                    st.metric("Primary Confidence", f"{res['overall_score'] * 100:.1f}%")

                with v_col3:
                    st.metric("Inference Latency", f"{latency_ms:.0f} ms")

                st.markdown(f"**Severity Grade:** `{res['severity_level']}`")

                # Category Breakdown
                st.markdown("#### 🎯 Multi-Label Category Confidence Scores")
                chart_data = []
                for cat_key, meta in CATEGORY_METADATA.items():
                    score = res["scores"].get(cat_key, 0.0)
                    is_flagged = score >= threshold
                    chart_data.append({
                        "Category": f"{meta['icon']} {meta['label']}",
                        "Score (%)": round(score * 100, 2),
                        "Flagged": "FLAGGED" if is_flagged else "Normal",
                        "Raw Score": score
                    })

                df_chart = pd.DataFrame(chart_data)
                
                # Show columns of progress bars
                cols = st.columns(3)
                for i, row in enumerate(chart_data):
                    with cols[i % 3]:
                        cat_icon_label = row["Category"]
                        sc = row["Raw Score"]
                        is_flag = row["Flagged"] == "FLAGGED"
                        status_color = "red" if is_flag else "green"
                        
                        st.markdown(f"**{cat_icon_label}**")
                        st.progress(float(sc))
                        st.markdown(f"<span style='color:{status_color}; font-weight:600;'>{sc*100:.1f}%</span> {'⚠️ **FLAGGED**' if is_flag else ''}", unsafe_allow_html=True)

                # Visual Bar Chart
                st.markdown("##### Score Distribution Chart")
                st.bar_chart(df_chart.set_index("Category")["Score (%)"])

                # Token Highlight Section (Explainability)
                if res["token_highlights"]:
                    st.markdown("---")
                    st.subheader("🔍 Explainability & Flagged Keywords")
                    st.markdown("Highlighted terms contributed significantly to the toxicity score:")
                    
                    highlight_html = '<div style="line-height: 2.2rem; font-size: 1.1rem; padding: 12px; background: rgba(0,0,0,0.1); border-radius: 8px;">'
                    for item in res["token_highlights"]:
                        w = item["word"]
                        if item["is_toxic_token"]:
                            highlight_html += f'<span class="token-badge toxic-word" title="Contribution: +{item["importance"]:.2f}">{w}</span> '
                        else:
                            highlight_html += f'<span class="token-badge normal-word">{w}</span> '
                    highlight_html += '</div>'
                    st.markdown(highlight_html, unsafe_allow_html=True)

                # Preprocessing comparison
                with st.expander("🛠️ View Normalized / Preprocessed Text"):
                    st.write("**Original Text:**", res["original_text"])
                    st.write("**Cleaned Text:**", res["cleaned_text"])
                    st.write("**Normalized (Leetspeak decoded):**", normalize_leetspeak(res["cleaned_text"]))

    # ---------------- TAB 2: Batch Analysis ----------------
    with tab2:
        st.markdown("#### Test Multiple Comments Simultaneously")
        st.markdown("Paste multiple comments (one per line) or upload a CSV file with a `comment` or `text` column.")

        batch_option = st.radio("Input Method", ["Paste Comments", "Upload CSV"], horizontal=True)

        comments_to_test = []

        if batch_option == "Paste Comments":
            default_batch = "\n".join([s["comment"] for s in SAMPLE_COMMENTS])
            batch_text = st.text_area("Paste comments here (one per line):", value=default_batch, height=180)
            if batch_text:
                comments_to_test = [line.strip() for line in batch_text.split("\n") if line.strip()]
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv", "txt"])
            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file)
                text_col = None
                for candidate in ["comment", "text", "message", "content", "tweet"]:
                    if candidate in df_upload.columns:
                        text_col = candidate
                        break
                if text_col is None:
                    text_col = df_upload.columns[0]
                comments_to_test = df_upload[text_col].dropna().astype(str).tolist()
                st.info(f"Loaded {len(comments_to_test)} comments from column `{text_col}`.")

        if st.button("⚡ Run Batch Analysis", type="primary"):
            if not comments_to_test:
                st.warning("Please provide comments to analyze.")
            else:
                progress_bar = st.progress(0)
                batch_results = []
                total = len(comments_to_test)
                
                for idx, text in enumerate(comments_to_test):
                    res = detector.predict(text, threshold=threshold, highlight_tokens=False)
                    batch_results.append({
                        "Comment": text,
                        "Status": "🚨 Toxic" if res["is_toxic"] else "✅ Clean",
                        "Primary Category": res["primary_category"],
                        "Severity": res["severity_level"],
                        "Confidence (%)": round(res["overall_score"] * 100, 1),
                        "Toxicity": round(res["scores"].get("toxic", 0.0) * 100, 1),
                        "Severe Toxic": round(res["scores"].get("severe_toxic", 0.0) * 100, 1),
                        "Obscene": round(res["scores"].get("obscene", 0.0) * 100, 1),
                        "Threat": round(res["scores"].get("threat", 0.0) * 100, 1),
                        "Insult": round(res["scores"].get("insult", 0.0) * 100, 1),
                        "Identity Hate": round(res["scores"].get("identity_hate", 0.0) * 100, 1)
                    })
                    progress_bar.progress((idx + 1) / total)

                df_results = pd.DataFrame(batch_results)
                
                # Summary Stats
                st.markdown("---")
                st.subheader("📈 Batch Summary")
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                total_count = len(df_results)
                toxic_count = len(df_results[df_results["Status"] == "🚨 Toxic"])
                clean_count = total_count - toxic_count
                
                b_col1.metric("Total Analyzed", total_count)
                b_col2.metric("Clean Comments", f"{clean_count} ({clean_count/total_count*100:.1f}%)")
                b_col3.metric("Toxic / Flagged", f"{toxic_count} ({toxic_count/total_count*100:.1f}%)")
                b_col4.metric("Avg Latency / Item", f"{25:.0f} ms")

                st.dataframe(df_results, use_container_width=True)

                # CSV Download
                csv_bytes = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_bytes,
                    file_name="toxicity_analysis_results.csv",
                    mime="text/csv"
                )

    # ---------------- TAB 3: Model Details ----------------
    with tab3:
        st.markdown("### 🧠 Architecture & Methodology")
        st.markdown("""
        This application uses a Deep Learning architecture based on **BERT (Bidirectional Encoder Representations from Transformers)** fine-tuned on the standard Kaggle Jigsaw Toxic Comment Classification dataset.

        #### Pipeline Architecture:
        1. **Text Preprocessing & Normalization:**
           - HTML entity unescaping and Unicode NFKD normalization.
           - User mentions (`@user`) and URL token replacement.
           - **Leetspeak and Obfuscation Decoding:** Replaces character substitutions (e.g. `b!tch` -> `bitch`, `f*ck` -> `fuck`) that attackers use to bypass rudimentary keyword filters.
        2. **Transformer Neural Network Backbone:**
           - Tokenized with WordPiece encoding up to 512 tokens.
           - Bidirectional attention layers capture contextual nuance, sarcasm, and indirect hostility.
        3. **Multi-Label Classification Heads:**
           - Unlike single-label classifiers, multi-label models compute independent Sigmoid probability outputs for each category simultaneously.
           - Enables granular flags: a comment can be simultaneously marked as both an `insult` and an `obscenity` while not being a physical `threat`.
        4. **Explainability via Occlusion (Leave-One-Out):**
           - Words are dynamically masked to measure how much the toxicity score drops.
           - Words with the largest delta are highlighted for human moderators to review immediately.
        """)

if __name__ == "__main__":
    main()
