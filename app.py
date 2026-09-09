import os
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page Configuration
st.set_page_config(
    page_title="Multiple Choice Question Classification Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Minimalist Styling
st.markdown("""
<style>
    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .app-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .app-title {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.92rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Result Cards */
    .result-container {
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        background: #f8fafc;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .result-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    
    .meta-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: #e2e8f0;
        color: #334155;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

OPTIONS = ['A', 'B', 'C', 'D', 'E']

SAMPLE_QUESTIONS = [
    {
        "prompt": "Pick the best possible answer: What is Martin Heidegger's view on human existence in time?",
        "A": "Martin Heidegger believes that humans exist with a static view of time.",
        "B": "Martin Heidegger believes that humans do not exist in time, but are rather temporal beings themselves.",
        "C": "Martin Heidegger does not believe in the existence of time at all.",
        "D": "Martin Heidegger believes that the relationship between time and existence is completely irrelevant.",
        "E": "Martin Heidegger believes that time is an illusion created by consciousness."
    },
    {
        "prompt": "Determine the correct option: What is the term for the shift of spectral lines toward longer wavelengths?",
        "A": "Blueshifting",
        "B": "Redshifting",
        "C": "Reddening",
        "D": "Whitening",
        "E": "Yellowing"
    }
]

def safe_load_tokenizer(model_source, fallback_base_id):
    for use_fast in [True, False]:
        try:
            return AutoTokenizer.from_pretrained(model_source, use_fast=use_fast)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(fallback_base_id)

@st.cache_resource(show_spinner="Loading model weights...")
def load_models(model_source_deb, model_source_rob):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DeBERTa
    tok_deb = safe_load_tokenizer(model_source_deb, "microsoft/deberta-v3-small")
    mod_deb = AutoModelForSequenceClassification.from_pretrained(model_source_deb).to(device)
    mod_deb.eval()
    
    # RoBERTa
    tok_rob = safe_load_tokenizer(model_source_rob, "roberta-base")
    mod_rob = AutoModelForSequenceClassification.from_pretrained(model_source_rob).to(device)
    mod_rob.eval()
    
    return tok_deb, mod_deb, tok_rob, mod_rob, device

def predict_single(prompt, a, b, c, d, e, tok_deb, mod_deb, tok_rob, mod_rob, device, mode, use_tta=True):
    choices_str = f" A: {a} B: {b} C: {c} D: {d} E: {e}"
    text_std = prompt + choices_str
    text_aug = "Answer the following multiple-choice question carefully: " + prompt + choices_str
    
    with torch.no_grad():
        p_deb = None
        p_rob = None
        
        if mode in ["Ensemble (0.70 DeBERTa + 0.30 RoBERTa)", "DeBERTa-v3-small"]:
            in_deb_std = {k: v.to(device) for k, v in tok_deb(text_std, return_tensors="pt", truncation=True, max_length=256).items()}
            logits_deb = mod_deb(**in_deb_std).logits
            p_deb = F.softmax(logits_deb, dim=-1)[0].cpu().numpy()
            
            if use_tta:
                in_deb_aug = {k: v.to(device) for k, v in tok_deb(text_aug, return_tensors="pt", truncation=True, max_length=256).items()}
                logits_deb_aug = mod_deb(**in_deb_aug).logits
                p_deb = (p_deb + F.softmax(logits_deb_aug, dim=-1)[0].cpu().numpy()) / 2.0
                
        if mode in ["Ensemble (0.70 DeBERTa + 0.30 RoBERTa)", "RoBERTa-base"]:
            in_rob_std = {k: v.to(device) for k, v in tok_rob(text_std, return_tensors="pt", truncation=True, max_length=256).items()}
            logits_rob = mod_rob(**in_rob_std).logits
            p_rob = F.softmax(logits_rob, dim=-1)[0].cpu().numpy()
            
            if use_tta:
                in_rob_aug = {k: v.to(device) for k, v in tok_rob(text_aug, return_tensors="pt", truncation=True, max_length=256).items()}
                logits_rob_aug = mod_rob(**in_rob_aug).logits
                p_rob = (p_rob + F.softmax(logits_rob_aug, dim=-1)[0].cpu().numpy()) / 2.0
                
    if mode == "Ensemble (0.70 DeBERTa + 0.30 RoBERTa)":
        if p_deb is not None and p_rob is not None and len(p_deb) == len(p_rob) == 5:
            probs = 0.70 * p_deb + 0.30 * p_rob
        elif p_deb is not None and len(p_deb) == 5:
            probs = p_deb
        elif p_rob is not None and len(p_rob) == 5:
            probs = p_rob
        else:
            probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    elif mode == "DeBERTa-v3-small":
        probs = p_deb if (p_deb is not None and len(p_deb) == 5) else np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    else:
        probs = p_rob if (p_rob is not None and len(p_rob) == 5) else np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
    return probs

# Sidebar Configuration
with st.sidebar:
    st.subheader("Configuration")
    
    deb_path = st.text_input(
        "DeBERTa Repository", 
        value="./deberta_small_nppe" if os.path.exists("./deberta_small_nppe") else "CaptainRohith/smart-mcq-deberta"
    )
    
    rob_path = st.text_input(
        "RoBERTa Repository", 
        value="./roberta_base_nppe" if os.path.exists("./roberta_base_nppe") else "CaptainRohith/smart-mcq-roberta"
    )
    
    model_choice = st.selectbox(
        "Architecture",
        ["Ensemble (0.70 DeBERTa + 0.30 RoBERTa)", "DeBERTa-v3-small", "RoBERTa-base"]
    )
    
    use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=True)
    
    if st.button("Reload Models / Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
        
    st.divider()
    st.caption("Classification Head: SequenceClassification (5 classes)")

# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">Multiple Choice Question Classification Engine</div>
    <div class="app-subtitle">Sequence classification benchmark with ensemble weighting and test-time augmentation</div>
</div>
""", unsafe_allow_html=True)

# Model Initialization
try:
    tok_deb, mod_deb, tok_rob, mod_rob, device = load_models(deb_path, rob_path)
    st.sidebar.caption(f"Compute Device: {device.type.upper()}")
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

tab_single, tab_batch = st.tabs(["Single Evaluation", "Batch Evaluation"])

# Tab 1: Single Question Analysis
with tab_single:
    col_input, col_output = st.columns([1.1, 0.9])
    
    with col_input:
        st.markdown("##### Input Specifications")
        
        sample_choice = st.selectbox(
            "Benchmark Examples", 
            ["Custom Input"] + [f"Reference {i+1}: {q['prompt'][:50]}..." for i, q in enumerate(SAMPLE_QUESTIONS)]
        )
        
        default_prompt = ""
        default_opts = ["", "", "", "", ""]
        
        if sample_choice != "Custom Input":
            idx = int(sample_choice.split(":")[0].replace("Reference ", "")) - 1
            s = SAMPLE_QUESTIONS[idx]
            default_prompt = s["prompt"]
            default_opts = [s["A"], s["B"], s["C"], s["D"], s["E"]]
            
        prompt_input = st.text_area("Prompt", value=default_prompt, height=85, placeholder="Enter premise or question context...")
        
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            opt_a = st.text_input("Option A", value=default_opts[0])
            opt_b = st.text_input("Option B", value=default_opts[1])
            opt_c = st.text_input("Option C", value=default_opts[2])
        with col_o2:
            opt_d = st.text_input("Option D", value=default_opts[3])
            opt_e = st.text_input("Option E", value=default_opts[4])
            st.write("")
            run_btn = st.button("Run Evaluation", type="primary", use_container_width=True)
            
    with col_output:
        st.markdown("##### Output & Probabilities")
        if run_btn:
            if not prompt_input.strip() or not (opt_a or opt_b or opt_c or opt_d or opt_e):
                st.warning("Prompt and options cannot be empty.")
            else:
                probs = predict_single(
                    prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e,
                    tok_deb, mod_deb, tok_rob, mod_rob, device, model_choice, use_tta
                )
                
                sorted_idx = np.argsort(-probs)
                top_class = OPTIONS[sorted_idx[0]]
                top_3 = [OPTIONS[i] for i in sorted_idx[:3]]
                top_confidence = probs[sorted_idx[0]] * 100
                
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-label">Predicted Class</div>
                    <div class="result-value">{top_class}</div>
                    <div style="margin-top: 0.5rem;">
                        <span class="meta-tag">Confidence: {top_confidence:.2f}%</span>
                        <span class="meta-tag">Top-3 Ranking: {' '.join(top_3)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                prob_df = pd.DataFrame({
                    "Option": OPTIONS,
                    "Probability": probs,
                    "Score (%)": [f"{p*100:.2f}%" for p in probs]
                })
                
                st.bar_chart(prob_df.set_index("Option")["Probability"], height=200)
                st.dataframe(prob_df[["Option", "Score (%)"]], use_container_width=True, hide_index=True)
        else:
            st.info("Awaiting input execution. Populate parameters and select 'Run Evaluation'.")

# Tab 2: Batch CSV Evaluation
with tab_batch:
    st.markdown("##### Dataset Evaluation")
    st.caption("Upload a formatted dataset with schema: id, prompt, A, B, C, D, E")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.dataframe(test_df.head(5), use_container_width=True)
        
        required_cols = {'id', 'prompt', 'A', 'B', 'C', 'D', 'E'}
        if not required_cols.issubset(set(test_df.columns)):
            st.error(f"Missing schema attributes. Expected: {required_cols}")
        else:
            if st.button("Process Dataset", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                predictions = []
                
                for idx, row in test_df.iterrows():
                    probs = predict_single(
                        str(row['prompt']), str(row['A']), str(row['B']), str(row['C']), str(row['D']), str(row['E']),
                        tok_deb, mod_deb, tok_rob, mod_rob, device, model_choice, use_tta
                    )
                    sorted_idx = np.argsort(-probs)
                    top_3 = [OPTIONS[i] for i in sorted_idx[:3]]
                    predictions.append(" ".join(top_3))
                    
                    if (idx + 1) % max(1, len(test_df) // 20) == 0 or idx == len(test_df) - 1:
                        pct = (idx + 1) / len(test_df)
                        progress.progress(pct)
                        status.text(f"Evaluated {idx + 1} of {len(test_df)} instances")
                        
                sub_df = pd.DataFrame({
                    'ID': test_df['id'],
                    'Prediction': predictions
                })
                
                st.markdown("##### Results Preview")
                st.dataframe(sub_df.head(10), use_container_width=True)
                
                csv_bytes = sub_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Export Results (CSV)",
                    data=csv_bytes,
                    file_name="submission.csv",
                    mime="text/csv",
                    type="primary"
                )
