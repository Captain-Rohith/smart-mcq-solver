import os
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set page config
st.set_page_config(
    page_title="Smart MCQ Solver & Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.25rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .top-answer {
        font-size: 2.5rem;
        font-weight: 800;
    }
    .option-card {
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6366F1;
        background-color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

OPTIONS = ['A', 'B', 'C', 'D', 'E']

# Sample questions for quick testing
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

# Safe tokenizer loader with fallback
def safe_load_tokenizer(model_source, fallback_base_id):
    for use_fast in [True, False]:
        try:
            return AutoTokenizer.from_pretrained(model_source, use_fast=use_fast)
        except Exception:
            pass
    # If custom repo missing tokenizer assets, use the official base tokenizer
    return AutoTokenizer.from_pretrained(fallback_base_id)

# Safe model loader with fallback
def safe_load_model(model_source, fallback_base_id, device):
    try:
        mod = AutoModelForSequenceClassification.from_pretrained(model_source).to(device)
    except Exception as e:
        mod = AutoModelForSequenceClassification.from_pretrained(fallback_base_id).to(device)
    mod.eval()
    return mod

# Cache model loading for fast inference
@st.cache_resource(show_spinner="Loading NLP models...")
def load_models(model_source_deb, model_source_rob):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DeBERTa
    tok_deb = safe_load_tokenizer(model_source_deb, "microsoft/deberta-v3-small")
    mod_deb = safe_load_model(model_source_deb, "microsoft/deberta-v3-small", device)
    
    # RoBERTa
    tok_rob = safe_load_tokenizer(model_source_rob, "roberta-base")
    mod_rob = safe_load_model(model_source_rob, "roberta-base", device)
    
    return tok_deb, mod_deb, tok_rob, mod_rob, device

def predict_single(prompt, a, b, c, d, e, tok_deb, mod_deb, tok_rob, mod_rob, device, mode, use_tta=True):
    choices_str = f" A: {a} B: {b} C: {c} D: {d} E: {e}"
    text_std = prompt + choices_str
    text_aug = "Answer the following multiple-choice question carefully: " + prompt + choices_str
    
    with torch.no_grad():
        p_deb = None
        p_rob = None
        
        if mode in ["Ensemble (70% DeBERTa + 30% RoBERTa)", "DeBERTa-v3-small Only"]:
            in_deb_std = {k: v.to(device) for k, v in tok_deb(text_std, return_tensors="pt", truncation=True, max_length=256).items()}
            logits_deb = mod_deb(**in_deb_std).logits
            p_deb = F.softmax(logits_deb, dim=-1)[0].cpu().numpy()
            
            if use_tta:
                in_deb_aug = {k: v.to(device) for k, v in tok_deb(text_aug, return_tensors="pt", truncation=True, max_length=256).items()}
                logits_deb_aug = mod_deb(**in_deb_aug).logits
                p_deb = (p_deb + F.softmax(logits_deb_aug, dim=-1)[0].cpu().numpy()) / 2.0
                
        if mode in ["Ensemble (70% DeBERTa + 30% RoBERTa)", "RoBERTa-base Only"]:
            in_rob_std = {k: v.to(device) for k, v in tok_rob(text_std, return_tensors="pt", truncation=True, max_length=256).items()}
            logits_rob = mod_rob(**in_rob_std).logits
            p_rob = F.softmax(logits_rob, dim=-1)[0].cpu().numpy()
            
            if use_tta:
                in_rob_aug = {k: v.to(device) for k, v in tok_rob(text_aug, return_tensors="pt", truncation=True, max_length=256).items()}
                logits_rob_aug = mod_rob(**in_rob_aug).logits
                p_rob = (p_rob + F.softmax(logits_rob_aug, dim=-1)[0].cpu().numpy()) / 2.0
                
    if mode == "Ensemble (70% DeBERTa + 30% RoBERTa)":
        probs = 0.70 * p_deb + 0.30 * p_rob
    elif mode == "DeBERTa-v3-small Only":
        probs = p_deb
    else:
        probs = p_rob
        
    return probs

# Sidebar settings
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=60)
    st.header("⚙️ Model Configuration")
    
    deb_path = st.text_input(
        "DeBERTa Model Path / HF Repo", 
        value="./deberta_small_nppe" if os.path.exists("./deberta_small_nppe") else "CaptainRohith/smart-mcq-deberta",
        help="Local directory path or Hugging Face repository ID (e.g. CaptainRohith/smart-mcq-deberta)"
    )
    
    rob_path = st.text_input(
        "RoBERTa Model Path / HF Repo", 
        value="./roberta_base_nppe" if os.path.exists("./roberta_base_nppe") else "CaptainRohith/smart-mcq-roberta",
        help="Local directory path or Hugging Face repository ID (e.g. CaptainRohith/smart-mcq-roberta)"
    )
    
    model_choice = st.selectbox(
        "Inference Strategy",
        ["Ensemble (70% DeBERTa + 30% RoBERTa)", "DeBERTa-v3-small Only", "RoBERTa-base Only"]
    )
    
    use_tta = st.checkbox("Enable Test-Time Augmentation (TTA)", value=True)
    
    st.divider()
    st.caption("🚀 Model: Fine-tuned Transformer for 5-Option Question Answering")

# Main Page Layout
st.markdown('<div class="main-title">🧠 Smart MCQ Solver AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning inference engine for multi-choice question answering and ranking</div>', unsafe_allow_html=True)

# Load Models
try:
    tok_deb, mod_deb, tok_rob, mod_rob, device = load_models(deb_path, rob_path)
    st.sidebar.success(f"✅ Models loaded on **{device.type.upper()}**")
except Exception as e:
    st.error(f"Error loading models from specified paths: {str(e)}")
    st.stop()

tab1, tab2 = st.tabs(["📝 Single Question Solver", "📁 Batch CSV Evaluation"])

# Tab 1: Single Prediction
with tab1:
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.subheader("Question Input")
        
        # Quick sample loader
        sample_choice = st.selectbox("Load an Example Question", ["Custom..."] + [f"Example {i+1}: {q['prompt'][:50]}..." for i, q in enumerate(SAMPLE_QUESTIONS)])
        
        default_prompt = ""
        default_opts = ["", "", "", "", ""]
        
        if sample_choice != "Custom...":
            idx = int(sample_choice.split(":")[0].replace("Example ", "")) - 1
            s = SAMPLE_QUESTIONS[idx]
            default_prompt = s["prompt"]
            default_opts = [s["A"], s["B"], s["C"], s["D"], s["E"]]
            
        prompt_input = st.text_area("Question Prompt", value=default_prompt, height=85, placeholder="e.g. What is the fundamental concept of...")
        
        opt_cols = st.columns(2)
        with opt_cols[0]:
            opt_a = st.text_input("Option A", value=default_opts[0], placeholder="Option A text")
            opt_b = st.text_input("Option B", value=default_opts[1], placeholder="Option B text")
            opt_c = st.text_input("Option C", value=default_opts[2], placeholder="Option C text")
        with opt_cols[1]:
            opt_d = st.text_input("Option D", value=default_opts[3], placeholder="Option D text")
            opt_e = st.text_input("Option E", value=default_opts[4], placeholder="Option E text")
            solve_btn = st.button("🔮 Solve Question", type="primary", use_container_width=True)
            
    with col2:
        st.subheader("Prediction & Ranking")
        if solve_btn:
            if not prompt_input.strip() or not (opt_a or opt_b or opt_c or opt_d or opt_e):
                st.warning("Please enter a question prompt and options.")
            else:
                with st.spinner("Analyzing semantics & calculating logits..."):
                    probs = predict_single(prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e, tok_deb, mod_deb, tok_rob, mod_rob, device, model_choice, use_tta)
                    
                    sorted_idx = np.argsort(-probs)
                    top_answer = OPTIONS[sorted_idx[0]]
                    top_3 = [OPTIONS[i] for i in sorted_idx[:3]]
                    
                    # Top Answer Banner
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Best Predicted Option</div>
                        <div class="top-answer">{top_answer}</div>
                        <div style="font-size: 1rem; opacity: 0.95;">Confidence: <b>{probs[sorted_idx[0]]*100:.1f}%</b> | Top 3 Ranking: <b>{' '.join(top_3)}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probabilities Breakdown
                    prob_df = pd.DataFrame({
                        "Option": OPTIONS,
                        "Confidence": probs,
                        "Probability (%)": [f"{p*100:.2f}%" for p in probs]
                    })
                    
                    st.bar_chart(prob_df.set_index("Option")["Confidence"], color="#6366F1")
                    st.dataframe(prob_df[["Option", "Probability (%)"]], use_container_width=True, hide_index=True)
        else:
            st.info("Enter a prompt and options on the left, then click **Solve Question**.")

# Tab 2: Batch CSV Evaluation
with tab2:
    st.subheader("Batch Inference on CSV")
    st.markdown("Upload a test dataset containing columns: `id`, `prompt`, `A`, `B`, `C`, `D`, `E` to generate `submission.csv`.")
    
    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])
    
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(test_df)}** rows. Preview:")
        st.dataframe(test_df.head(3), use_container_width=True)
        
        required_cols = {'id', 'prompt', 'A', 'B', 'C', 'D', 'E'}
        if not required_cols.issubset(set(test_df.columns)):
            st.error(f"CSV must contain all required columns: {required_cols}")
        else:
            if st.button("🚀 Run Batch Prediction", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
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
                        progress = (idx + 1) / len(test_df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {idx + 1}/{len(test_df)} questions ({progress*100:.0f}%)")
                        
                sub_df = pd.DataFrame({
                    'ID': test_df['id'],
                    'Prediction': predictions
                })
                
                st.success("🎉 Batch inference complete!")
                st.dataframe(sub_df.head(10), use_container_width=True)
                
                csv_bytes = sub_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download submission.csv",
                    data=csv_bytes,
                    file_name="submission.csv",
                    mime="text/csv",
                    type="primary"
                )
