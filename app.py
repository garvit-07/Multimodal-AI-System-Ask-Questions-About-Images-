"""
app.py — Streamlit VQA Demo App

Supports two modes:
  1. TRAINED MODEL mode  → uses your checkpoint from checkpoints/best_model.pth
  2. DEMO mode (CLIP)    → uses OpenAI CLIP zero-shot, works right away without training

Run:
    streamlit run app.py
"""

import os
import sys
import io
import time
import torch
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "VQA — Visual Question Answering",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .answer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: white;
        margin: 1rem 0;
    }
    .answer-main {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .answer-conf {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }
    .top-k-row {
        display: flex;
        align-items: center;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    .badge {
        background: #f0f4ff;
        border: 1px solid #c3d1ff;
        border-radius: 8px;
        padding: 0.15rem 0.5rem;
        font-size: 0.82rem;
        color: #3355cc;
        margin-right: 0.5rem;
        min-width: 2.5rem;
        text-align: center;
    }
    .mode-info {
        background: #fffbe6;
        border-left: 4px solid #f5a623;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div { border-radius: 4px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading (cached) ────────────────────────────────────────────────

CHECKPOINT_PATH = "checkpoints/best_model.pth"

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load the custom VQA model from checkpoint."""
    from models.vqa_model import VQAModel
    from utils.tokenizer  import VQATokenizer
    from utils.helpers    import load_answer_vocab

    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
    ans_path = os.path.join(checkpoint_dir, "answer_vocab.json")

    if not all(os.path.exists(p) for p in [CHECKPOINT_PATH, tok_path, ans_path]):
        return None, None, None, None

    tokenizer = VQATokenizer.load(tok_path)
    answer2idx, idx2answer = load_answer_vocab(ans_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(CHECKPOINT_PATH, map_location=device)
    saved  = ckpt.get("args", {})

    model = VQAModel(
        vocab_size  = tokenizer.vocab_size,
        num_answers = len(answer2idx),
        embed_dim   = saved.get("embed_dim", 512),
        use_bert    = saved.get("use_bert", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, idx2answer, device


@st.cache_resource(show_spinner=False)
def load_clip_model():
    """Load CLIP for zero-shot demo mode (no training needed)."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()
        return model, processor, device
    except Exception as e:
        return None, None, None


# ── Inference functions ───────────────────────────────────────────────────

def predict_trained(image: Image.Image, question: str, model, tokenizer, idx2answer, device, top_k=5):
    from utils.dataset import get_transform
    transform  = get_transform("val")
    img_tensor = transform(image).unsqueeze(0).to(device)
    q_ids      = torch.tensor([tokenizer.encode(question)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(img_tensor, q_ids)
        probs  = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(top_k, dim=-1)

    return [(idx2answer[top_ids[0, i].item()], top_probs[0, i].item()) for i in range(top_k)]


def predict_clip_demo(image: Image.Image, question: str, clip_model, processor, device, top_k=5):
    """
    Zero-shot VQA with CLIP: scores a list of candidate answers
    by image-text similarity.
    """
    CANDIDATE_ANSWERS = [
        # yes/no
        "yes", "no",
        # numbers
        "1", "2", "3", "4", "5", "6", "7", "8", "none", "many",
        # colors
        "red", "blue", "green", "yellow", "white", "black", "brown", "orange", "purple", "pink", "gray",
        # common objects
        "dog", "cat", "person", "man", "woman", "child", "bird", "car", "tree", "food",
        "ball", "book", "phone", "table", "chair", "sky", "water", "grass", "building",
        # actions / states
        "playing", "eating", "sitting", "standing", "running", "walking", "swimming",
        # other common answers
        "sunny", "cloudy", "raining", "daytime", "nighttime",
        "left", "right", "center", "top", "bottom",
        "large", "small", "tall", "short", "old", "young",
        "happy", "sad", "smiling",
        "wood", "metal", "plastic", "concrete", "glass",
        "tennis", "soccer", "baseball", "basketball", "frisbee", "skateboard", "surfing",
        "pizza", "sandwich", "hot dog", "cake", "apple", "banana", "orange", "broccoli",
        "indoors", "outdoors", "kitchen", "bedroom", "bathroom", "living room", "street",
    ]

    prompts = [f"This is a photo where the answer to '{question}' is {ans}" for ans in CANDIDATE_ANSWERS]

    inputs = processor(
        text   = prompts,
        images = image,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = 77,
    ).to(device)

    with torch.no_grad():
        outputs  = clip_model(**inputs)
        # logits_per_image: (1, num_candidates)
        logits   = outputs.logits_per_image[0]
        probs    = torch.softmax(logits, dim=0)

    top_probs, top_ids = probs.topk(min(top_k, len(CANDIDATE_ANSWERS)))
    return [(CANDIDATE_ANSWERS[top_ids[i].item()], top_probs[i].item()) for i in range(len(top_ids))]


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode_options = ["Auto-detect", "Demo (CLIP)", "Trained Model"]
    mode = st.selectbox("Inference mode", mode_options, index=0)

    top_k = st.slider("Top-K answers", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### 📁 Checkpoint Status")
    if os.path.exists(CHECKPOINT_PATH):
        st.success(f"✅ Found: `{CHECKPOINT_PATH}`")
    else:
        st.warning("⚠️ No checkpoint found\n\nRun: `python train.py --debug`")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
**Architecture**
- Vision: ResNet-50 (ImageNet)
- Language: Bi-LSTM / BERT
- Fusion: Hadamard product
- Output: 3,129 answer classes

**Dataset**
- VQA v2 (MSCOCO)
- 1.1M questions
- ~200k images

[📄 Paper](https://arxiv.org/abs/1612.00837) · [📦 Dataset](https://visualqa.org)
    """)


# ── Main UI ───────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🔍 Visual Question Answering</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image → ask any question → AI answers it</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 🖼️ Image")
    upload_tab, url_tab, sample_tab = st.tabs(["Upload", "URL", "Sample"])

    uploaded_image = None

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

    with url_tab:
        img_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
        if img_url:
            try:
                import urllib.request
                with urllib.request.urlopen(img_url) as r:
                    uploaded_image = Image.open(io.BytesIO(r.read())).convert("RGB")
            except Exception as e:
                st.error(f"Could not load image: {e}")

    with sample_tab:
        st.markdown("Choose a sample to try:")
        samples = {
            "🐶 Dog on grass": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg",
            "🏙️ City street":  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
            "🍕 Food plate":   "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/800px-Eq_it-na_pizza-margherita_sep2005_sml.jpg",
        }
        chosen = st.radio("", list(samples.keys()), label_visibility="collapsed")
        if st.button("Load sample"):
            try:
                import urllib.request
                with urllib.request.urlopen(samples[chosen]) as r:
                    uploaded_image = Image.open(io.BytesIO(r.read())).convert("RGB")
                st.success("Sample loaded!")
            except Exception as e:
                st.error(f"Could not load: {e}")

    if uploaded_image:
        st.image(uploaded_image, use_container_width=True, caption="Input image")
        st.caption(f"Size: {uploaded_image.width} × {uploaded_image.height} px")


with col_right:
    st.markdown("### 💬 Ask a Question")

    # Suggested questions
    st.markdown("**Suggested questions:**")
    suggestion_cols = st.columns(2)
    suggestions = [
        "What is in the image?",
        "What color is the main object?",
        "How many objects are there?",
        "Is this indoors or outdoors?",
        "What is the person doing?",
        "What animal is this?",
    ]
    preset_q = None
    for i, s in enumerate(suggestions):
        col = suggestion_cols[i % 2]
        if col.button(s, key=f"sugg_{i}", use_container_width=True):
            preset_q = s

    question = st.text_input(
        "Your question",
        value=preset_q if preset_q else "",
        placeholder="Type any question about the image…",
    )

    ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

    # ── Inference ─────────────────────────────────────────────────────────
    if ask_btn:
        if not uploaded_image:
            st.warning("Please upload or select an image first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            # Determine which mode to use
            checkpoint_exists = os.path.exists(CHECKPOINT_PATH)

            use_clip = (
                mode == "Demo (CLIP)"
                or (mode == "Auto-detect" and not checkpoint_exists)
            )

            with st.spinner("Thinking…"):
                t0 = time.time()

                if use_clip:
                    if mode == "Auto-detect":
                        st.markdown('<div class="mode-info">💡 No checkpoint found — using <b>CLIP demo mode</b> (zero-shot). Train the model for better accuracy.</div>', unsafe_allow_html=True)

                    clip_model, processor, device = load_clip_model()

                    if clip_model is None:
                        st.error("CLIP could not be loaded. Install transformers: `pip install transformers`")
                    else:
                        results = predict_clip_demo(
                            uploaded_image, question, clip_model, processor, device, top_k=top_k
                        )
                        mode_label = "CLIP zero-shot"

                else:
                    model, tokenizer, idx2answer, device = load_trained_model()

                    if model is None:
                        st.error("Could not load checkpoint. Run `python train.py` first.")
                        st.stop()

                    results = predict_trained(
                        uploaded_image, question, model, tokenizer, idx2answer, device, top_k=top_k
                    )
                    mode_label = "Trained VQA model"

                elapsed = time.time() - t0

            # ── Display results ────────────────────────────────────────
            if results:
                top_answer, top_conf = results[0]

                st.markdown(f"""
<div class="answer-card">
    <div style="font-size:0.85rem;opacity:0.8;margin-bottom:0.4rem">TOP ANSWER</div>
    <div class="answer-main">✨ {top_answer.upper()}</div>
    <div class="answer-conf">Confidence: {top_conf*100:.1f}% &nbsp;·&nbsp; {mode_label} &nbsp;·&nbsp; {elapsed*1000:.0f}ms</div>
</div>
""", unsafe_allow_html=True)

                if len(results) > 1:
                    st.markdown("**All top answers:**")
                    for rank, (ans, conf) in enumerate(results, 1):
                        bar_pct = int(conf * 100)
                        st.markdown(
                            f'<div class="top-k-row">'
                            f'<span class="badge">#{rank}</span>'
                            f'<span style="min-width:100px">{ans}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.progress(conf, text=f"{bar_pct}%")

                # Question summary
                st.markdown("---")
                st.caption(f"🖼️ Image  →  ❓ *{question}*  →  💬 **{top_answer}**")


# ── Architecture Explainer (expander) ────────────────────────────────────

with st.expander("📐 Model Architecture & How It Works"):
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
**Pipeline overview:**

```
Image (224×224×3)
   ↓
ResNet-50 backbone
   ↓
Linear(2048 → 512)  ← visual feature
   ↓ ──────────────────────────┐
                               ↓
Question (text)         Hadamard Fusion
   ↓                           ↓
Embedding + Bi-LSTM      MLP Classifier
   ↓                           ↓
Linear(512 → 512)       Softmax (3129 classes)
   ↑──── text feature ─────────┘
```
        """)
    with arch_col2:
        st.markdown("""
**Training details:**
- Dataset: VQA v2 (MSCOCO)
- 1.1M question–answer pairs
- 200k unique COCO images
- 3,129 most-common answers
- Optimizer: AdamW + cosine LR
- Loss: CrossEntropy + label smoothing

**Files:**
| File | Purpose |
|---|---|
| `train.py` | Train the model |
| `evaluate.py` | Evaluate on val set |
| `inference.py` | CLI prediction |
| `app.py` | This Streamlit app |
        """)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("VQA Demo · Built with PyTorch + Streamlit · Dataset: VQA v2 (visualqa.org)")
