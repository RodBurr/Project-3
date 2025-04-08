import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
import torch

# Model paths
MODEL_OPTIONS = {
    "English → German": "./marian-finetuned-en-de",
    "English → Romanian": "./finetuned-en-ro"
}

# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Page setup
st.set_page_config(page_title="Multilingual Translator", layout="wide")
st.title("🌍 Multilingual Translator")
st.markdown("Select a translation direction and input English text to see the translation result.")

# Create two columns: 30% (sidebar) and 70% (main content)
sidebar, main = st.columns([1, 2.5])

with sidebar:
    st.markdown("### 🔄 Choose Language Pair")

    # Language selection tiles
    selected_tile = st.radio(
        label="Select Language",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        format_func=lambda x: x,
        help="Choose the translation direction"
    )

    model_path = MODEL_OPTIONS[selected_tile]
    model, tokenizer, device = load_model_and_tokenizer(model_path)

with main:
    st.markdown(f"#### Translation: {selected_tile}")
    user_input = st.text_area("Enter English Text", height=200)

    if st.button("Translate", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=128)
            output = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success("**Translated Text:**")
            st.text_area("Output", value=output, height=150)

# Styling: make radio buttons vertical
st.markdown("""
<style>
    .stRadio > div { flex-direction: column; }
</style>
""", unsafe_allow_html=True)
