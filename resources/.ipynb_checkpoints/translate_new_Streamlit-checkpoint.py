import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianTokenizer, MarianMTModel
import torch
import evaluate

# Define model configurations
MODEL_OPTIONS = {
    "T5 English → German": {
        "path": "./t5-finetuned-en-de",
        "prefix": "translate English to German: ",
        "type": "t5"
    },
    "T5 English → Romanian": {
        "path": "./t5-finetuned-en-ro",
        "prefix": "translate English to Romanian: ",
        "type": "t5"
    },
    "Marian English → German": {
        "path": "./marian-finetuned-en-de",
        "prefix": "",
        "type": "marian"
    },
    "Marian English → Romanian": {
        "path": "./finetuned-en-ro",
        "prefix": "",
        "type": "marian"
    }
}

# Load BLEU evaluator
bleu = evaluate.load("bleu")

# Cache model/tokenizer loading
@st.cache_resource
def load_model_and_tokenizer(model_path, model_type):
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Streamlit UI setup
st.set_page_config(page_title="Multilingual Translator", layout="wide")
st.title("🌍 Multilingual Translator")
st.markdown("Translate English to multiple languages using fine-tuned T5 or MarianMT models.")

# Layout: sidebar (30%) and main area (70%)
sidebar, main = st.columns([1, 2.5])

with sidebar:
    st.markdown("### 🔄 Select Translation Model")
    selected_label = st.radio("Choose model:", list(MODEL_OPTIONS.keys()), index=0)
    model_info = MODEL_OPTIONS[selected_label]
    model, tokenizer, device = load_model_and_tokenizer(model_info["path"], model_info["type"])
    prefix = model_info["prefix"]
    model_type = model_info["type"]

with main:
    st.subheader(f"Translation: {selected_label}")
    user_input = st.text_area("Enter English Text", height=200)

    if st.button("Translate", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Please enter some English text to translate.")
        else:
            # Prepare input
            input_text = prefix + user_input.strip()
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

            # Generate translation
            with torch.no_grad():
                if model_type == "t5":
                    output = model.generate(**inputs, max_length=128)
                else:
                    output = model.generate(**inputs, max_length=128)

            translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.success("✅ Translation Result:")
            st.text_area("Translated Text", value=translated_text, height=150)

            # Optional BLEU score (if user provides reference)
            st.markdown("### 📏 Evaluate Translation (Optional)")
            reference = st.text_input("Enter Reference Translation (Expected Output)")

            if reference:
                score = bleu.compute(predictions=[translated_text.strip()], references=[[reference.strip()]])
                st.info(f"BLEU Score: {score['bleu']:.4f}")

# Optional styling
st.markdown("""
<style>
    .stRadio > div { flex-direction: column; }
    .css-1v0mbdj p { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)
