import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
import torch

# Model paths
MODEL_OPTIONS = {
    "English → German": "./marian-finetuned-en-de",
    "English → Romanian": "./finetuned-en-ro"
}

@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Streamlit UI setup
st.set_page_config(page_title="English Translator", layout="centered")
st.title("🌍 Translator")
st.markdown("Select a model below, enter English text, and get your translation in German or Romanian.")

# Dropdown selector
model_choice = st.selectbox("Choose Translation Model", list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[model_choice]
model, tokenizer, device = load_model_and_tokenizer(model_path)

# Text input
text_input = st.text_area("Enter English Text", height=150)

# Translate button
if st.button("Translate"):
    if not text_input.strip():
        st.warning("Please enter some English text.")
    else:
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=128)
        output = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.success("**Translated Text:**")
        st.text_area("Output", value=output, height=150)
