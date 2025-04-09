import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianTokenizer, MarianMTModel
import torch

# Page setup
st.set_page_config(page_title="AI Translation Hub", layout="wide")

# Logo/Header
# Logo and Title Side-by-Side
logo_col, title_col = st.columns([1, 5])  # Adjust width ratio as needed
with logo_col:
    st.image("logo.png", width=80)  # Adjust size to fit nicely
with title_col:
    st.markdown("## Tabby Language Translation Hub")
   # st.markdown("#### Select an AI model and language pair to translate English text.")


# Model and Language Configurations
MODEL_CARDS = {
    "T5": {
        "image": "t5.jpg",  # placeholder
        "languages": {
            "English → German": {
                "path": "./t5-finetuned-en-de",
                "prefix": "translate English to German: ",
                "type": "t5"
            },
            "English → Romanian": {
                "path": "./t5-finetuned-en-ro",
                "prefix": "translate English to Romanian: ",
                "type": "t5"
            }
        }
    },
    "MarianMT": {
        "image": "AI.jpg",  # placeholder
        "languages": {
            "English → German": {
                "path": "./marian-finetuned-en-de",
                "prefix": "",
                "type": "marian"
            },
            "English → Romanian": {
                "path": "./finetuned-en-ro",
                "prefix": "",
                "type": "marian"
            }
        }
    },
    "MBART": {
        "image": "AI.jpg",  # placeholder
        "languages": {}
    },
    "NLLB": {
        "image": "AI.jpg",  # placeholder
        "languages": {}
    }
}

# Model selection step
st.subheader("Choose Your Translation Model")
cols = st.columns(4)
model_choice = None
for i, model_name in enumerate(MODEL_CARDS):
    with cols[i]:
        st.image(MODEL_CARDS[model_name]["image"], use_column_width=True)
        if st.button(f"Select {model_name}", key=f"btn_{model_name}"):
            st.session_state.selected_model = model_name

# Show language options only after a model is selected
if "selected_model" in st.session_state:
    selected_model = st.session_state.selected_model
    st.subheader(f"🌐 Languages for {selected_model}")
    language_options = list(MODEL_CARDS[selected_model]["languages"].keys())

    if language_options:
        selected_lang = st.selectbox("Choose a Language Pair", language_options)
        lang_config = MODEL_CARDS[selected_model]["languages"][selected_lang]

        # Load model/tokenizer
        @st.cache_resource
        def load_model_and_tokenizer(path, model_type):
            if model_type == "t5":
                tokenizer = T5Tokenizer.from_pretrained(path)
                model = T5ForConditionalGeneration.from_pretrained(path)
            else:
                tokenizer = MarianTokenizer.from_pretrained(path)
                model = MarianMTModel.from_pretrained(path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            return model, tokenizer, device

        model, tokenizer, device = load_model_and_tokenizer(lang_config["path"], lang_config["type"])

        # Text input/output
        st.subheader("✏️ Translate Your Text")
        user_input = st.text_area("Enter English Text", height=150)

        if st.button("Translate", use_container_width=True):
            if not user_input.strip():
                st.warning("⚠️ Please enter some English text to translate.")
            else:
                input_text = lang_config["prefix"] + user_input.strip()
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    output = model.generate(**inputs, max_length=128)
                translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                st.success("✅ Translated Text:")
                st.text_area("Output", value=translated_text, height=150)
    else:
        st.warning("⚠️ No languages available for this model yet.")
