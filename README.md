# Tabby Language Translator

Personal Translator and basic language and sentence structure assistant. 


## Description

## Overview
This project demonstrates how to fine-tune and deploy multilingual translation models using Hugging Face Transformers and Gradio. Two translation directions are supported: 

- **English → German** (`marian-finetuned-en-de`)
- **English → Romanian** (`finetuned-en-ro`)
-  **English → Spanish** (`Helsinki-NLP/opus-mt-en-es`)

Each model is fine-tuned on a subset of the WMT/Kaggle datasets and deployed through a Gradio user interface.



## Files Needed

### 1. `train_marian_en_de_with_bleu.ipynb`
- Trains the MarianMT model on English → German using WMT14 dataset
- Includes preprocessing, BLEU evaluation, and saving the model

### 2. `train_marian_en_ro.ipynb`
- Equivalent training script for English → Romanian using WMT16 dataset

### 3. `EN_to_ES_Model.ipynb`
- Equivalent training script for English → Spanish using Kaggle dataset

### 4. `translate_ui_selector.ipynb`
- Gradio interface with a dropdown selector to pick either the EN→DE or EN→RO model
- Accepts user input and returns the translated output



### How to Train the Model

1. **Choose a dataset**:
   - For English → German: `wmt14`, split `de-en`
   - For English → Romanian: `wmt16`, split `ro-en`
   -  For English → Spanish: `EN_ES.txt`, split `en-es`

2. **Run the appropriate training script**:
   - Notebook: `train_marian_en_de_with_bleu.ipynb`
   - Script: `train_marian_en_ro.ipynb`

3. **What each script does**:
   - Tokenizes the dataset
   - Fine-tunes the MarianMT model
   - Saves the model to a local folder (`./marian-finetuned-en-de` or `./finetuned-en-ro`)
   - Optionally computes BLEU score


---

## How to Test Translations

1. **Run the Gradio UI**:

   ```bash
   python translate_ui_selector.py


## Notes

 - For large-scale training, switch to full dataset splits and train for more epochs
 - BLEU scores may vary depending on data size and fine-tuning depth
 - Trained models are saved locally and reused by the Gradio UI
 - Powerpoint and streamlit code located in resources file
 - Google Slides are located in resources file




## Authors/Contributors 



Christopher Davis [github](https://github.com/Glibbly)

Laxmi Atluri [github](https://github.com/la-aicode)

Leonard Forrester [github](https://github.com/leoforr)

Sara Moujahed [github](https://github.com/saracmoujahed)

Roderick Burroughs [github](https://github.com/RodBurr)



