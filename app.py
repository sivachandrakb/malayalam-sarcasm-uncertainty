import streamlit as st
import torch
from transformers import AutoTokenizer
from model import EvidentialDeBERTa
from utils import mc_dropout_predict

MODEL_PATH = "saved_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = EvidentialDeBERTa("microsoft/deberta-v3-base")
    model.load_state_dict(torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location="cpu"))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Malayalam Sarcasm Detection with Uncertainty")
st.write("Uncertainty-Aware DeBERTa Model")

text = st.text_area("Enter Malayalam Text")

if st.button("Predict"):
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=128)

    with torch.no_grad():
        mean_probs, epi_uncertainty = mc_dropout_predict(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            T=20
        )

    prediction = mean_probs.argmax()
    confidence = mean_probs.max()

    if prediction == 1:
        label = "Sarcastic"
    else:
        label = "Non-Sarcastic"

    st.subheader("Prediction")
    st.write(f"Label: {label}")

    st.subheader("Confidence")
    st.write(f"{confidence:.4f}")

    st.subheader("Epistemic Uncertainty")
    st.write(f"{epi_uncertainty:.6f}")

    if confidence > 0.8:
        st.success("High Confidence Prediction")
    elif confidence > 0.6:
        st.warning("Moderate Confidence")
    else:
        st.error("Low Confidence - Model Unsure")
