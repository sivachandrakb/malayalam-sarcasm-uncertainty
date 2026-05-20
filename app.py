import streamlit as st
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from model import EvidentialDeberta
from utils import predictive_entropy


MODEL_NAME = "microsoft/mdeberta-v3-base"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABELS = {
    0: "Non-Sarcastic",
    1: "Sarcastic"
}


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = EvidentialDeberta()

model.load_state_dict(
    torch.load(
        "model.pt",
        map_location=device
    )
)

model.to(device)

st.title("Uncertainty-Aware Text Classification")

st.write(
    "mDeBERTa + Evidential Learning + MC Dropout"
)

text = st.text_area(
    "Enter text"
)

MC_RUNS = 20


if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter text")
        st.stop()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)

    attention_mask = encoding["attention_mask"].to(device)

    model.train()

    mc_probs = []

    for _ in range(MC_RUNS):

        with torch.no_grad():

            alpha = model(
                input_ids,
                attention_mask
            )

            probs = alpha / alpha.sum(
                dim=1,
                keepdim=True
            )

            mc_probs.append(probs)

    mc_probs = torch.stack(mc_probs)

    mean_probs = mc_probs.mean(dim=0)

    entropy = predictive_entropy(
        mean_probs
    )

    confidence, pred = torch.max(
        mean_probs,
        dim=1
    )

    st.subheader("Prediction")

    st.success(
        f"Predicted Class: {LABELS[pred.item()]}"
    )

    st.write(
        f"Confidence: {confidence.item():.4f}"
    )

    st.write(
        f"Predictive Entropy: {entropy.item():.4f}"
    )

    fig, ax = plt.subplots()

    ax.bar(
        LABELS.values(),
        mean_probs.squeeze().cpu().numpy()
    )

    ax.set_ylabel("Probability")

    ax.set_title("Class Probabilities")

    st.pyplot(fig)

    if entropy.item() < 0.3:

        st.success(
            "Very confident prediction"
        )

    elif entropy.item() < 0.6:

        st.warning(
            "Moderate uncertainty"
        )

    else:

        st.error(
            "High uncertainty detected"
        )

    st.info(
        "MC Dropout performs multiple stochastic forward passes to estimate uncertainty."
    )
