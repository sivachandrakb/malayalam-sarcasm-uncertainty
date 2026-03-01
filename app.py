# import streamlit as st
# import torch
# from transformers import AutoTokenizer
# from model import EvidentialDeBERTa
# from utils import mc_dropout_predict

# st.set_page_config(page_title="Malayalam Sarcasm Detector", layout="centered")

# import streamlit as st
# import torch
# from transformers import AutoTokenizer
# from model import EvidentialDeBERTa

# @st.cache_resource
# def load_model():
#     MODEL_NAME = "microsoft/deberta-v3-base"
#     HF_MODEL = "sivachandrakb/malayalam-sarcasm-deberta"

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     model = EvidentialDeBERTa(MODEL_NAME)

#     # Download large model from HuggingFace
#     state_dict = torch.hub.load_state_dict_from_url(
#         f"https://huggingface.co/{HF_MODEL}/resolve/main/best_model.pt",
#         map_location="cpu"
#     )

#     model.load_state_dict(state_dict)
#     model.eval()

#     return tokenizer, model

# # -----------------------------
# # UI
# # -----------------------------
# st.title("🧠 Malayalam Sarcasm Detection")
# st.write("Uncertainty-Aware DeBERTa Model (EDL + MC Dropout)")

# text = st.text_area("Enter Malayalam Text")

# if st.button("Predict"):

#     if text.strip() == "":
#         st.warning("Please enter text.")
#     else:
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             max_length=128
#         )

#         with torch.no_grad():
#             mean_probs, epi_uncertainty = mc_dropout_predict(
#                 model,
#                 inputs["input_ids"],
#                 inputs["attention_mask"],
#                 T=20
#             )

#         prediction = mean_probs.argmax()
#         confidence = mean_probs.max()

#         label = "Sarcastic" if prediction == 1 else "Non-Sarcastic"

#         st.subheader("Prediction")
#         st.write(f"### {label}")

#         st.subheader("Confidence")
#         st.write(f"{confidence:.4f}")

#         st.subheader("Epistemic Uncertainty")
#         st.write(f"{epi_uncertainty:.6f}")

#         # Reliability Indicator
#         if confidence > 0.85:
#             st.success("High Confidence Prediction")
#         elif confidence > 0.65:
#             st.warning("Moderate Confidence")
#         else:
#             st.error("Low Confidence - Model Unsure")

import streamlit as st
import torch
from transformers import AutoTokenizer
from model import EvidentialDeBERTa
from utils import mc_dropout_predict

st.set_page_config(page_title="Malayalam Sarcasm Detection")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    MODEL_NAME = "microsoft/deberta-v3-base"
    HF_MODEL = "sivachandrakb/malayalam-sarcasm-deberta"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = EvidentialDeBERTa(MODEL_NAME)

    state_dict = torch.hub.load_state_dict_from_url(
        f"https://huggingface.co/{HF_MODEL}/resolve/main/best_model.pt",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, model

# IMPORTANT: unpack here
tokenizer, model = load_model()

# -------------------------
# UI
# -------------------------
st.title("🧠 Malayalam Sarcasm Detection")
st.write("Uncertainty-Aware DeBERTa Model (EDL + MC Dropout)")

text = st.text_area("Enter Malayalam Text")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            mean_probs, epi_uncertainty = mc_dropout_predict(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                T=20
            )

        prediction = mean_probs.argmax()
        confidence = mean_probs.max()

        label = "Sarcastic" if prediction == 1 else "Non-Sarcastic"

        st.success(f"Prediction: {label}")
        st.write("Confidence:", round(float(confidence), 4))
        st.write("Epistemic Uncertainty:", round(float(epi_uncertainty), 6))
