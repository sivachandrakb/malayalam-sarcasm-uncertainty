import gradio as gr
import torch
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

MC_RUNS = 20


def predict(text):

    if text.strip() == "":
        return "Please enter text"

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

    result = f"""
Prediction: {LABELS[pred.item()]}

Confidence: {confidence.item():.4f}

Entropy: {entropy.item():.4f}
"""

    return result


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter Malayalam text here..."
    ),
    outputs="text",
    title="Uncertainty-Aware Malayalam Sarcasm Detection",
    description="mDeBERTa + Evidential Learning + MC Dropout"
)

demo.launch()
