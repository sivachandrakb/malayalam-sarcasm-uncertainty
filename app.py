import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================
# Configuration
# ============================

MODEL_NAME = "sivachandrakb/malayalam-sarcasm-deberta"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

# ============================
# Load Model + Tokenizer
# ============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model.to(DEVICE)
model.eval()

# ============================
# Enable MC Dropout
# ============================

def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()

# ============================
# Prediction Function
# ============================

def predict(text, T):

    if text.strip() == "":
        return "Please enter text.", {}, 0.0, 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    ).to(DEVICE)

    model.apply(enable_dropout)

    mc_probs = []
    alpha_last = None

    for _ in range(T):
        with torch.no_grad():
            outputs = model(**inputs)

            logits = outputs.logits

            # Evidential assumption: Softplus for evidence
            evidence = torch.nn.functional.softplus(logits)
            alpha = evidence + 1
            alpha_last = alpha

            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S

            mc_probs.append(probs.detach().cpu().numpy())

    mc_probs = np.array(mc_probs)
    mean_probs = np.mean(mc_probs, axis=0)

    # Epistemic Uncertainty (variance across MC passes)
    epistemic = float(np.mean(np.var(mc_probs, axis=0)))

    # Aleatoric Uncertainty (Dirichlet uncertainty mass)
    alpha_np = alpha_last.detach().cpu().numpy()
    S_total = np.sum(alpha_np)
    K = alpha_np.shape[1]
    aleatoric = float(K / S_total)

    predicted_class = int(np.argmax(mean_probs))

    label = "Sarcastic" if predicted_class == 1 else "Not Sarcastic"

    prob_dict = {
        "Not Sarcastic": float(mean_probs[0][0]),
        "Sarcastic": float(mean_probs[0][1])
    }

    return label, prob_dict, aleatoric, epistemic


# ============================
# Gradio UI
# ============================

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=4, placeholder="Enter Malayalam or Tamil text here..."),
        gr.Slider(5, 50, value=20, step=5, label="MC Dropout Passes (T)")
    ],
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.Label(label="Class Probabilities"),
        gr.Number(label="Aleatoric Uncertainty (Dirichlet Mass)"),
        gr.Number(label="Epistemic Uncertainty (MC Variance)")
    ],
    title="🧠 Uncertainty-Aware Transformer for Sarcasm Detection",
    description="""
    This model integrates Evidential Deep Learning and Monte Carlo Dropout
    to estimate both aleatoric and epistemic uncertainty for sarcasm detection
    in Malayalam and Tamil text.
    """,
)

if __name__ == "__main__":
    demo.launch()
