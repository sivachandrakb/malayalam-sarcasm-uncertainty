# Uncertainty-Aware Malayalam Sarcasm Detection

An uncertainty-aware multilingual NLP system using:

- mDeBERTa-v3
- Evidential Deep Learning
- Monte Carlo Dropout
- Streamlit

## Features

- Malayalam sarcasm detection
- Confidence-calibrated predictions
- Predictive entropy estimation
- MC Dropout uncertainty estimation
- Interactive Streamlit dashboard

---

## Model Architecture

Input Text
↓
mDeBERTa-v3 Encoder
↓
Evidential Head
↓
MC Dropout Sampling
↓
Prediction + Uncertainty Estimation

---

## Dataset

Malayalam Sarcasm Dataset:
https://www.kaggle.com/datasets/subodhuniyal/malyalam-sarcasm

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train.py
```

---

## Run Demo

```bash
streamlit run app.py
```

---

## Example

Input:
വളരെ നല്ല സര്‍വീസാണ്, മൂന്ന് മണിക്കൂര്‍ കാത്തിരുന്നു

Prediction:
Sarcastic

Confidence:
0.91

Entropy:
0.22

---

## Technologies Used

- PyTorch
- Hugging Face Transformers
- Streamlit
- Evidential Deep Learning
- Monte Carlo Dropout

---

## Future Improvements

- Deep Ensembles
- Temperature Scaling
- OOD Detection
- Attention Visualization
- Reliability Diagrams

---

## Author

Sivachandra K B
