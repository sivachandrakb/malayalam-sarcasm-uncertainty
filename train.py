import torch
import torch.nn.functional as F
import pandas as pd
import kagglehub
import os

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from model import EvidentialDeberta
from utils import evidential_loss

MODEL_NAME = "microsoft/mdeberta-v3-base"

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
NUM_CLASSES = 2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# DOWNLOAD DATASET
# -----------------------------

path = kagglehub.dataset_download(
    "subodhuniyal/malyalam-sarcasm"
)

print("Dataset Path:", path)

files = os.listdir(path)

print("Files:", files)

csv_file = None

for f in files:
    if f.endswith(".csv"):
        csv_file = os.path.join(path, f)

print("CSV File:", csv_file)

df = pd.read_csv(csv_file)

print(df.head())

# -----------------------------
# CHANGE COLUMN NAMES IF NEEDED
# -----------------------------

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

texts = df[TEXT_COLUMN].astype(str).tolist()

labels = df[LABEL_COLUMN].tolist()

# -----------------------------
# TOKENIZER
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# -----------------------------
# DATASET CLASS
# -----------------------------

class SarcasmDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "label":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long
                )
        }

dataset = SarcasmDataset(
    texts,
    labels,
    tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# -----------------------------
# MODEL
# -----------------------------

model = EvidentialDeberta(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES
)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

# -----------------------------
# TRAINING
# -----------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in loader:

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        y = F.one_hot(
            labels,
            NUM_CLASSES
        ).float()

        alpha = model(
            input_ids,
            attention_mask
        )

        loss = evidential_loss(y, alpha)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch+1} Loss: {avg_loss:.4f}"
    )

# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(
    model.state_dict(),
    "malayalam_sarcasm_model.pt"
)

print("Model Saved")
