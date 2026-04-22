import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.constants import COUNT_MARKER
from LSTM.lstm_model import NetObsReliability
from env.field_generator import FieldGenerator
from LSTM.dataset_lstm import ProceduralPatchDataset

STEPS = 10000
BATCH_SIZE = 128
LR = 0.001
SEQ_LEN = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_patch_reliability():
    print(f"\n\n{'='*60}")
    print("TRAINING OCCLUSION-AWARE PATCH RELIABILITY MODEL")
    print(f"{'='*60}")

    # -------------------------
    # Dataset
    # -------------------------
    gen = FieldGenerator(size=20, len_scale=5.0)
    dataset = ProceduralPatchDataset(generator=gen, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # -------------------------
    # Model
    # -------------------------
    model = NetObsReliability(
        num_classes=COUNT_MARKER,
        hidden_size=128,
        num_layers=1,
        dropout=0.2
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Target continuo in [0,1] per ciascuna delle 9 celle
    criterion = nn.SmoothL1Loss()

    model.train()

    total_loss = 0.0
    step = 0
    iterator = iter(dataloader)

    while step < STEPS:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            inputs, targets = next(iterator)

        inputs = inputs.to(DEVICE)    # (B, seq_len, 9 + 9*COUNT_MARKER)
        targets = targets.to(DEVICE)  # (B, 9)

        optimizer.zero_grad()

        outputs = model(inputs)
        pred_conf = outputs["pred_confidence_patch"]   # (B, 9)

        loss = criterion(pred_conf, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 100 == 0:
            with torch.no_grad():
                pred_mean = pred_conf.mean().item()
                target_mean = targets.mean().item()

            print(
                f"Step {step}/{STEPS} | "
                f"Avg Loss = {total_loss / 500:.4f} | "
                f"PredMean = {pred_mean:.4f} | "
                f"TargetMean = {target_mean:.4f}"
            )
            total_loss = 0.0

    base_dir = os.path.dirname(__file__)  # cartella dove sta train_lstm.py
    save_dir = os.path.join(base_dir, "models")

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "patch_reliability_model.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Model saved: {save_path}")


if __name__ == "__main__":
    train_patch_reliability()