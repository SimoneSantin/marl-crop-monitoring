import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.constants import COUNT_MARKER
from env.field_generator import FieldGenerator
from LSTM.lstm_model import CellObserverLSTM
from LSTM.dataset_lstm import CellObserverDataset, collate_pad

STEPS = 20000
BATCH_SIZE = 256
LR = 0.001
HIDDEN_SIZE = 128          # punto 1: era 64
MAX_OBS = 8
OBS_DISTRIBUTION = "geometric"  # punto 2: favorisce sequenze corte
GEO_P = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_cell_observer():
    print(f"\n{'='*60}")
    print("TRAINING CELL OBSERVER LSTM (hidden=128, geometric n_obs)")
    print(f"{'='*60}")

    gen = FieldGenerator(size=20, len_scale=5.0)
    dataset = CellObserverDataset(
        generator=gen, min_obs=1, max_obs=MAX_OBS,
        obs_distribution=OBS_DISTRIBUTION, geo_p=GEO_P,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_pad)

    model = CellObserverLSTM(num_classes=COUNT_MARKER, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    # accuracy separata per sequenze corte (n_obs <= 2) = caso difficile/realistico
    correct_short = 0
    total_short = 0
    step = 0
    iterator = iter(dataloader)

    while step < STEPS:
        try:
            padded, lengths, targets = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            padded, lengths, targets = next(iterator)

        padded = padded.to(DEVICE)
        targets = targets.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()

        B, T, _ = padded.shape
        h = torch.zeros(B, model.hidden_size, device=DEVICE)
        c = torch.zeros(B, model.hidden_size, device=DEVICE)
        logits_final = torch.zeros(B, model.num_classes, device=DEVICE)

        for t in range(T):
            logits, h, c = model.step(padded[:, t, :], h, c)
            mask = (lengths == (t + 1))
            if mask.any():
                logits_final[mask] = logits[mask]
            still_active = (lengths > (t + 1)).float().unsqueeze(1)
            h = h * still_active + h.detach() * (1 - still_active)

        loss = criterion(logits_final, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits_final.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += B

        # accuracy sul sottoinsieme difficile (poche osservazioni)
        short_mask = (lengths <= 2)
        if short_mask.any():
            correct_short += (preds[short_mask] == targets[short_mask]).sum().item()
            total_short += int(short_mask.sum().item())

        step += 1

        if step % 200 == 0:
            acc = correct / max(total, 1)
            acc_short = correct_short / max(total_short, 1)
            print(f"Step {step}/{STEPS} | Loss={total_loss/200:.4f} | "
                  f"Acc={acc:.4f} | Acc(n_obs<=2)={acc_short:.4f}")
            total_loss = 0.0
            correct = 0; total = 0
            correct_short = 0; total_short = 0

    base_dir = os.path.dirname(__file__)
    save_dir = os.path.join(base_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cell_observer_lstm.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")


if __name__ == "__main__":
    train_cell_observer()