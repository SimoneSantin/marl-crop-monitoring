import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.constants import COUNT_MARKER
from LSTM.lstm_model import NetObsReliability
from env.field_generator import FieldGenerator
from LSTM.dataset_lstm import ProceduralPatchDataset

STEPS = 15000
BATCH_SIZE = 128
LR = 0.001
SEQ_LEN = 8
LOCAL_WINDOW = 2
VAR_PENALTY_WEIGHT = 0.5   # peso della penalità sulla varianza bassa
MIN_PRED_STD = 0.05        # varianza minima che vogliamo dalle predizioni
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_patch_reliability():
    print(f"\n\n{'='*60}")
    print("TRAINING OCCLUSION-AWARE PATCH RELIABILITY MODEL (v3)")
    print(f"{'='*60}")

    gen = FieldGenerator(size=20, len_scale=5.0)
    dataset = ProceduralPatchDataset(
        generator=gen,
        seq_len=SEQ_LEN,
        local_window=LOCAL_WINDOW,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = NetObsReliability(
        num_classes=COUNT_MARKER,
        hidden_size=128,
        num_layers=1,
        dropout=0.2
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse_criterion = nn.MSELoss()

    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_var_penalty = 0.0
    step = 0
    iterator = iter(dataloader)

    while step < STEPS:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            inputs, targets = next(iterator)

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        pred_conf = outputs["pred_confidence_patch"]   # (B, 9)

        # loss principale: MSE sul target
        mse_loss = mse_criterion(pred_conf, targets)

        # penalità se la std delle predizioni è troppo bassa
        # vogliamo che il modello produca output diversi, non sempre ~0.59
        pred_std = pred_conf.std()
        var_penalty = torch.clamp(MIN_PRED_STD - pred_std, min=0.0)

        loss = mse_loss + VAR_PENALTY_WEIGHT * var_penalty
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_var_penalty += var_penalty.item()
        step += 1

        if step % 100 == 0:
            with torch.no_grad():
                pred_mean = pred_conf.mean().item()
                pred_std_val = pred_conf.std().item()
                target_mean = targets.mean().item()
                target_std_val = targets.std().item()

                # correlazione tra pred e target: se alta, il modello sta imparando
                # non solo la media ma anche la variazione
                pred_flat = pred_conf.flatten()
                tgt_flat = targets.flatten()
                corr = torch.corrcoef(torch.stack([pred_flat, tgt_flat]))[0, 1].item()

            print(
                f"Step {step}/{STEPS} | "
                f"Loss={total_loss/100:.4f} | "
                f"MSE={total_mse/100:.4f} | "
                f"VarPen={total_var_penalty/100:.4f} | "
                f"PredMean={pred_mean:.3f} PredStd={pred_std_val:.3f} | "
                f"TgtMean={target_mean:.3f} TgtStd={target_std_val:.3f} | "
                f"Corr={corr:.3f}"
            )
            total_loss = 0.0
            total_mse = 0.0
            total_var_penalty = 0.0

    base_dir = os.path.dirname(__file__)
    save_dir = os.path.join(base_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "patch_reliability_model_v3.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")


if __name__ == "__main__":
    train_patch_reliability()