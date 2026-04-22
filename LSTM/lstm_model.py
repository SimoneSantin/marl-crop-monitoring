import torch
import torch.nn as nn

class NetObsReliability(nn.Module):
    def __init__(self, num_classes=10, hidden_size=128, num_layers=1, dropout=0.2):
        """
        Modello LSTM per stimare l'affidabilità delle 9 celle di una patch 3x3.

        Input per timestep:
            [alignment_patch (9), sensor_patch_flat (9 * num_classes)]

        Output:
            pred_confidence_patch: tensor shape (B, 9), valori in [0,1]
        """
        super().__init__()

        self.num_classes = num_classes
        self.patch_cells = 9
        self.input_size = self.patch_cells + self.patch_cells * num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, self.patch_cells)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: tensor di shape (B, T, input_size)

        Returns:
            dict con:
                pred_confidence_patch: (B, 9) in [0,1]
        """
        x = x.float()

        lstm_out, _ = self.lstm(x)          # (B, T, hidden_size)
        last_out = lstm_out[:, -1, :]       # (B, hidden_size)

        confidence_logits = self.mlp(last_out)          # (B, 9)
        pred_confidence_patch = self.sigmoid(confidence_logits)

        return {
            "pred_confidence_patch": pred_confidence_patch
        }