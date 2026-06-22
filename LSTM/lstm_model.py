import torch
import torch.nn as nn


class CellObserverLSTM(nn.Module):
    """
    LSTM condiviso che osserva una singola cella alla volta.

    Predice DIRETTAMENTE la distribuzione sulle K classi di una cella,
    integrando la sequenza temporale delle osservazioni di quella cella.

    - pesi CONDIVISI tra tutte le celle
    - hidden state (h, c) SEPARATO per cella (gestito esternamente dall'Agent)

    Input per passo: [sensor_dist (K), alignment (1)]
    Output: logits sulle K classi

    MIGLIORAMENTO (punto 1): hidden_size di default portato da 64 a 128 per
    maggiore capacità rappresentativa. La head ora ha un hidden intermedio
    proporzionato al nuovo hidden_size.
    """

    def __init__(self, num_classes=10, hidden_size=128):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.input_size = num_classes + 1  # sensor_dist (K) + alignment (1)

        self.cell = nn.LSTMCell(self.input_size, hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def step(self, x, h, c):
        """Un passo per un batch di celle.
        x: (B, input_size), h/c: (B, hidden_size)
        ritorna logits (B, num_classes), h_new, c_new
        """
        h_new, c_new = self.cell(x, (h, c))
        logits = self.head(h_new)
        return logits, h_new, c_new

    def forward(self, seq):
        """Forward su sequenza completa (usato in training senza padding).
        seq: (B, T, input_size) -> logits finali (B, num_classes)
        """
        B, T, _ = seq.shape
        h = torch.zeros(B, self.hidden_size, device=seq.device)
        c = torch.zeros(B, self.hidden_size, device=seq.device)
        logits = None
        for t in range(T):
            logits, h, c = self.step(seq[:, t, :], h, c)
        return logits