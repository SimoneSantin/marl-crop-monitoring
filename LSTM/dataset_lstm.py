import torch
import numpy as np
from torch.utils.data import IterableDataset

from utils.constants import COUNT_MARKER
from env.sensor import ScalarSensor


class CellObserverDataset(IterableDataset):
    """
    Dataset per addestrare il CellObserverLSTM.

    Ogni campione è la sequenza di osservazioni di UNA SINGOLA cella, osservata
    più volte da direzioni di movimento diverse. Il target è la classe VERA.

    MIGLIORAMENTO (punto 2): la distribuzione del numero di osservazioni n_obs
    NON è più uniforme tra 1 e max_obs. È invece sbilanciata verso valori BASSI,
    perché in deployment la maggior parte delle celle viene osservata 1-2 volte.
    Questo evita che il training sia dominato da sequenze lunghe (task facile,
    accuracy ~1.0) e costringe il modello a specializzarsi sul caso difficile
    e realistico di poche osservazioni.
    """

    def __init__(self, generator, min_obs=1, max_obs=8, samples_per_field=200,
                 obs_distribution="geometric", geo_p=0.5):
        super().__init__()
        self.generator = generator
        self.min_obs = min_obs
        self.max_obs = max_obs
        self.samples_per_field = samples_per_field
        self.sensor = ScalarSensor()

        # distribuzione di n_obs: "geometric" favorisce valori bassi,
        # "uniform" è il vecchio comportamento
        self.obs_distribution = obs_distribution
        self.geo_p = geo_p

        # pre-calcola i pesi per il campionamento di n_obs
        ks = np.arange(min_obs, max_obs + 1)
        if obs_distribution == "geometric":
            # P(n) ∝ (1-p)^(n-1) : decresce con n, favorisce poche osservazioni
            w = (1.0 - geo_p) ** (ks - min_obs)
        else:
            w = np.ones_like(ks, dtype=np.float64)
        self.n_obs_values = ks
        self.n_obs_probs = (w / w.sum()).astype(np.float64)

        self.directions = [
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707),
        ]

    def _sample_n_obs(self):
        return int(np.random.choice(self.n_obs_values, p=self.n_obs_probs))

    def _observe_cell(self, true_val, alpha_angle, drone_dir):
        plant_dx = np.sin(alpha_angle)
        plant_dy = np.cos(alpha_angle)
        dx, dy = drone_dir
        alignment = abs(dx * plant_dx + dy * plant_dy)
        alignment = float(np.clip(alignment, 0.0, 1.0))
        noise_intensity = 1.0 - alignment
        sensor_dist = self.sensor.observe(true_val, noise_intensity).astype(np.float32)
        return sensor_dist, alignment

    def __iter__(self):
        while True:
            field = self.generator.generate_field()
            grid_counts = field["true_counts"]
            grid_angles = field["true_angles"]
            size = self.generator.size

            for _ in range(self.samples_per_field):
                r = np.random.randint(0, size)
                c = np.random.randint(0, size)
                true_val = int(grid_counts[r, c])
                alpha_angle = grid_angles[r, c]

                n_obs = self._sample_n_obs()

                seq = []
                for _ in range(n_obs):
                    drone_dir = self.directions[np.random.randint(len(self.directions))]
                    sensor_dist, alignment = self._observe_cell(true_val, alpha_angle, drone_dir)
                    step = np.concatenate([sensor_dist, [alignment]]).astype(np.float32)
                    seq.append(step)

                seq_tensor = torch.tensor(np.array(seq), dtype=torch.float32)
                target = torch.tensor(true_val, dtype=torch.long)
                yield seq_tensor, target


def collate_pad(batch):
    seqs, targets = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    feat = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, feat, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0], :] = s
    targets = torch.stack(targets)
    return padded, lengths, targets