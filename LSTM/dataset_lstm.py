import torch
import numpy as np
from torch.utils.data import IterableDataset

from utils.constants import COUNT_MARKER
from env.sensor import ScalarSensor


class ProceduralPatchDataset(IterableDataset):
    def __init__(self, generator, seq_len=5):
        """
        Dataset procedurale per addestrare un modello di observation reliability.

        Input per esempio:
            input_tensor shape = (seq_len, 9 + 9 * COUNT_MARKER)

        Target per esempio:
            target_tensor shape = (9,)
            Ogni valore rappresenta quanto l'osservazione è utile
            per migliorare la belief (CE gain normalizzato in [0,1]).
        """
        self.generator = generator
        self.seq_len = seq_len
        self.sensor = ScalarSensor()

        self.moves = [
            np.array([-1, 0], dtype=np.float32),  # up
            np.array([1, 0], dtype=np.float32),   # down
            np.array([0, -1], dtype=np.float32),  # left
            np.array([0, 1], dtype=np.float32)    # right
        ]

    def _normalize_ce_gain(self, ce_gain):
        """
        Rimappa CE gain in [0,1].
        negativo -> osservazione fuorviante
        positivo -> osservazione utile
        """
        ce_gain = np.clip(ce_gain, -1.0, 1.0)
        return (ce_gain + 1.0) / 2.0

    def __iter__(self):
        while True:
            field_data = self.generator.generate_field()

            grid_counts = field_data["true_counts"]   # (H, W)
            grid_angles = field_data["true_angles"]   # (H, W)
            size = self.generator.size

            for _ in range(50):
                # centro valido per patch 3x3 piena
                r = np.random.randint(1, size - 1)
                c = np.random.randint(1, size - 1)

                input_sequence = []
                final_target = None

                for _ in range(self.seq_len):
                    move = self.moves[np.random.randint(0, len(self.moves))]
                    dx, dy = move[0], move[1]

                    alignment_patch = np.zeros(9, dtype=np.float32)
                    sensor_patch_flat = np.zeros(9 * COUNT_MARKER, dtype=np.float32)
                    target_patch = np.zeros(9, dtype=np.float32)

                    idx = 0
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            nx, ny = r + i, c + j

                            true_val = int(grid_counts[nx, ny])
                            alpha = grid_angles[nx, ny]

                            plant_vec_dx = np.sin(alpha)
                            plant_vec_dy = np.cos(alpha)

                            alignment = abs(dx * plant_vec_dx + dy * plant_vec_dy)
                            alignment = float(np.clip(alignment, 0.0, 1.0))

                            noise_intensity = 1.0 - alignment
                            sensor_dist = self.sensor.observe(true_val, noise_intensity)

                            # prior simulata: belief iniziale casuale
                            prior = np.random.dirichlet(np.ones(COUNT_MARKER)).astype(np.float32)

                            # CE before
                            ce_before = -np.log(prior[true_val] + 1e-9)

                            # posterior semplice Bayes-like
                            posterior = prior * sensor_dist
                            posterior /= (posterior.sum() + 1e-9)

                            # CE after
                            ce_after = -np.log(posterior[true_val] + 1e-9)

                            ce_gain = ce_before - ce_after
                            reliability_target = self._normalize_ce_gain(ce_gain)

                            alignment_patch[idx] = alignment
                            sensor_patch_flat[idx * COUNT_MARKER:(idx + 1) * COUNT_MARKER] = sensor_dist
                            target_patch[idx] = reliability_target

                            idx += 1

                    step_input = np.concatenate([
                        alignment_patch,
                        sensor_patch_flat
                    ]).astype(np.float32)

                    input_sequence.append(step_input)
                    final_target = target_patch

                input_tensor = torch.tensor(np.array(input_sequence), dtype=torch.float32)
                target_tensor = torch.tensor(final_target, dtype=torch.float32)  # shape (9,)

                yield input_tensor, target_tensor