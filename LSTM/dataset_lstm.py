import torch
import numpy as np
from torch.utils.data import IterableDataset

from utils.constants import COUNT_MARKER
from env.sensor import ScalarSensor


class ProceduralPatchDataset(IterableDataset):

    def __init__(
        self,
        generator,
        seq_len=5,
        episodes_per_field=50
    ):
        super().__init__()

        self.generator = generator
        self.seq_len = seq_len
        self.episodes_per_field = episodes_per_field

        self.sensor = ScalarSensor()

        self.moves = [
            np.array([-1, 0], dtype=np.int32),
            np.array([1, 0], dtype=np.int32),
            np.array([0, -1], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
        ]

    def _normalize_ce_gain(self, ce_gain):
        ce_gain = np.clip(ce_gain, -1.0, 1.0)
        return (ce_gain + 1.0) / 2.0

    def _build_step(
        self,
        r,
        c,
        dx,
        dy,
        grid_counts,
        grid_angles,
        belief_map
    ):

        alignment_patch = np.zeros(9, dtype=np.float32)

        sensor_patch_flat = np.zeros(
            9 * COUNT_MARKER,
            dtype=np.float32
        )

        target_patch = np.zeros(9, dtype=np.float32)

        idx = 0

        for i in range(-1, 2):
            for j in range(-1, 2):

                nx = r + i
                ny = c + j

                true_val = int(grid_counts[nx, ny])
                alpha = grid_angles[nx, ny]

                plant_dx = np.sin(alpha)
                plant_dy = np.cos(alpha)

                alignment = abs(
                    dx * plant_dx +
                    dy * plant_dy
                )

                alignment = float(
                    np.clip(alignment, 0.0, 1.0)
                )

                noise_intensity = 1.0 - alignment

                sensor_dist = self.sensor.observe(
                    true_val,
                    noise_intensity
                ).astype(np.float32)

                prior = belief_map[nx, ny].copy()

                ce_before = -np.log(
                    prior[true_val] + 1e-9
                )

                posterior = prior * sensor_dist
                posterior /= (
                    posterior.sum() + 1e-9
                )

                ce_after = -np.log(
                    posterior[true_val] + 1e-9
                )

                ce_gain = ce_before - ce_after

                reliability_target = self._normalize_ce_gain(
                    ce_gain
                )

                belief_map[nx, ny] = posterior

                alignment_patch[idx] = alignment

                start = idx * COUNT_MARKER
                end = (idx + 1) * COUNT_MARKER

                sensor_patch_flat[start:end] = sensor_dist

                target_patch[idx] = reliability_target

                idx += 1

        movement_feature = np.array(
            [dx, dy],
            dtype=np.float32
        )

        step_input = np.concatenate([
            movement_feature,
            alignment_patch,
            sensor_patch_flat
        ]).astype(np.float32)

        return step_input, target_patch

    def __iter__(self):

        while True:

            field_data = self.generator.generate_field()

            grid_counts = field_data["true_counts"]
            grid_angles = field_data["true_angles"]

            size = self.generator.size

            for _ in range(self.episodes_per_field):

                r = np.random.randint(1, size - 1)
                c = np.random.randint(1, size - 1)

                belief_map = np.ones(
                    (size, size, COUNT_MARKER),
                    dtype=np.float32
                )

                belief_map /= COUNT_MARKER

                input_sequence = []
                final_target = None

                for _ in range(self.seq_len):

                    move = self.moves[
                        np.random.randint(
                            0,
                            len(self.moves)
                        )
                    ]

                    dx = int(move[0])
                    dy = int(move[1])

                    r = np.clip(
                        r + dx,
                        1,
                        size - 2
                    )

                    c = np.clip(
                        c + dy,
                        1,
                        size - 2
                    )

                    step_input, target_patch = self._build_step(
                        r=r,
                        c=c,
                        dx=dx,
                        dy=dy,
                        grid_counts=grid_counts,
                        grid_angles=grid_angles,
                        belief_map=belief_map
                    )

                    input_sequence.append(step_input)
                    final_target = target_patch

                input_tensor = torch.tensor(
                    np.array(input_sequence),
                    dtype=torch.float32
                )

                target_tensor = torch.tensor(
                    final_target,
                    dtype=torch.float32
                )

                yield input_tensor, target_tensor