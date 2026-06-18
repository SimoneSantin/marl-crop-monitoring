import torch
import numpy as np
from torch.utils.data import IterableDataset

from utils.constants import COUNT_MARKER
from env.sensor import ScalarSensor


class ProceduralPatchDataset(IterableDataset):
    """
    Dataset per il training dell'LSTM di reliability.

    Differenze chiave rispetto alla versione precedente:
    1. L'agente compie una traiettoria LOCALE coerente (random walk in una
       finestra ristretta), così le patch si sovrappongono e la stessa cella
       viene osservata più volte da DIREZIONI di movimento diverse.
    2. La belief si accumula realmente su ogni cella attraverso le osservazioni
       ripetute, riproducendo le condizioni di deployment.
    3. L'input contiene due feature temporali aggiuntive per cella:
       - visit_count: quante volte la cella è stata osservata finora
       - max_alignment: il massimo alignment con cui è stata osservata finora
       Queste danno all'LSTM informazione che il singolo alignment^gamma non ha.
    4. Il target è il CE gain dell'ULTIMA osservazione rispetto alla belief
       accumulata fino a quel momento: misura quanto l'osservazione corrente
       aggiunge, tenendo conto della storia.
    """

    def __init__(
        self,
        generator,
        seq_len=8,
        episodes_per_field=50,
        local_window=2,
    ):
        super().__init__()

        self.generator = generator
        self.seq_len = seq_len
        self.episodes_per_field = episodes_per_field
        self.local_window = local_window  # raggio della finestra di movimento locale

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
        belief_map,
        visit_count_grid,
        max_align_grid,
    ):
        """
        Costruisce l'input di un singolo timestep e il target.

        visit_count_grid e max_align_grid sono mappe (size, size) che tracciano
        lo stato temporale di ogni cella PRIMA di questa osservazione.
        """
        alignment_patch = np.zeros(9, dtype=np.float32)
        sensor_patch_flat = np.zeros(9 * COUNT_MARKER, dtype=np.float32)
        visit_count_patch = np.zeros(9, dtype=np.float32)
        max_align_patch = np.zeros(9, dtype=np.float32)
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

                alignment = abs(dx * plant_dx + dy * plant_dy)
                alignment = float(np.clip(alignment, 0.0, 1.0))

                noise_intensity = 1.0 - alignment

                sensor_dist = self.sensor.observe(
                    true_val, noise_intensity
                ).astype(np.float32)

                # --- feature temporali PRIMA dell'update (stato accumulato) ---
                vc = visit_count_grid[nx, ny]
                # normalizziamo il conteggio per seq_len (cap a 1.0)
                visit_count_patch[idx] = min(vc / self.seq_len, 1.0)
                max_align_patch[idx] = max_align_grid[nx, ny]

                # --- belief accumulata e CE gain dell'osservazione corrente ---
                prior = belief_map[nx, ny].copy()
                ce_before = -np.log(prior[true_val] + 1e-9)

                posterior = prior * sensor_dist
                posterior /= (posterior.sum() + 1e-9)

                ce_after = -np.log(posterior[true_val] + 1e-9)
                ce_gain = ce_before - ce_after

                reliability_target = self._normalize_ce_gain(ce_gain)

                # aggiorna belief e stato temporale accumulato
                belief_map[nx, ny] = posterior
                visit_count_grid[nx, ny] = vc + 1.0
                max_align_grid[nx, ny] = max(max_align_grid[nx, ny], alignment)

                # riempi i vettori di input/target
                alignment_patch[idx] = alignment
                start = idx * COUNT_MARKER
                end = (idx + 1) * COUNT_MARKER
                sensor_patch_flat[start:end] = sensor_dist
                target_patch[idx] = reliability_target

                idx += 1

        step_input = np.concatenate([
            alignment_patch,        # 9
            sensor_patch_flat,      # 9 * K
            visit_count_patch,      # 9
            max_align_patch,        # 9
        ]).astype(np.float32)

        return step_input, target_patch

    def __iter__(self):
        while True:
            field_data = self.generator.generate_field()
            grid_counts = field_data["true_counts"]
            grid_angles = field_data["true_angles"]
            size = self.generator.size

            for _ in range(self.episodes_per_field):
                # centro della finestra locale
                cr = np.random.randint(1 + self.local_window, size - 1 - self.local_window)
                cc = np.random.randint(1 + self.local_window, size - 1 - self.local_window)
                r, c = cr, cc

                belief_map = np.ones((size, size, COUNT_MARKER), dtype=np.float32)
                belief_map /= COUNT_MARKER

                # stato temporale accumulato per cella
                visit_count_grid = np.zeros((size, size), dtype=np.float32)
                max_align_grid = np.zeros((size, size), dtype=np.float32)

                input_sequence = []
                final_target = None

                for _ in range(self.seq_len):
                    move = self.moves[np.random.randint(0, len(self.moves))]
                    dx = int(move[0])
                    dy = int(move[1])

                    # movimento confinato alla finestra locale attorno al centro:
                    # garantisce sovrapposizione delle patch e ri-osservazione
                    # delle stesse celle da direzioni diverse
                    r = int(np.clip(r + dx, cr - self.local_window, cr + self.local_window))
                    c = int(np.clip(c + dy, cc - self.local_window, cc + self.local_window))
                    # clip anche ai bordi della griglia per sicurezza
                    r = int(np.clip(r, 1, size - 2))
                    c = int(np.clip(c, 1, size - 2))

                    step_input, target_patch = self._build_step(
                        r=r, c=c, dx=dx, dy=dy,
                        grid_counts=grid_counts,
                        grid_angles=grid_angles,
                        belief_map=belief_map,
                        visit_count_grid=visit_count_grid,
                        max_align_grid=max_align_grid,
                    )

                    input_sequence.append(step_input)
                    final_target = target_patch

                input_tensor = torch.tensor(
                    np.array(input_sequence), dtype=torch.float32
                )
                target_tensor = torch.tensor(final_target, dtype=torch.float32)

                yield input_tensor, target_tensor