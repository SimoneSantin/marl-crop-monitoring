import numpy as np


class Agent:

    def __init__(self, env, num_classes, agent_id, planner):

        self.env = env
        self.planner = planner
        self.agent_id = agent_id
        self.field_size = env.field_size
        self.num_classes = num_classes
        self.belief_map = None
        self.observed_mask = np.zeros((self.field_size, self.field_size))
        self.last_movement = np.array([0, 0], dtype=np.float32)
        # parametri Gaussian Field
        self.sigma = 2.5
        self.inference_radius = 5

        # precalcola kernel una volta sola nel costruttore
        self._precompute_kernel()

        self.reset()

    def _precompute_kernel(self):
        """
        Precalcola offset e pesi gaussiani per il kernel RBF.
        Vengono calcolati una volta sola e riusati ad ogni step
        per evitare di ricalcolarli 2000 volte per episodio.
        """
        r = self.inference_radius
        offsets = []
        weights = []

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                dist2 = dx * dx + dy * dy
                if dist2 <= r * r and dist2 > 0:  # cerchio, escludi cella centrale
                    w = np.exp(-dist2 / (2 * self.sigma ** 2))
                    offsets.append([dx, dy])
                    weights.append(w)

        self._kernel_offsets = np.array(offsets, dtype=np.int32)   # (K, 2)
        self._kernel_weights = np.array(weights, dtype=np.float32)  # (K,)

    def reset(self):
        self.belief_map = np.ones(
            (self.field_size, self.field_size, self.num_classes)
        ) / self.num_classes

        # reset observed mask ad ogni episodio
        self.observed_mask = np.zeros((self.field_size, self.field_size))

    def in_bounds(self, x, y):
        return (
            0 <= x < self.field_size
            and 0 <= y < self.field_size
        )

    def update_belief_patch(
        self,
        sensor_patch,
        alignment_patch,
        gamma=3.0,
        confidence_patch=None
    ):
        """
        Step 1: Bayesian update sulle celle direttamente osservate (patch 3x3).
                Usa alignment^gamma come reliability per pesare la likelihood.

        Step 2: Gaussian Field — propaga la belief delle celle osservate
                verso le celle non osservate vicine, con pesi che decadono
                gaussianamente con la distanza (kernel RBF precalcolato).
                Solo le celle NON osservate vengono aggiornate.
        """
        x, y = self.env.agent_pos[self.agent_id]
        H, W, C = self.belief_map.shape

        idx = 0
        observed_cells = []

        # =========================
        # STEP 1 — BAYESIAN UPDATE
        # =========================
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx = x + dx
                cy = y + dy

                if not self.in_bounds(cx, cy):
                    idx += 1
                    continue

                sensor_dist = sensor_patch[idx]
                alignment   = alignment_patch[idx]

                # reliability: quanto ci si fida di questa osservazione
                # alignment basso -> reliability bassa -> likelihood appiattita
                # base reliability dal movimento/alignment
                reliability = alignment ** gamma

                if confidence_patch is not None:
                    reliability = reliability * confidence_patch[idx]

                # clamp di sicurezza
                reliability = np.clip(reliability, 0.0, 1.0)

                # evita instabilità numerica
                reliability = np.clip(reliability, 0.0, 1.0)

                # likelihood modulata dalla confidenza
                likelihood = sensor_dist ** reliability
                likelihood /= (np.sum(likelihood) + 1e-9)
               

                # Bayesian update: prior * likelihood normalizzato
                old_belief     = self.belief_map[cx, cy]
                updated_belief = old_belief * likelihood
                updated_belief /= (np.sum(updated_belief) + 1e-9)

                self.belief_map[cx, cy] = updated_belief
                self.observed_mask[cx, cy] = 1
                observed_cells.append((cx, cy))

                idx += 1

        if len(observed_cells) == 0:
            return


    def choose_action(self, obs):
        if hasattr(self.planner, "act"):
            action, log_prob = self.planner.act(obs, self.agent_id)
            return action, log_prob
        else:
            raise ValueError("Planner interface not recognized")

    def get_prediction_map(self):
        return np.argmax(self.belief_map, axis=2)

    def compute_accuracy(self):
        pred_map = self.get_prediction_map()
        true_map = self.env.grid_counts
        return (pred_map == true_map).mean()