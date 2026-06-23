import numpy as np
import torch
import torch.nn.functional as F
from LSTM.lstm_model import CellObserverLSTM


# ─────────────────────────────────────────────
# FUNZIONI STANDALONE
# ─────────────────────────────────────────────

def make_gaussian_kernel_torch(radius, sigma, device):
    size = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist2 = xx ** 2 + yy ** 2
    kernel = torch.exp(-dist2 / (2 * sigma ** 2))
    kernel[radius, radius] = 0.0
    kernel = torch.where(dist2 <= radius ** 2, kernel, torch.zeros_like(kernel))
    return kernel


@torch.no_grad()
def update_belief_patch_gaussian_evidence_torch(
    belief_map, observed_mask, sensor_patch, alignment_patch,
    agent_pos, gaussian_kernel, gamma=3.0, alpha=0.1, eps=1e-9, confidence_patch=None
):
    device = belief_map.device
    H, W, C = belief_map.shape
    x, y = agent_pos

    delta_map   = torch.zeros((1, C, H, W), device=device)
    source_mask = torch.zeros((1, 1, H, W), device=device)

    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            cx = x + dx
            cy = y + dy
            if cx < 0 or cx >= H or cy < 0 or cy >= W:
                idx += 1
                continue

            sensor_dist = sensor_patch[idx].to(device)
            alignment   = alignment_patch[idx].to(device)

            reliability = alignment ** gamma
            if confidence_patch is not None:
                confidence  = confidence_patch[idx].to(device)
                reliability = reliability * confidence
            reliability = torch.clamp(reliability, 0.0, 1.0)

            likelihood = sensor_dist ** reliability
            likelihood = likelihood / (likelihood.sum() + eps)

            old_belief     = belief_map[cx, cy].clone()
            updated_belief = old_belief * likelihood
            updated_belief = updated_belief / (updated_belief.sum() + eps)

            belief_map[cx, cy]    = updated_belief
            observed_mask[cx, cy] = True

            delta_log = (torch.log(updated_belief + eps)
                         - torch.log(old_belief + eps))

            delta_map[0, :, cx, cy]   = delta_log
            source_mask[0, 0, cx, cy] = 1.0
            idx += 1

    if source_mask.sum() == 0:
        return belief_map, observed_mask

    K       = gaussian_kernel.shape[0]
    padding = K // 2
    kernel_c = gaussian_kernel.view(1, 1, K, K).repeat(C, 1, 1, 1)

    propagated_delta = F.conv2d(delta_map, kernel_c, padding=padding, groups=C)
    propagated_weight = F.conv2d(
        source_mask, gaussian_kernel.view(1, 1, K, K), padding=padding
    )
    propagated_delta  = propagated_delta / (propagated_weight + eps)
    propagated_delta  = propagated_delta[0].permute(1, 2, 0)
    propagated_weight = propagated_weight[0, 0]

    target_mask = (propagated_weight > 0) & (~observed_mask)

    if target_mask.any():
        old_target_belief = belief_map[target_mask]
        log_belief        = torch.log(old_target_belief + eps)
        log_belief        = log_belief + alpha * propagated_delta[target_mask]
        log_belief        = log_belief - log_belief.max(dim=1, keepdim=True).values
        new_target_belief = torch.exp(log_belief)
        new_target_belief = new_target_belief / (
            new_target_belief.sum(dim=1, keepdim=True) + eps
        )
        belief_map[target_mask] = new_target_belief

    return belief_map, observed_mask


# ─────────────────────────────────────────────
# CLASSE AGENT
# ─────────────────────────────────────────────

class Agent:

    def __init__(
        self, env, num_classes, agent_id, planner, device='cpu',
        use_gaussian=True, use_lstm=False,
        use_cell_lstm=False,
        cell_lstm_path="./LSTM/models/cell_observer_lstm.pth",
        cell_lstm_hidden=64,
        use_random_policy=False,
    ):
        self.env          = env
        self.planner      = planner
        self.agent_id     = agent_id
        self.field_size   = env.field_size
        self.num_classes  = num_classes
        self.device       = torch.device(device)

        self.sigma            = 2.5
        self.inference_radius = 5
        self.alpha            = 0.1

        self.last_movement = np.array([0.0, 0.0], dtype=np.float32)

        self.use_gaussian   = use_gaussian
        self.use_lstm       = use_lstm
        self.use_cell_lstm  = use_cell_lstm
        self.cell_lstm_hidden = cell_lstm_hidden
        self.use_random_policy = use_random_policy
        # ── CellObserverLSTM ──────────────────────────────────────────────
        if self.use_cell_lstm:
            self.cell_lstm = CellObserverLSTM(
                num_classes=self.num_classes,
                hidden_size=cell_lstm_hidden,
            ).to(self.device)
            self.cell_lstm.load_state_dict(
                torch.load(cell_lstm_path, map_location=self.device)
            )
            self.cell_lstm.eval()

        # ── Gaussian kernel ───────────────────────────────────────────────
        if self.use_gaussian:
            self.gaussian_kernel = make_gaussian_kernel_torch(
                radius=self.inference_radius,
                sigma=self.sigma,
                device=self.device,
            )

        self.belief_map    = None
        self.observed_mask = None
        self.reset()

    # ─────────────────────────────────────────
    def reset(self):
        self.belief_map = torch.ones(
            (self.field_size, self.field_size, self.num_classes),
            device=self.device,
        ) / self.num_classes

        self.observed_mask = torch.zeros(
            (self.field_size, self.field_size),
            dtype=torch.bool,
            device=self.device,
        )

        # hidden state per cella — azzerati a ogni episodio
        if self.use_cell_lstm:
            self.cell_h = torch.zeros(
                (self.field_size, self.field_size, self.cell_lstm_hidden),
                device=self.device,
            )
            self.cell_c = torch.zeros(
                (self.field_size, self.field_size, self.cell_lstm_hidden),
                device=self.device,
            )

    # ─────────────────────────────────────────
    def in_bounds(self, x, y):
        return 0 <= x < self.field_size and 0 <= y < self.field_size

    # ─────────────────────────────────────────
    def update_belief_patch(self, sensor_patch, alignment_patch,
                            confidence_patch=None, gamma=3.0):
        """Dispatcher: sceglie il meccanismo di update corretto."""
        x, y = self.env.agent_pos[self.agent_id]
    
        sensor_tensor = torch.tensor(
            sensor_patch, dtype=torch.float32, device=self.device
        )
        alignment_tensor = torch.tensor(
            alignment_patch, dtype=torch.float32, device=self.device
        )
        confidence_tensor = (
            torch.tensor(confidence_patch, dtype=torch.float32, device=self.device)
            if confidence_patch is not None else None
        )

        # ── priorità 1: CellObserverLSTM ─────────────────────────────────
        if self.use_cell_lstm:
            return self.update_belief_patch_cell_lstm(
                sensor_patch=sensor_tensor,
                alignment_patch=alignment_tensor,
            )

        # ── priorità 2: Bayesian + Gaussian Field ─────────────────────────
        if self.use_gaussian:
            self.belief_map, self.observed_mask = \
                update_belief_patch_gaussian_evidence_torch(
                    belief_map       = self.belief_map,
                    observed_mask    = self.observed_mask,
                    sensor_patch     = sensor_tensor,
                    alignment_patch  = alignment_tensor,
                    agent_pos        = (x, y),
                    gaussian_kernel  = self.gaussian_kernel,
                    gamma            = gamma,
                    alpha            = self.alpha,
                    confidence_patch = confidence_tensor,
                )
            return self.belief_map, self.observed_mask

        # ── priorità 3: solo Bayesian ─────────────────────────────────────
        idx = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx = x + dx
                cy = y + dy
                if not self.in_bounds(cx, cy):
                    idx += 1
                    continue

                sensor_dist = sensor_tensor[idx]
                alignment   = alignment_tensor[idx]
                reliability = alignment ** gamma

                if self.use_lstm and confidence_tensor is not None:
                    reliability = reliability * confidence_tensor[idx]

                reliability = torch.clamp(reliability, 0.0, 1.0)
                likelihood  = sensor_dist ** reliability
                likelihood  = likelihood / (likelihood.sum() + 1e-9)

                old_belief = self.belief_map[cx, cy]
                updated    = old_belief * likelihood
                updated    = updated / (updated.sum() + 1e-9)

                self.belief_map[cx, cy]    = updated
                self.observed_mask[cx, cy] = True
                idx += 1

    # ─────────────────────────────────────────
    @torch.no_grad()
    def update_belief_patch_cell_lstm(self, sensor_patch, alignment_patch, eps=1e-9):
        """
        Aggiorna la belief usando il CellObserverLSTM per le celle osservate
        e propaga il delta_log via Gaussian Field alle celle non osservate.

        sensor_patch e alignment_patch sono già tensori torch sul device corretto.
        """
        x, y = self.env.agent_pos[self.agent_id]

        # sensor_patch arriva già come tensore dal dispatcher
        sensor_t = sensor_patch.to(self.device)   # (9, K)
        align_t  = alignment_patch.to(self.device) # (9,)

        # raccogli celle valide della patch e i loro input LSTM
        coords = []
        inputs = []
        idx = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx, cy = x + dx, y + dy
                if 0 <= cx < self.field_size and 0 <= cy < self.field_size:
                    inp = torch.cat([sensor_t[idx], align_t[idx].view(1)])
                    coords.append((cx, cy))
                    inputs.append(inp)
                idx += 1

        if len(coords) == 0:
            return self.belief_map, self.observed_mask

        x_in = torch.stack(inputs, dim=0)  # (M, K+1)

        # recupera hidden state delle celle osservate
        cxs = torch.tensor([p[0] for p in coords], device=self.device)
        cys = torch.tensor([p[1] for p in coords], device=self.device)
        h_prev = self.cell_h[cxs, cys]  # (M, hidden)
        c_prev = self.cell_c[cxs, cys]  # (M, hidden)

        # un passo LSTM → nuova distribuzione di belief per ogni cella
        logits, h_new, c_new = self.cell_lstm.step(x_in, h_prev, c_prev)
        new_belief = torch.softmax(logits, dim=-1)  # (M, K)
        self.cell_h[cxs, cys] = h_new
        self.cell_c[cxs, cys] = c_new

        old_belief = self.belief_map[cxs, cys]
        delta_log  = torch.log(new_belief + eps) - torch.log(old_belief + eps)

        self.belief_map[cxs, cys]    = new_belief
        self.observed_mask[cxs, cys] = True

        # ── Gaussian Field: propaga il delta_log alle celle non osservate ──
        if self.use_gaussian:
            C  = self.num_classes
            H  = self.field_size
            W  = self.field_size

            delta_map   = torch.zeros((1, C, H, W), device=self.device)
            source_mask = torch.zeros((1, 1, H, W), device=self.device)
            delta_map[0, :, cxs, cys]   = delta_log.T
            source_mask[0, 0, cxs, cys] = 1.0

            K_sz    = self.gaussian_kernel.shape[0]
            padding = K_sz // 2
            kernel_c = self.gaussian_kernel.view(1, 1, K_sz, K_sz).repeat(C, 1, 1, 1)

            propagated_delta  = F.conv2d(delta_map, kernel_c,
                                         padding=padding, groups=C)
            propagated_weight = F.conv2d(source_mask,
                                         self.gaussian_kernel.view(1, 1, K_sz, K_sz),
                                         padding=padding)
            propagated_delta  = propagated_delta / (propagated_weight + eps)
            propagated_delta  = propagated_delta[0].permute(1, 2, 0)
            propagated_weight = propagated_weight[0, 0]

            target_mask = (propagated_weight > 0) & (~self.observed_mask)
            if target_mask.any():
                old_t = self.belief_map[target_mask]
                log_b = torch.log(old_t + eps) + self.alpha * propagated_delta[target_mask]
                log_b = log_b - log_b.max(dim=1, keepdim=True).values
                new_t = torch.exp(log_b)
                new_t = new_t / (new_t.sum(dim=1, keepdim=True) + eps)
                self.belief_map[target_mask] = new_t

        return self.belief_map, self.observed_mask

    # ─────────────────────────────────────────
    def choose_action(self, obs):
        if self.use_random_policy:
            action_dim = self.env.action_space.n
            action     = np.random.randint(0, action_dim)
            log_prob   = torch.tensor(
                -np.log(action_dim),
                dtype=torch.float32,
                device=self.device
            )
            return action, log_prob

        if hasattr(self.planner, "act"):
            action, log_prob = self.planner.act(obs, self.agent_id)
            return action, log_prob
        raise ValueError("Planner interface not recognized")
    
    def get_prediction_map(self):
        return self.belief_map.argmax(dim=-1).cpu().numpy()

    def compute_accuracy(self):
        pred_map = self.get_prediction_map()
        true_map = self.env.grid_counts
        return float((pred_map == true_map).mean())