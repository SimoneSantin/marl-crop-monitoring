import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


# =========================
# ACTOR RICORRENTE
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.patch_dim = 27
        self.other_dim = obs_dim - self.patch_dim

        # CNN sulla patch 3x3 con 3 canali
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.cnn_out_dim = 16 * 3 * 3

        self.fc_other = nn.Sequential(
            nn.Linear(self.other_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc_combined = nn.Linear(self.cnn_out_dim + hidden_dim, hidden_dim)
        self.pre_lstm_norm = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, h):
        # x: (B, T, obs_dim)
        B, T, D = x.shape

        patch = x[:, :, :self.patch_dim]   # (B, T, 27)
        other = x[:, :, self.patch_dim:]   # (B, T, other_dim)

        patch = patch.contiguous().view(B * T, 3, 3, 3)   # (B*T, C, H, W)
        patch_feat = self.cnn(patch)

        other = other.contiguous().view(B * T, self.other_dim)
        other_feat = self.fc_other(other)

        combined = torch.cat([patch_feat, other_feat], dim=-1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.pre_lstm_norm(combined)

        combined = combined.view(B, T, self.hidden_dim)

        x, h = self.lstm(combined, h)
        logits = self.fc_out(x)

        return logits, h


# =========================
# CRITIC RICORRENTE
# =========================
class Critic(nn.Module):
    def __init__(self, field_size, n_agents, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.field_size = field_size
        self.n_agents = n_agents

        # -------------------------
        # CNN su visited_mask (1, H, W)
        # -------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, field_size, field_size)
            self.cnn_out_dim = self.cnn(dummy).shape[1]

        # -------------------------
        # Agent positions
        # -------------------------
        self.fc_agents = nn.Sequential(
            nn.Linear(n_agents * 2, hidden_dim),
            nn.ReLU()
        )

        # -------------------------
        # Fusione
        # -------------------------
        self.fc_combined = nn.Linear(self.cnn_out_dim + hidden_dim, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, h):
        # x: (B, T, global_dim)

        B, T, D = x.shape

        # -------------------------
        # SPLIT
        # -------------------------
        map_size = self.field_size * self.field_size

        visited = x[:, :, :map_size]
        agents = x[:, :, map_size:]

        # -------------------------
        # CNN
        # -------------------------
        visited = visited.view(B * T, 1, self.field_size, self.field_size)
        cnn_feat = self.cnn(visited)

        # -------------------------
        # Agents
        # -------------------------
        agents = agents.view(B * T, -1)
        agent_feat = self.fc_agents(agents)

        # -------------------------
        # Combine
        # -------------------------
        combined = torch.cat([cnn_feat, agent_feat], dim=-1)
        combined = F.relu(self.fc_combined(combined))

        combined = combined.view(B, T, self.hidden_dim)

        # -------------------------
        # LSTM
        # -------------------------
        x, h = self.lstm(combined, h)

        values = self.fc_out(x).squeeze(-1)

        return values, h

# =========================
# RECURRENT MAPPO
# =========================
class MAPPOPlannerMultiAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        agents,
        field_size,
        num_classes,
        lr,
        gamma,
        clip_eps,
        lam,
        epochs,
        mini_batch_size,
        hidden_dim=128,
        chunk_len=16
    ):
        self.agents = agents
        self.n_agents = len(agents)

        self.hidden_dim = hidden_dim
        self.chunk_len = chunk_len

        # hidden state online per l'actor (esecuzione step-by-step)
        self.actor_hidden_states = [
            (
                torch.zeros(1, 1, self.hidden_dim),  # h
                torch.zeros(1, 1, self.hidden_dim)   # c
            )
            for _ in range(self.n_agents)
        ]

        # actor e critic ricorrenti
        self.actor = Actor(obs_dim, action_dim, hidden_dim=self.hidden_dim)
        self.critic = Critic(
            field_size=field_size,
            n_agents=self.n_agents,
            hidden_dim=self.hidden_dim
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lam = lam
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        self.buffer = []

    # =========================
    # RESET HIDDEN STATES
    # =========================
    def reset_hidden_states(self):
        self.actor_hidden_states = [
            (
                torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim)
            )
            for _ in range(self.n_agents)
        ]
    # =========================
    # ACT
    # =========================
    def act(self, obs, agent_id):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        logits, h_new = self.actor(obs_tensor, self.actor_hidden_states[agent_id])
        self.actor_hidden_states[agent_id] = (
            h_new[0].detach(),
            h_new[1].detach()
        )

        logits = logits.squeeze(0).squeeze(0)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach()

    # =========================
    # STORE TRANSITION
    # =========================
    def store_transition(self, obs, action, log_prob, reward, global_state, agent_id, done):
        self.buffer.append({
            "obs": obs,
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "global_state": global_state,
            "agent_id": agent_id,
            "done": done
        })

    # =========================
    # GAE SU SINGOLA TRAIETTORIA
    # =========================
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    # =========================
    # COSTRUZIONE CHUNK
    # =========================
    def _build_chunks(self):
        chunks = []

        for agent_id in range(self.n_agents):
            agent_data = [b for b in self.buffer if b["agent_id"] == agent_id]
            if len(agent_data) == 0:
                continue

            obs = torch.tensor(
                np.array([b["obs"] for b in agent_data]),
                dtype=torch.float32
            )
            actions = torch.tensor(
                [b["action"] for b in agent_data],
                dtype=torch.long
            )
            old_log_probs = torch.stack(
                [b["log_prob"] for b in agent_data]
            ).detach()
            rewards = torch.tensor(
                [b["reward"] for b in agent_data],
                dtype=torch.float32
            )
            dones = torch.tensor(
                [b["done"] for b in agent_data],
                dtype=torch.float32
            )
            global_states = torch.tensor(
                np.array([b["global_state"] for b in agent_data]),
                dtype=torch.float32
            )

            # critic ricorrente: valori sull'intera traiettoria
            with torch.no_grad():
                gs_seq = global_states.unsqueeze(0)  # (1, T, global_dim)
                h0_v = (
                    torch.zeros(1, 1, self.hidden_dim),
                    torch.zeros(1, 1, self.hidden_dim)
                )
                values_seq, _ = self.critic(gs_seq, h0_v)
                values = values_seq.squeeze(0)  # (T,)
                
            raw_adv = self.compute_gae(rewards, values, dones)
            returns = raw_adv + values
            adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)

            T = obs.size(0)
            for start in range(0, T, self.chunk_len):
                end = min(start + self.chunk_len, T)

                chunks.append({
                    "obs": obs[start:end],                      # (L, obs_dim)
                    "actions": actions[start:end],             # (L,)
                    "old_log_probs": old_log_probs[start:end], # (L,)
                    "adv": adv[start:end],                     # (L,)
                    "returns": returns[start:end],             # (L,)
                    "global_states": global_states[start:end], # (L, global_dim)
                })

        return chunks

    # =========================
    # PADDING CHUNKS
    # =========================
    def _pad_batch(self, batch_chunks):
        max_len = max(c["obs"].size(0) for c in batch_chunks)
        batch_size = len(batch_chunks)

        obs_batch = []
        actions_batch = []
        old_log_probs_batch = []
        adv_batch = []
        returns_batch = []
        global_states_batch = []
        mask_batch = []

        for c in batch_chunks:
            seq_len = c["obs"].size(0)
            pad = max_len - seq_len

            obs_batch.append(F.pad(c["obs"], (0, 0, 0, pad)))
            actions_batch.append(F.pad(c["actions"], (0, pad), value=0))
            old_log_probs_batch.append(F.pad(c["old_log_probs"], (0, pad), value=0.0))
            adv_batch.append(F.pad(c["adv"], (0, pad), value=0.0))
            returns_batch.append(F.pad(c["returns"], (0, pad), value=0.0))
            global_states_batch.append(F.pad(c["global_states"], (0, 0, 0, pad)))

            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(pad, dtype=torch.bool)
            ])
            mask_batch.append(mask)

        obs_batch = torch.stack(obs_batch)                   # (B, L, obs_dim)
        actions_batch = torch.stack(actions_batch)           # (B, L)
        old_log_probs_batch = torch.stack(old_log_probs_batch)
        adv_batch = torch.stack(adv_batch)
        returns_batch = torch.stack(returns_batch)
        global_states_batch = torch.stack(global_states_batch)
        mask_batch = torch.stack(mask_batch)                 # (B, L)

        return (
            obs_batch,
            actions_batch,
            old_log_probs_batch,
            adv_batch,
            returns_batch,
            global_states_batch,
            mask_batch
        )

    # =========================
    # UPDATE
    # =========================
    def update(self):
        if len(self.buffer) == 0:
            return

        chunks = self._build_chunks()
        if len(chunks) == 0:
            return

        for _ in range(self.epochs):
            np.random.shuffle(chunks)

            for start in range(0, len(chunks), self.mini_batch_size):
                batch_chunks = chunks[start:start + self.mini_batch_size]

                (
                    obs_batch,
                    actions_batch,
                    old_log_probs_batch,
                    adv_batch,
                    returns_batch,
                    global_states_batch,
                    mask_batch
                ) = self._pad_batch(batch_chunks)

                batch_size = obs_batch.size(0)

                # -------------------------
                # Critic
                # -------------------------
                h0_v = (
                    torch.zeros(1, batch_size, self.hidden_dim),
                    torch.zeros(1, batch_size, self.hidden_dim)
                )
                values_pred, _ = self.critic(global_states_batch, h0_v)

                critic_loss = F.smooth_l1_loss(
                    values_pred[mask_batch],
                    returns_batch[mask_batch].detach()
                )

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.critic_optimizer.step()

                # -------------------------
                # Actor
                # -------------------------
                h0_pi = (
                    torch.zeros(1, batch_size, self.hidden_dim),
                    torch.zeros(1, batch_size, self.hidden_dim)
                )
                logits, _ = self.actor(obs_batch, h0_pi)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_batch)   # (B, L)
                entropy = dist.entropy()                       # (B, L)

                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                surr1 = ratio * adv_batch.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_batch.detach()

                actor_loss = -torch.min(surr1, surr2)
                actor_loss = actor_loss[mask_batch].mean()

                entropy_loss = entropy[mask_batch].mean()

                total_actor_loss = actor_loss - 0.01 * entropy_loss

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()

        self.buffer = []