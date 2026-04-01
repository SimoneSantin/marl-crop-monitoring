import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


# =========================
# ACTOR (CONDIVISO)
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc_out(x), dim=-1)


# =========================
# CRITIC (CENTRALIZZATO)
# =========================
class Critic(nn.Module):
    def __init__(self, global_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(global_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


# =========================
# MAPPO SHARED ACTOR
# =========================
class MAPPOPlannerMultiAgent:
    def __init__(self, obs_dim, action_dim, agents, field_size, num_classes,
                 lr, gamma, clip_eps, lam, epochs, mini_batch_size):

        self.agents = agents
        self.n_agents = len(agents)

        # Actor condiviso
        self.actor = Actor(obs_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic centralizzato
        input_dim = field_size * field_size + self.n_agents * 2
        self.critic = Critic(input_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lam = lam
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        # Buffer unico
        self.buffer = []

    # =========================
    # ACT
    # =========================
    def act(self, obs, agent_id):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        probs = self.actor(obs_tensor)
        dist = Categorical(probs)

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
    # GAE
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
    # UPDATE
    # =========================
    def update(self):
        if len(self.buffer) == 0:
            return

        all_obs = []
        all_actions = []
        all_log_probs = []
        all_adv = []
        all_global_states = []
        all_returns = []

        # ==========================================
        # PREPARA DATI PER OGNI AGENTE DAL BUFFER UNICO
        # ==========================================
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

            # critic values per questa traiettoria agente
            values = self.critic(global_states).squeeze().detach()

            # GAE per agente
            adv = self.compute_gae(rewards, values, dones)

            # normalizzazione advantage
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            returns = adv + values

            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(old_log_probs)
            all_adv.append(adv)
            all_global_states.append(global_states)
            all_returns.append(returns)

        if len(all_obs) == 0:
            return

        # ==========================================
        # CONCATENAZIONE GLOBALE
        # ==========================================
        obs = torch.cat(all_obs, dim=0)
        actions = torch.cat(all_actions, dim=0)
        old_log_probs = torch.cat(all_log_probs, dim=0)
        adv = torch.cat(all_adv, dim=0)

        global_states = torch.cat(all_global_states, dim=0)
        returns = torch.cat(all_returns, dim=0)

        dataset_size = obs.size(0)
        indices = torch.randperm(dataset_size)
        for _ in range(self.epochs):

            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]

                mb_obs = obs[batch_idx]
                mb_actions = actions[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_adv = adv[batch_idx]
                mb_global_states = global_states[batch_idx]
                mb_returns = returns[batch_idx]

                # =========================
                # CRITIC
                # =========================
                values = self.critic(mb_global_states).squeeze()

                critic_loss = F.smooth_l1_loss(values, mb_returns.detach())

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.critic_optimizer.step()

                # =========================
                # ACTOR
                # =========================
                probs = self.actor(mb_obs)
                dist = Categorical(probs)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_adv.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv.detach()

                ppo_loss = -torch.min(surr1, surr2).mean()
                actor_loss = ppo_loss - 0.01 * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.actor_optimizer.step()
        # =========================
        # RESET BUFFER
        # =========================
        self.buffer = []