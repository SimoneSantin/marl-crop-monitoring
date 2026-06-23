import time
import numpy as np
import torch
from collections import deque

from utils.constants import COUNT_MARKER
from LSTM.lstm_model import CellObserverLSTM


class MAPPOTrainer:
    def __init__(self, env, planner, num_episodes, reward_weights, config,
                 reliability_model_path="./LSTM/models/patch_reliability_model_v3.pth",
                 reliability_seq_len=8,
                 reliability_hidden_size=128,
                 reliability_num_layers=1):
        self.env            = env
        self.planner        = planner
        self.num_episodes   = num_episodes
        self.reward_weights = reward_weights
        self.config         = config
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"

        self.episode_accuracy_traces = {}
        self.visit_heatmap = np.zeros(
            (self.env.field_size, self.env.field_size), dtype=np.float32
        )
        self.last_episode_path = None
        self.accuracy_history  = []

        self.reliability_seq_len = reliability_seq_len

        # ── carica il modello di reliability SOLO se serve ────────────────
        # use_cell_lstm=True → l'LSTM è dentro l'Agent, qui non serve nulla
        # use_lstm=True      → serve il vecchio NetObsReliability
        # altrimenti         → nessun modello
        use_lstm      = config.get("use_lstm", False)
        use_cell_lstm = config.get("use_cell_lstm", False)

        self.reliability_model = None
        if use_lstm and not use_cell_lstm:
            self.reliability_model = CellObserverLSTM(
                num_classes=COUNT_MARKER,
                hidden_size=reliability_hidden_size,
                num_layers=reliability_num_layers,
                dropout=0.2,
            ).to(self.device)
            self.reliability_model.load_state_dict(
                torch.load(reliability_model_path, map_location=self.device)
            )
            self.reliability_model.eval()

        self.agent_patch_histories = [
            deque(maxlen=self.reliability_seq_len)
            for _ in range(self.env.num_agents)
        ]
        self.agent_visit_count = [
            np.zeros((self.env.field_size, self.env.field_size), dtype=np.float32)
            for _ in range(self.env.num_agents)
        ]
        self.agent_max_align = [
            np.zeros((self.env.field_size, self.env.field_size), dtype=np.float32)
            for _ in range(self.env.num_agents)
        ]

    # ─────────────────────────────────────────────────────────────────────
    def reset_reliability_histories(self):
        self.agent_patch_histories = [
            deque(maxlen=self.reliability_seq_len)
            for _ in range(self.env.num_agents)
        ]
        self.agent_visit_count = [
            np.zeros((self.env.field_size, self.env.field_size), dtype=np.float32)
            for _ in range(self.env.num_agents)
        ]
        self.agent_max_align = [
            np.zeros((self.env.field_size, self.env.field_size), dtype=np.float32)
            for _ in range(self.env.num_agents)
        ]

    def build_reliability_step_feature(self, obs_i, agent_id, agent_pos, alignment_patch):
        align_end   = 9
        sensor_end  = align_end + 9 * COUNT_MARKER

        alignment_patch_arr = obs_i[0:align_end].astype(np.float32)
        sensor_patch_flat   = obs_i[align_end:sensor_end].astype(np.float32)

        x, y = agent_pos
        visit_count_patch = np.zeros(9, dtype=np.float32)
        max_align_patch   = np.zeros(9, dtype=np.float32)

        vc_grid = self.agent_visit_count[agent_id]
        ma_grid = self.agent_max_align[agent_id]

        idx = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                    vc = vc_grid[nx, ny]
                    visit_count_patch[idx] = min(vc / self.reliability_seq_len, 1.0)
                    max_align_patch[idx]   = ma_grid[nx, ny]
                    align_val = float(alignment_patch_arr[idx])
                    vc_grid[nx, ny] = vc + 1.0
                    ma_grid[nx, ny] = max(ma_grid[nx, ny], align_val)
                idx += 1

        return np.concatenate([
            alignment_patch_arr, sensor_patch_flat,
            visit_count_patch, max_align_patch,
        ]).astype(np.float32)

    def predict_patch_confidence(self, agent_id, step_feature):
        history = self.agent_patch_histories[agent_id]
        history.append(step_feature)
        seq = list(history)
        while len(seq) < self.reliability_seq_len:
            seq.insert(0, np.zeros_like(step_feature, dtype=np.float32))
        seq_array = np.stack(seq, axis=0)
        x = torch.tensor(seq_array, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.reliability_model(x)
            confidence_patch = outputs["pred_confidence_patch"][0].cpu().numpy()
        return confidence_patch.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    def compute_global_accuracy(self):
        return float(np.mean([a.compute_accuracy() for a in self.planner.agents]))

    def compute_observed_accuracy(self):
        """Celle osservate dalla patch (vecchia 'visited' allargata)."""
        observed = self.env.observed_mask.astype(bool)
        if observed.sum() == 0:
            return 0.0
        true_map = self.env.grid_counts
        return float(np.mean([
            (a.get_prediction_map()[observed] == true_map[observed]).mean()
            for a in self.planner.agents
        ]))

    def compute_inferred_accuracy(self):
        """Celle MAI osservate da nessun agente — vera generalizzazione spaziale."""
        inferred = ~self.env.observed_mask.astype(bool)
        if inferred.sum() == 0:
            return 0.0
        true_map = self.env.grid_counts
        return float(np.mean([
            (a.get_prediction_map()[inferred] == true_map[inferred]).mean()
            for a in self.planner.agents
        ]))
    # ─────────────────────────────────────────────────────────────────────
    def train(self):
        alignment_history         = []
        rewards_history           = []
        coverage_history          = []
        episode_lengths           = []
        collisions_history        = []
        terms_history             = []
        visited_accuracy_history  = []
        unvisited_accuracy_history= []
        checkpoint_episodes = [0] + sorted(set([
            max(0, int((k + 1) * self.num_episodes / 4) - 1)
            for k in range(4)
        ]))
        episode_accuracy_traces = {}

        # leggi i flag UNA VOLTA sola
        use_belief        = self.config.get("use_belief", True)
        use_oracle        = self.config.get("use_oracle_confidence", False)
        use_lstm          = self.config.get("use_lstm", False)
        use_cell_lstm     = self.config.get("use_cell_lstm", False)

        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            done   = False

            track_this = episode in checkpoint_episodes
            if track_this:
                episode_accuracy_trace = []

            for agent in self.planner.agents:
                agent.reset()

            self.planner.reset_hidden_states()
            self.reset_reliability_histories()

            episode_paths      = [[] for _ in range(self.env.num_agents)]
            episode_reward     = 0.0
            episode_collisions = 0
            episode_alignment  = 0
            steps              = 0
            episode_terms = {k: 0.0 for k in
                             ["new_cells","collisions","step","alignment","belief","accuracy"]}

            while not done:
                if steps % 500 == 0:
                    print(f"Episode {episode} | Step {steps} | Reward {episode_reward:.3f}")

                actions, log_probs, current_obs_for_buffer = [], [], []

                shared_uncertainty_map = self.compute_shared_uncertainty_map()
                shared_uncertainty_coarse = self.downsample_uncertainty_map(
                    shared_uncertainty_map, coarse_size=4
                )
                global_state = np.concatenate([
                    self.env.visited_mask.flatten().astype(np.float32),
                    shared_uncertainty_map.flatten().astype(np.float32),
                    (np.array(self.env.agent_pos, dtype=np.float32) /
                     self.env.field_size).flatten()
                ]).astype(np.float32)

                # ── 1. Action selection ───────────────────────────────────
                for agent_id, agent in enumerate(self.planner.agents):
                    enriched = self.enrich_obs_with_belief(
                        obs[agent_id], agent, shared_uncertainty_coarse
                    )
                    action, log_prob = agent.choose_action(enriched)
                    actions.append(action)
                    log_probs.append(log_prob)
                    current_obs_for_buffer.append(enriched)

                # ── 2. Env step ───────────────────────────────────────────
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated

                # ── 3. Belief update ──────────────────────────────────────
                local_accuracy_bonus_per_agent = []

                for i, agent in enumerate(self.planner.agents):
                    x, y = self.env.agent_pos[i]

                    old_beliefs = []
                    if use_belief:
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.env.field_size and \
                                   0 <= ny < self.env.field_size:
                                    old_beliefs.append(
                                        agent.belief_map[nx, ny].cpu().numpy().copy()
                                    )

                    obs_i           = next_obs[i]
                    alignment_patch = obs_i[:9]
                    sensor_patch    = obs_i[9:9 + 9 * COUNT_MARKER].reshape(9, COUNT_MARKER)

                    # ── calcolo confidence ────────────────────────────────
                    if use_oracle:
                        oracle_confidence = np.zeros(9, dtype=np.float32)
                        idx_o = 0; old_idx = 0
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.env.field_size and \
                                   0 <= ny < self.env.field_size:
                                    tc = self.env.grid_counts[nx, ny]
                                    ob = old_beliefs[old_idx]
                                    ce_b = -np.log(ob[tc] + 1e-9)
                                    post = ob * sensor_patch[idx_o]
                                    post /= (post.sum() + 1e-9)
                                    ce_a = -np.log(post[tc] + 1e-9)
                                    oracle_confidence[idx_o] = (
                                        np.clip(ce_b - ce_a, -1.0, 1.0) + 1.0
                                    ) / 2.0
                                    old_idx += 1
                                idx_o += 1
                        confidence_patch = oracle_confidence

                    elif use_lstm and not use_cell_lstm:
                        # vecchio modello di reliability (NetObsReliability)
                        step_feature = self.build_reliability_step_feature(
                            obs_i, agent_id=i, agent_pos=(x, y),
                            alignment_patch=alignment_patch,
                        )
                        confidence_patch = self.predict_patch_confidence(i, step_feature)

                    else:
                        # use_cell_lstm=True → confidence gestita dentro Agent
                        # nessun lstm → nessuna confidence
                        confidence_patch = None

                    # ── aggiornamento belief ──────────────────────────────
                    if use_belief:
                        if use_oracle:
                            fake_alignment = np.ones(9, dtype=np.float32)
                            agent.update_belief_patch(
                                sensor_patch=sensor_patch,
                                alignment_patch=fake_alignment,
                                confidence_patch=confidence_patch,
                                gamma=3.0,
                            )
                        else:
                            agent.update_belief_patch(
                                sensor_patch=sensor_patch,
                                alignment_patch=alignment_patch,
                                confidence_patch=confidence_patch,
                                gamma=3.0,
                            )

                    # ── CE gain reward ────────────────────────────────────
                    ce_gains = []
                    old_idx  = 0

                    if use_belief:   # ← AGGIUNGI QUESTA GUARDIA
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.env.field_size and \
                                0 <= ny < self.env.field_size:
                                    ob = old_beliefs[old_idx]
                                    nb = agent.belief_map[nx, ny].cpu().numpy()
                                    tc = self.env.grid_counts[nx, ny]
                                    ce_gains.append(
                                        -np.log(ob[tc] + 1e-9) +
                                        np.log(nb[tc] + 1e-9)
                                    )
                                    old_idx += 1

                    patch_ce_gain = float(np.mean(ce_gains)) if ce_gains else 0.0
                    local_accuracy_bonus_per_agent.append(
                        self.reward_weights["accuracy"] * patch_ce_gain
                    )

                shaped_rewards = [
                    rewards[i] + local_accuracy_bonus_per_agent[i]
                    for i in range(self.env.num_agents)
                ]

                if track_this:
                    current_accuracy = np.mean(
                        [a.compute_accuracy() for a in self.planner.agents]
                    )
                    episode_accuracy_trace.append(current_accuracy)

                # ── 4. Store in buffer ────────────────────────────────────
                for agent_id in range(self.env.num_agents):
                    self.planner.store_transition(
                        obs=current_obs_for_buffer[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs[agent_id],
                        reward=shaped_rewards[agent_id],
                        global_state=global_state,
                        agent_id=agent_id,
                        done=done,
                    )

                obs = next_obs
                episode_reward     += np.mean(shaped_rewards)
                episode_collisions += info["collisions"]
                episode_alignment  += info["reward_terms"]["alignment"]
                episode_terms["collisions"] += info["reward_terms"]["collisions"]
                episode_terms["alignment"]  += info["reward_terms"]["alignment"]
                episode_terms["accuracy"]   += float(np.mean(local_accuracy_bonus_per_agent))
                steps += 1
                for i, pos in enumerate(self.env.agent_pos):
                    episode_paths[i].append(tuple(pos))

            # ── MAPPO update ──────────────────────────────────────────────
            if not self.config.get("use_random_policy", False):
                self.planner.update()

            for k in episode_terms:
                episode_terms[k] /= max(steps, 1)
            terms_history.append(episode_terms.copy())

            if episode == self.num_episodes - 1:
                self.last_episode_paths = episode_paths
            if track_this:
                episode_accuracy_traces[episode] = episode_accuracy_trace

            final_accuracy          = self.compute_global_accuracy()
            final_visited_accuracy  = self.compute_observed_accuracy()
            final_unvisited_accuracy= self.compute_inferred_accuracy()

            unvisited_accuracy_history.append(final_unvisited_accuracy)
            self.accuracy_history.append(final_accuracy)
            visited_accuracy_history.append(final_visited_accuracy)
            rewards_history.append(episode_reward)
            alignment_history.append(episode_alignment / max(steps, 1))
            coverage_history.append(np.mean(self.env.visited_mask))
            episode_lengths.append(steps)
            collisions_history.append(episode_collisions)

        return {
            "rewards":            rewards_history,
            "coverage":           coverage_history,
            "lengths":            episode_lengths,
            "collisions":         collisions_history,
            "episode_paths":      self.last_episode_paths,
            "accuracy":           self.accuracy_history,
            "visited_accuracy":   visited_accuracy_history,
            "unvisited_accuracy": unvisited_accuracy_history,
            "terms":              terms_history,
            "alignment":          alignment_history,
            "accuracy_traces":    episode_accuracy_traces,
        }

    # ─────────────────────────────────────────────────────────────────────
    def enrich_obs_with_belief(self, obs, agent, shared_uncertainty_coarse):
        x, y = self.env.agent_pos[agent.agent_id]
        entropy_patch = np.zeros((3, 3), dtype=np.float32)
        agents_patch  = np.zeros((3, 3), dtype=np.float32)

        for i in range(-1, 2):
            for j in range(-1, 2):
                px, py = i + 1, j + 1
                nx, ny = x + i, y + j
                if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                    belief  = agent.belief_map[nx, ny].cpu().numpy()
                    entropy = -np.sum(belief * np.log(belief + 1e-9))
                    entropy /= np.log(agent.num_classes)
                    entropy_patch[px, py] = entropy
                    for other_id, (ax, ay) in enumerate(self.env.agent_pos):
                        if other_id != agent.agent_id and ax == nx and ay == ny:
                            agents_patch[px, py] = 1.0
                            break

        local_patch_flat = np.stack([entropy_patch, agents_patch]).flatten().astype(np.float32)
        return np.concatenate([
            local_patch_flat, obs.astype(np.float32), shared_uncertainty_coarse
        ]).astype(np.float32)

    def downsample_uncertainty_map(self, uncertainty_map, coarse_size=4):
        fs = self.env.field_size
        bh = fs // coarse_size
        bw = fs // coarse_size
        coarse = np.zeros((coarse_size, coarse_size), dtype=np.float32)
        for bi in range(coarse_size):
            for bj in range(coarse_size):
                xs = bi * bh
                xe = (bi + 1) * bh if bi < coarse_size - 1 else fs
                ys = bj * bw
                ye = (bj + 1) * bw if bj < coarse_size - 1 else fs
                coarse[bi, bj] = np.mean(uncertainty_map[xs:xe, ys:ye])
        return coarse.flatten().astype(np.float32)

    def compute_shared_uncertainty_map(self):
        beliefs = torch.stack(
            [agent.belief_map for agent in self.planner.agents], dim=0
        )
        entropy = -torch.sum(beliefs * torch.log(beliefs + 1e-9), dim=-1)
        entropy = entropy / np.log(self.planner.agents[0].num_classes)
        return entropy.mean(dim=0).cpu().numpy().astype(np.float32)