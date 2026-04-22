import numpy as np
import torch
from collections import deque

from utils.constants import COUNT_MARKER
from LSTM.lstm_model import NetObsReliability


class MAPPOTrainer:
    def __init__(self, env, planner, num_episodes, reward_weights,
                 reliability_model_path="./LSTM/models/patch_reliability_model.pth",
                 reliability_seq_len=5,
                 reliability_hidden_size=128,
                 reliability_num_layers=1):
        self.env = env
        self.planner = planner
        self.num_episodes = num_episodes
        self.reward_weights = reward_weights

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.episode_accuracy_traces = {}
        self.visit_heatmap = np.zeros(
            (self.env.field_size, self.env.field_size),
            dtype=np.float32
        )
        self.last_episode_path = None
        self.accuracy_history = []

        # -------------------------
        # Reliability model
        # -------------------------
        self.reliability_seq_len = reliability_seq_len

        self.reliability_model = NetObsReliability(
            num_classes=COUNT_MARKER,
            hidden_size=reliability_hidden_size,
            num_layers=reliability_num_layers,
            dropout=0.2
        ).to(self.device)

        self.reliability_model.load_state_dict(
            torch.load(reliability_model_path, map_location=self.device)
        )
        self.reliability_model.eval()

        self.agent_patch_histories = [
            deque(maxlen=self.reliability_seq_len)
            for _ in range(self.env.num_agents)
        ]

    def reset_reliability_histories(self):
        self.agent_patch_histories = [
            deque(maxlen=self.reliability_seq_len)
            for _ in range(self.env.num_agents)
        ]

    def build_reliability_step_feature(self, obs_i):
        """
        raw obs attuale:
        [alignment_patch (9) | sensor_patch_flat (9 * COUNT_MARKER)]
        """
        align_start = 0
        align_end = 9

        sensor_start = align_end
        sensor_end = sensor_start + 9 * COUNT_MARKER

        alignment_patch = obs_i[align_start:align_end].astype(np.float32)
        sensor_patch_flat = obs_i[sensor_start:sensor_end].astype(np.float32)

        step_feature = np.concatenate([
            alignment_patch,
            sensor_patch_flat
        ]).astype(np.float32)

        return step_feature

    def predict_patch_confidence(self, agent_id, step_feature):
        history = self.agent_patch_histories[agent_id]
        history.append(step_feature)

        seq = list(history)

        while len(seq) < self.reliability_seq_len:
            seq.insert(0, np.zeros_like(step_feature, dtype=np.float32))

        seq_array = np.stack(seq, axis=0)  # (T, F)
        x = torch.tensor(seq_array, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, F)

        with torch.no_grad():
            outputs = self.reliability_model(x)
            confidence_patch = outputs["pred_confidence_patch"][0].cpu().numpy()  # (9,)

        return confidence_patch.astype(np.float32)

    def train(self):
        alignment_history = []
        rewards_history = []
        coverage_history = []
        episode_lengths = []
        collisions_history = []
        terms_history = []

        checkpoint_episodes = [0] + sorted(set([
            max(0, int((k + 1) * self.num_episodes / 4) - 1)
            for k in range(4)
        ]))

        episode_accuracy_traces = {}

        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            done = False

            track_accuracy_this_episode = episode in checkpoint_episodes
            if track_accuracy_this_episode:
                episode_accuracy_trace = []

            for agent in self.planner.agents:
                agent.reset()

            self.planner.reset_hidden_states()
            self.reset_reliability_histories()

            episode_paths = [[] for _ in range(self.env.num_agents)]
            episode_reward = 0.0
            episode_collisions = 0
            episode_alignment = 0
            steps = 0

            episode_terms = {
                "new_cells": 0.0,
                "collisions": 0.0,
                "step": 0.0,
                "alignment": 0.0,
                "belief": 0.0,
                "accuracy": 0.0
            }

            while not done:
                if steps % 500 == 0:
                    print(f"Episode {episode} | Step {steps} | Reward {episode_reward:.3f}")

                actions = []
                log_probs = []
                current_obs_for_buffer = []

                # -------------------------
                # Shared uncertainty + global state
                # -------------------------
                shared_uncertainty_map = self.compute_shared_uncertainty_map()
                shared_uncertainty_coarse = self.downsample_uncertainty_map(
                    shared_uncertainty_map,
                    coarse_size=4
                )

                global_state = np.concatenate([
                    self.env.visited_mask.flatten().astype(np.float32),
                    shared_uncertainty_map.flatten().astype(np.float32),
                    (np.array(self.env.agent_pos, dtype=np.float32) / self.env.field_size).flatten()
                ]).astype(np.float32)

                # -------------------------
                # 1. Action selection
                # -------------------------
                for agent_id, agent in enumerate(self.planner.agents):
                    enriched_obs = self.enrich_obs_with_belief(
                        obs[agent_id],
                        agent,
                        shared_uncertainty_coarse
                    )

                    action, log_prob = agent.choose_action(enriched_obs)

                    actions.append(action)
                    log_probs.append(log_prob)
                    current_obs_for_buffer.append(enriched_obs)

                # -------------------------
                # 2. Env step
                # -------------------------
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated

                # -------------------------
                # 3. Belief update + local accuracy reward
                # -------------------------
                local_accuracy_bonus_per_agent = []

                for i, agent in enumerate(self.planner.agents):
                    x, y = self.env.agent_pos[i]

                    # salva belief della patch prima dell'update
                    old_beliefs = []

                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                                old_beliefs.append(agent.belief_map[nx, ny].copy())

                    # parse raw obs
                    obs_i = next_obs[i]

                    align_start = 0
                    align_end = 9

                    sensor_start = align_end
                    sensor_end = sensor_start + 9 * COUNT_MARKER

                    alignment_patch = obs_i[align_start:align_end]
                    sensor_patch_flat = obs_i[sensor_start:sensor_end]
                    sensor_patch = sensor_patch_flat.reshape(9, COUNT_MARKER)

                    # reliability model
                    step_feature = self.build_reliability_step_feature(obs_i)
                    confidence_patch = self.predict_patch_confidence(i, step_feature)

                    # belief update
                    agent.update_belief_patch(
                        sensor_patch=sensor_patch,
                        alignment_patch=alignment_patch,
                        confidence_patch=confidence_patch,
                        gamma=3.0
                    )

                    # CE gain medio sulla patch
                    ce_gains = []
                    old_idx = 0

                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                                old_belief = old_beliefs[old_idx]
                                new_belief = agent.belief_map[nx, ny]

                                true_class = self.env.grid_counts[nx, ny]

                                ce_before = -np.log(old_belief[true_class] + 1e-9)
                                ce_after = -np.log(new_belief[true_class] + 1e-9)

                                ce_gain = ce_before - ce_after
                                ce_gains.append(ce_gain)

                                old_idx += 1

                    patch_ce_gain = float(np.mean(ce_gains)) if len(ce_gains) > 0 else 0.0
                    local_info_bonus = self.reward_weights["accuracy"] * patch_ce_gain
                    local_accuracy_bonus_per_agent.append(local_info_bonus)

                shaped_rewards = [
                    rewards[i] + local_accuracy_bonus_per_agent[i]
                    for i in range(self.env.num_agents)
                ]

                if track_accuracy_this_episode:
                    current_accuracy = np.mean(
                        [agent.compute_accuracy() for agent in self.planner.agents]
                    )
                    episode_accuracy_trace.append(current_accuracy)

                # -------------------------
                # 4. Store in buffer
                # -------------------------
                for agent_id in range(self.env.num_agents):
                    self.planner.store_transition(
                        obs=current_obs_for_buffer[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs[agent_id],
                        reward=shaped_rewards[agent_id],
                        global_state=global_state,
                        agent_id=agent_id,
                        done=done
                    )

                # -------------------------
                # 5. Logging
                # -------------------------
                obs = next_obs

                episode_reward += np.mean(shaped_rewards)
                episode_collisions += info["collisions"]
                episode_alignment += info["reward_terms"]["alignment"]

                episode_terms["collisions"] += info["reward_terms"]["collisions"]
                episode_terms["alignment"] += info["reward_terms"]["alignment"]
                episode_terms["accuracy"] += float(np.mean(local_accuracy_bonus_per_agent))

                steps += 1

                for i, pos in enumerate(self.env.agent_pos):
                    episode_paths[i].append(tuple(pos))

            # -------------------------
            # MAPPO update
            # -------------------------
            self.planner.update()

            for k in episode_terms:
                episode_terms[k] /= max(steps, 1)
            terms_history.append(episode_terms.copy())

            if episode == self.num_episodes - 1:
                self.last_episode_paths = episode_paths

            if track_accuracy_this_episode:
                episode_accuracy_traces[episode] = episode_accuracy_trace

            final_accuracy = np.mean(
                [agent.compute_accuracy() for agent in self.planner.agents]
            )
            self.accuracy_history.append(final_accuracy)

            rewards_history.append(episode_reward)
            alignment_history.append(episode_alignment / max(steps, 1))
            coverage_history.append(np.mean(self.env.visited_mask))
            episode_lengths.append(steps)
            collisions_history.append(episode_collisions)

        return {
            "rewards": rewards_history,
            "coverage": coverage_history,
            "lengths": episode_lengths,
            "collisions": collisions_history,
            "episode_paths": self.last_episode_paths,
            "accuracy": self.accuracy_history,
            "terms": terms_history,
            "alignment": alignment_history,
            "accuracy_traces": episode_accuracy_traces
        }

    def enrich_obs_with_belief(self, obs, agent, shared_uncertainty_coarse):
        x, y = self.env.agent_pos[agent.agent_id]

        entropy_patch = np.zeros((3, 3), dtype=np.float32)
        agents_patch = np.zeros((3, 3), dtype=np.float32)

        for i in range(-1, 2):
            for j in range(-1, 2):
                px, py = i + 1, j + 1
                nx, ny = x + i, y + j

                if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                    belief = agent.belief_map[nx, ny]
                    entropy = -np.sum(belief * np.log(belief + 1e-9))
                    entropy /= np.log(agent.num_classes)
                    entropy_patch[px, py] = entropy

                    occupied = 0.0
                    for other_id, (ax, ay) in enumerate(self.env.agent_pos):
                        if other_id != agent.agent_id and ax == nx and ay == ny:
                            occupied = 1.0
                            break
                    agents_patch[px, py] = occupied

        local_patch = np.stack([
            entropy_patch,
            agents_patch
        ]).astype(np.float32)

        local_patch_flat = local_patch.flatten()
        raw_obs_features = obs.astype(np.float32)

        enriched_obs = np.concatenate([
            local_patch_flat,
            raw_obs_features,
            shared_uncertainty_coarse
        ]).astype(np.float32)

        return enriched_obs

    def downsample_uncertainty_map(self, uncertainty_map, coarse_size=4):
        field_size = self.env.field_size
        block_h = field_size // coarse_size
        block_w = field_size // coarse_size

        coarse = np.zeros((coarse_size, coarse_size), dtype=np.float32)

        for bi in range(coarse_size):
            for bj in range(coarse_size):
                xs = bi * block_h
                xe = (bi + 1) * block_h if bi < coarse_size - 1 else field_size
                ys = bj * block_w
                ye = (bj + 1) * block_w if bj < coarse_size - 1 else field_size

                coarse[bi, bj] = np.mean(uncertainty_map[xs:xe, ys:ye])

        return coarse.flatten().astype(np.float32)

    def compute_shared_uncertainty_map(self):
        beliefs = np.stack(
            [agent.belief_map for agent in self.planner.agents],
            axis=0
        )  # shape: (N, H, W, C)

        entropy = -np.sum(beliefs * np.log(beliefs + 1e-9), axis=-1)
        entropy /= np.log(self.planner.agents[0].num_classes)

        shared_uncertainty = np.mean(entropy, axis=0).astype(np.float32)
        return shared_uncertainty