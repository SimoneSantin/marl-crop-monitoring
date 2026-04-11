import numpy as np
import torch
from utils.constants import COUNT_MARKER


class MAPPOTrainer:
    def __init__(self, env, planner, num_episodes, reward_weights):
        self.env = env
        self.planner = planner
        self.num_episodes = num_episodes
        self.reward_weights = reward_weights  # dict: {"belief": x, "accuracy": y}

        self.visit_heatmap = np.zeros(
            (self.env.field_size, self.env.field_size),
            dtype=np.float32
        )
        self.last_episode_path = None
        self.accuracy_history = []

    def train(self):
        alignment_history = []
        rewards_history = []
        coverage_history = []
        episode_lengths = []
        new_cells_history = []
        collisions_history = []
        efficiency_history = []
        terms_history = []

        for episode in range(self.num_episodes):
            obs, _ = self.env.reset()
            done = False

            # 🔥 RESET BELIEF DEGLI AGENTI
            for agent in self.planner.agents:
                agent.reset()
            self.planner.reset_hidden_states()
            episode_paths = [[] for _ in range(self.env.num_agents)]
            episode_accuracy = []
            episode_reward = 0.0
            episode_new_cells = 0
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
                print(f"Episode {episode} | Step {steps} | Reward {episode_reward:.3f}")

                actions = []
                log_probs = []
                current_obs_for_buffer = []

                # stato globale
                global_state = np.concatenate([
                    self.env.visited_mask.flatten().astype(np.float32),
                    (np.array(self.env.agent_pos) / self.env.field_size).flatten()
                ])

                # =========================
                # 1. Azioni
                # =========================
                for agent_id, agent in enumerate(self.planner.agents):
                    enriched_obs = self.enrich_obs_with_belief(obs[agent_id], agent)

                    action, log_prob = agent.choose_action(enriched_obs)

                    actions.append(action)
                    log_probs.append(log_prob)
                    current_obs_for_buffer.append(enriched_obs)

                # =========================
                # 2. Step env
                # =========================
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated

                # =========================
                # 3. Belief update + belief reward
                # =========================

                local_accuracy_bonus_per_agent = []

                for i, agent in enumerate(self.planner.agents):
                    x, y = self.env.agent_pos[i]

                    old_belief = agent.belief_map[x, y].copy()

                    sensor_dist = next_obs[i][9: 9 + COUNT_MARKER]
                    agent.update_belief(sensor_dist)

                    new_belief = agent.belief_map[x, y]

                    true_class = self.env.grid_counts[x, y]   # indice classe vera, es. 0,1,2,...

                    ce_before = -np.log(old_belief[true_class] + 1e-9)
                    ce_after  = -np.log(new_belief[true_class] + 1e-9)

                    ce_gain = ce_before - ce_after

                    local_info_bonus = self.reward_weights["accuracy"] * ce_gain
                    local_accuracy_bonus_per_agent.append(local_info_bonus)

                shaped_rewards = [
                    rewards[i] + local_accuracy_bonus_per_agent[i]
                    for i in range(self.env.num_agents)
                ]

                current_accuracy = np.mean(
                    [agent.compute_accuracy() for agent in self.planner.agents]
                )
                episode_accuracy.append(current_accuracy)

                # =========================
                # 6. Buffer
                # =========================
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

                # =========================
                # 7. Logging
                # =========================
                obs = next_obs

                episode_reward += np.mean(shaped_rewards)
                episode_new_cells += info["new_cells"]
                episode_collisions += info["collisions"]
                episode_alignment += info["reward_terms"]["alignment"]
                print(episode_collisions)
                episode_terms["new_cells"] += info["reward_terms"]["new_cells"]
                episode_terms["collisions"] += info["reward_terms"]["collisions"]
                episode_terms["step"] += info["reward_terms"]["step"]
                episode_terms["alignment"] += info["reward_terms"]["alignment"]
                episode_terms["accuracy"] += float(np.mean(local_accuracy_bonus_per_agent))

                steps += 1

                for i, pos in enumerate(self.env.agent_pos):
                    episode_paths[i].append(tuple(pos))


            # =========================
            # UPDATE MAPPO
            # =========================
            self.planner.update()

            # normalizza termini
            for k in episode_terms:
                episode_terms[k] /= max(steps, 1)
            terms_history.append(episode_terms.copy())

            if episode == self.num_episodes - 1:
                self.last_episode_paths = episode_paths

            self.accuracy_history.append(
                np.mean(episode_accuracy) if episode_accuracy else 0.0
            )
            rewards_history.append(episode_reward)
            alignment_history.append(episode_alignment / max(steps, 1))
            coverage_history.append(np.mean(self.env.visited_mask))
            episode_lengths.append(steps)
            new_cells_history.append(episode_new_cells)
            collisions_history.append(episode_collisions)
            efficiency_history.append(episode_new_cells / max(steps, 1))
            self.visit_heatmap += self.env.visited_mask.astype(np.float32)

        return {
            "rewards": rewards_history,
            "coverage": coverage_history,
            "lengths": episode_lengths,
            "new_cells": new_cells_history,
            "collisions": collisions_history,
            "efficiency": efficiency_history,
            "visit_heatmap": self.visit_heatmap,
            "episode_paths": self.last_episode_paths,
            "accuracy": self.accuracy_history,
            "terms": terms_history,
            "alignment": alignment_history
        }

    def enrich_obs_with_belief(self, obs, agent):
        x, y = self.env.agent_pos[agent.agent_id]

        visited_patch = np.zeros((3, 3), dtype=np.float32)
        entropy_patch = np.zeros((3, 3), dtype=np.float32)
        agents_patch = np.zeros((3, 3), dtype=np.float32)

        for i in range(-1, 2):
            for j in range(-1, 2):
                px, py = i + 1, j + 1
                nx, ny = x + i, y + j

                if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                    visited_patch[px, py] = float(self.env.visited_mask[nx, ny])

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
                else:
                    visited_patch[px, py] = -1.0
                    entropy_patch[px, py] = 0.0
                    agents_patch[px, py] = 0.0

        # CNN input (3 canali)
        local_patch = np.stack([
            visited_patch,
            entropy_patch,
            agents_patch
        ]).astype(np.float32)

        local_patch_flat = local_patch.flatten()  # 27

        sensor_dist = obs[9:].astype(np.float32)

        # 🔥 NUOVA FEATURE: shared uncertainty
        shared_uncertainty_coarse = self.compute_shared_uncertainty_coarse_map(
            target_agent_id=agent.agent_id,
            coarse_size=4
        )

        enriched_obs = np.concatenate([
            local_patch_flat,
            sensor_dist,
            shared_uncertainty_coarse
        ]).astype(np.float32)

        return enriched_obs
                
    def compute_shared_uncertainty_coarse_map(self, target_agent_id, coarse_size=4):
        """
        Coarse map dell'incertezza media degli ALTRI agenti (sempre disponibile).
        
        Args:
            target_agent_id: agente per cui costruisci l'obs
            coarse_size: dimensione della griglia coarse (es. 4 -> 4x4)

        Returns:
            np.array shape (coarse_size * coarse_size,)
        """
        field_size = self.env.field_size
        block_h = field_size // coarse_size
        block_w = field_size // coarse_size

        shared_entropy_coarse = []

        for bi in range(coarse_size):
            for bj in range(coarse_size):
                xs = bi * block_h
                xe = (bi + 1) * block_h
                ys = bj * block_w
                ye = (bj + 1) * block_w

                block_entropies = []

                for other_id, other_agent in enumerate(self.planner.agents):
                    if other_id == target_agent_id:
                        continue

                    belief_block = other_agent.belief_map[xs:xe, ys:ye]

                    entropy_block = -np.sum(
                        belief_block * np.log(belief_block + 1e-9),
                        axis=2
                    )
                    entropy_block /= np.log(other_agent.num_classes)

                    block_entropies.append(np.mean(entropy_block))

                if len(block_entropies) == 0:
                    shared_entropy_coarse.append(0.0)
                else:
                    shared_entropy_coarse.append(np.mean(block_entropies))

        return np.array(shared_entropy_coarse, dtype=np.float32)