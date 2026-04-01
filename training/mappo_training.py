import numpy as np
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

            episode_paths = [[] for _ in range(self.env.num_agents)]
            episode_accuracy = []
            episode_reward = 0.0
            episode_new_cells = 0
            episode_collisions = 0
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
                    H_before = -np.sum(old_belief * np.log(old_belief + 1e-9))
                    H_before /= np.log(agent.num_classes)

                    sensor_dist = next_obs[i][9: 9 + COUNT_MARKER]
                    agent.update_belief(sensor_dist)

                    new_belief = agent.belief_map[x, y]
                    H_after = -np.sum(new_belief * np.log(new_belief + 1e-9))
                    H_after /= np.log(agent.num_classes)

                    entropy_gain = H_before - H_after

                    local_info_bonus = self.reward_weights["accuracy"] * entropy_gain
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

                episode_terms["new_cells"] += info["reward_terms"]["new_cells"]
                episode_terms["collisions"] += info["reward_terms"]["collisions"]
                episode_terms["step"] += info["reward_terms"]["step"]
                episode_terms["alignment"] += info["reward_terms"]["alignment"]
                episode_terms["belief"] += 0.0
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
            "terms": terms_history
        }

    def enrich_obs_with_belief(self, obs, agent):
        x, y = self.env.agent_pos[agent.agent_id]

        confidence_patch = []
        entropy_patch = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = x + i, y + j

                if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                    local_belief = agent.belief_map[nx, ny]

                    confidence = np.max(local_belief)

                    entropy = -np.sum(local_belief * np.log(local_belief + 1e-9))
                    entropy /= np.log(agent.num_classes)

                else:
                    confidence = 0.0
                    entropy = 0.0

                confidence_patch.append(confidence)
                entropy_patch.append(entropy)

        confidence_patch = np.array(confidence_patch, dtype=np.float32)
        entropy_patch = np.array(entropy_patch, dtype=np.float32)

        # alignment corrente
        alpha = self.env.grid_angles[x, y]

        plant_vec_dy = np.cos(alpha)
        plant_vec_dx = np.sin(alpha)

        drone_dx = self.env.last_move_vector[agent.agent_id][0]
        drone_dy = self.env.last_move_vector[agent.agent_id][1]

        norm_drone = np.sqrt(drone_dx ** 2 + drone_dy ** 2)

        if norm_drone > 0:
            dot_product = (drone_dx * plant_vec_dx) + (drone_dy * plant_vec_dy)
            alignment = abs(dot_product / norm_drone)
        else:
            alignment = 0.0

        extra_features = np.concatenate([
            confidence_patch,
            entropy_patch,
            np.array([alignment], dtype=np.float32)
        ])

        return np.concatenate((obs, extra_features)).astype(np.float32)