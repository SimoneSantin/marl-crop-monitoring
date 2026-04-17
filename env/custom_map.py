import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import (
    COUNT_MARKER,
    COUNT_VECTOR,
)

from env.field_generator import FieldGenerator
from env.sensor import ScalarSensor, VectorSensor


class CustomMapEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        field_size,
        num_agents=1,
        max_steps=2000,
        reward_config=None,
        algorithm=None,
        device="cpu"
    ):

        super(CustomMapEnv, self).__init__()

        self.field_size = field_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.reward_config = reward_config
        self.algorithm = algorithm
        self.device = device

        self.generator = FieldGenerator(size=self.field_size, len_scale=5.0, var=1.0)

        self.scalar_sensor = ScalarSensor()
        self.vector_sensor = VectorSensor()

        self.action_space = spaces.Discrete(4)

        # osservazione: POV + sensori 
        self.obs_dim = 27 + 9 * COUNT_MARKER + 16

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # stato ambiente
        self.agent_pos = None
        self.visited_mask = None
        self.current_step = 0
        self.plant_vec_dx = None
        self.plant_vec_dy = None
        self.grid_counts = None
        self.grid_angles = None
        self.uniform_sensor_dist = np.ones(COUNT_MARKER, dtype=np.float32) / COUNT_MARKER
        # movimento precedente per ogni agente
        self.last_move_vector = None

        # coverage tracking
        self.prev_coverage = 0.0

        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        field_data = self.generator.generate_field(seed=seed)

        self.grid_counts = field_data["true_counts"]
        self.grid_angles = field_data["true_angles"]

        # inizializzazione agenti
        self.agent_pos = []
        self.last_move_vector = []

        for _ in range(self.num_agents):

            x = np.random.randint(0, self.field_size)
            y = np.random.randint(0, self.field_size)

            self.agent_pos.append([x, y])

        self.visited_mask = np.zeros((self.field_size, self.field_size), dtype=bool)

        for pos in self.agent_pos:
            self.visited_mask[pos[0], pos[1]] = True

        self.current_step = 0
        self.prev_coverage = 0.0

        self.last_move_vector = [np.array([0.0, 0.0]) for _ in range(self.num_agents)]
        self.plant_vec_dx = np.sin(self.grid_angles).astype(np.float32)
        self.plant_vec_dy = np.cos(self.grid_angles).astype(np.float32)
        obs = self._get_obs()

        return obs, {}

    def step(self, actions):

        self.current_step += 1

        alignments = []
        new_positions = []
        new_moves = []

        # ----------------------------
        # FASE 1: calcolo nuove posizioni
        # ----------------------------
        for agent_id, action in enumerate(actions):

            x, y = self.agent_pos[agent_id]

            dx, dy = 0, 0

            if action == 0:   # UP
                x_new = max(0, x - 1)
                y_new = y
                dx = -1

            elif action == 1: # DOWN
                x_new = min(self.field_size - 1, x + 1)
                y_new = y
                dx = 1

            elif action == 2: # LEFT
                x_new = x
                y_new = max(0, y - 1)
                dy = -1

            elif action == 3: # RIGHT
                x_new = x
                y_new = min(self.field_size - 1, y + 1)
                dy = 1

            new_positions.append([x_new, y_new])
            new_moves.append(np.array([dx, dy]))

        # ----------------------------
        # FASE 2: aggiornamento stato ambiente
        # ----------------------------
        new_cells = 0

        for pos in new_positions:
            x, y = pos

            if not self.visited_mask[x, y]:
                new_cells += 1

            self.visited_mask[x, y] = True

        unique_positions = set(tuple(p) for p in new_positions)
        collisions = len(new_positions) - len(unique_positions)

        visited_cells = np.sum(self.visited_mask)
        total_cells = self.field_size * self.field_size
        coverage = visited_cells / total_cells

        self.prev_coverage = coverage
        self.agent_pos = new_positions
        self.last_move_vector = new_moves

        obs = self._get_obs(alignments)

        mean_alignment = float(np.mean(alignments))
        
        if self.algorithm == "MAPPO":

            proximity_penalty = self.compute_proximity_penalty(
                self.agent_pos,
                threshold=1,
                weight=0.05
            )

            reward = (
                self.reward_config["new_cell_weight"] * new_cells
                - self.reward_config["collision_weight"] * collisions
                - self.reward_config["step_penalty"]
                + self.reward_config["alignment_weight"] * mean_alignment
                - proximity_penalty
            )

            #if coverage > self.reward_config["completion_threshold"]:
                #reward += self.reward_config["completion_bonus"]
                
            rewards = [reward for _ in range(self.num_agents)]

        else: #MCTS BOTA
            reward = -0.05
            rewards = [reward for _ in range(self.num_agents)]
        # aggiorna stato
        
        #check metriche
        new_cells_term = self.reward_config["new_cell_weight"] * new_cells
        collisions_term = self.reward_config["collision_weight"] * collisions
  
        step_term = self.reward_config["step_penalty"]
        alignment_term = self.reward_config["alignment_weight"] * mean_alignment

        terminated = coverage > 0.95
        truncated = self.current_step >= self.max_steps
        
        return obs, rewards, terminated, truncated, {
            "new_cells": new_cells,
            "collisions": collisions,
            "coverage": coverage,
            "reward_terms": {
                "new_cells": new_cells_term,
                "collisions": collisions_term,
                "step": step_term,
                "alignment": mean_alignment
            }
        }

    def _get_obs(self, alignments=None):
        observations = []

        for agent_id in range(self.num_agents):
            x, y = self.agent_pos[agent_id]

            #pov = np.zeros((3, 3), dtype=np.float32)
            sensor_patch = []
            alignment_patch = []

            k_r = 0
            for i in range(-1, 2):
                k_c = 0
                for j in range(-1, 2):
                    nx, ny = x + i, y + j

                    if 0 <= nx < self.field_size and 0 <= ny < self.field_size:
                        # manteniamo la POV come semplice occupancy/known mask locale
                        #pov[k_r, k_c] = 1.0 if self.visited_mask[nx, ny] else 0.0

                        alignment = self.compute_cell_alignment(nx, ny, agent_id)
                        noise_intensity = 1.0 - alignment

                        true_val = self.grid_counts[nx, ny]
                        sensor_dist = self.scalar_sensor.observe(true_val, noise_intensity)

                    else:
                        #pov[k_r, k_c] = -1.0
                        alignment = 0.0
                        sensor_dist = self.uniform_sensor_dist

                    alignment_patch.append(alignment)
                    sensor_patch.append(sensor_dist)

                    k_c += 1
                k_r += 1

           # pov_flat = pov.flatten().astype(np.float32)
            sensor_patch_flat = np.concatenate(sensor_patch).astype(np.float32)
            alignment_patch_flat = np.array(alignment_patch, dtype=np.float32)

            obs = np.concatenate([
                #pov_flat,               # 9
                alignment_patch_flat,   # 9
                sensor_patch_flat       # 9 * COUNT_MARKER
            ]).astype(np.float32)

            if alignments is not None:
                alignments.append(float(np.mean(alignment_patch)))

            observations.append(obs)

        return observations

    def render(self):

        if self.fig is None:

            plt.ion()

            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.ax.clear()

        self.ax.imshow(self.visited_mask.T, origin="lower", cmap="Blues")

        for pos in self.agent_pos:

            self.ax.plot(pos[1], pos[0], "ro", markersize=8)

        self.ax.set_title(f"Step: {self.current_step}")

        plt.draw()
        plt.pause(0.001)

    def set_noise_levels(self, scalar=None, vector=None):

        pass

    def compute_proximity_penalty(self, agent_positions, threshold=2, weight=0.05):
        penalty = 0.0
        n = len(agent_positions)

        for i in range(n):
            xi, yi = agent_positions[i]

            for j in range(i + 1, n):
                xj, yj = agent_positions[j]

                dist = abs(xi - xj) + abs(yi - yj)

                # penalizza solo se vicini ma non nella stessa cella
                if 1 <= dist <= threshold:
                    penalty += weight * (threshold - dist + 1)

        return penalty
    
    def get_patch_coords(self, x, y, radius=1):
        coords = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                nx, ny = x + i, y + j
                coords.append((nx, ny))
        return coords

    def step(self, actions):

        self.current_step += 1

        alignments = []
        new_positions = []
        new_moves = []

        # ----------------------------
        # FASE 1: calcolo nuove posizioni
        # ----------------------------
        for agent_id, action in enumerate(actions):

            x, y = self.agent_pos[agent_id]

            dx, dy = 0, 0

            if action == 0:   # UP
                x_new = max(0, x - 1)
                y_new = y
                dx = -1

            elif action == 1: # DOWN
                x_new = min(self.field_size - 1, x + 1)
                y_new = y
                dx = 1

            elif action == 2: # LEFT
                x_new = x
                y_new = max(0, y - 1)
                dy = -1

            elif action == 3: # RIGHT
                x_new = x
                y_new = min(self.field_size - 1, y + 1)
                dy = 1

            new_positions.append([x_new, y_new])
            new_moves.append(np.array([dx, dy]))

        # ----------------------------
        # FASE 2: aggiornamento stato ambiente
        # ----------------------------
        new_cells = 0

        for pos in new_positions:
            x, y = pos

            if not self.visited_mask[x, y]:
                new_cells += 1

            self.visited_mask[x, y] = True

        unique_positions = set(tuple(p) for p in new_positions)
        collisions = len(new_positions) - len(unique_positions)

        visited_cells = np.sum(self.visited_mask)
        total_cells = self.field_size * self.field_size
        coverage = visited_cells / total_cells

        self.prev_coverage = coverage
        self.agent_pos = new_positions
        self.last_move_vector = new_moves

        obs = self._get_obs(alignments)

        mean_alignment = float(np.mean(alignments))
        
        if self.algorithm == "MAPPO":

            proximity_penalty = self.compute_proximity_penalty(
                self.agent_pos,
                threshold=1,
                weight=0.01
            )

            reward = (
                self.reward_config["new_cell_weight"] * new_cells
                - self.reward_config["collision_weight"] * collisions
                - self.reward_config["step_penalty"]
                + self.reward_config["alignment_weight"] * mean_alignment
                - proximity_penalty
            )

            if coverage > self.reward_config["completion_threshold"]:
                reward += self.reward_config["completion_bonus"]
            
            #check metriche
            new_cells_term = self.reward_config["new_cell_weight"] * new_cells
            collisions_term = self.reward_config["collision_weight"] * collisions
    
            step_term = self.reward_config["step_penalty"]
            rewards = [reward for _ in range(self.num_agents)]

        else: #MCTS BOTA
            reward = -0.05
            new_cells_term = 0
            collisions_term = 0
            step_term = 0
            rewards = [reward for _ in range(self.num_agents)]


        terminated = coverage > 0.95
        truncated = self.current_step >= self.max_steps
        
        return obs, rewards, terminated, truncated, {
            "new_cells": new_cells,
            "collisions": collisions,
            "coverage": coverage,
            "reward_terms": {
                "new_cells": new_cells_term,
                "collisions": collisions_term,
                "step": step_term,
                "alignment": mean_alignment
            }
        }

    def _get_obs(self, alignments=None):
        observations = []

        for agent_id in range(self.num_agents):
            x, y = self.agent_pos[agent_id]

            pov = np.zeros((3, 3), dtype=np.float32)
            sensor_patch = []
            alignment_patch = []

            k_r = 0
            for i in range(-1, 2):
                k_c = 0
                for j in range(-1, 2):
                    nx, ny = x + i, y + j

                    if 0 <= nx < self.field_size and 0 <= ny < self.field_size:
                        # manteniamo la POV come semplice occupancy/known mask locale
                        pov[k_r, k_c] = 1.0 if self.visited_mask[nx, ny] else 0.0

                        alignment = self.compute_cell_alignment(nx, ny, agent_id)
                        noise_intensity = 1.0 - alignment

                        true_val = self.grid_counts[nx, ny]
                        sensor_dist = self.scalar_sensor.observe(true_val, noise_intensity)

                    else:
                        pov[k_r, k_c] = -1.0
                        alignment = 0.0
                        sensor_dist = self.uniform_sensor_dist

                    alignment_patch.append(alignment)
                    sensor_patch.append(sensor_dist)

                    k_c += 1
                k_r += 1

            pov_flat = pov.flatten().astype(np.float32)
            sensor_patch_flat = np.concatenate(sensor_patch).astype(np.float32)
            alignment_patch_flat = np.array(alignment_patch, dtype=np.float32)

            obs = np.concatenate([
                #pov_flat,               # 9
                alignment_patch_flat,   # 9
                sensor_patch_flat       # 9 * COUNT_MARKER
            ]).astype(np.float32)

            if alignments is not None:
                alignments.append(float(np.mean(alignment_patch)))

            observations.append(obs)

        return observations

    def render(self):

        if self.fig is None:

            plt.ion()

            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.ax.clear()

        self.ax.imshow(self.visited_mask.T, origin="lower", cmap="Blues")

        for pos in self.agent_pos:

            self.ax.plot(pos[1], pos[0], "ro", markersize=8)

        self.ax.set_title(f"Step: {self.current_step}")

        plt.draw()
        plt.pause(0.001)

    def set_noise_levels(self, scalar=None, vector=None):

        pass

    def compute_proximity_penalty(self, agent_positions, threshold=2, weight=0.05):
        penalty = 0.0
        n = len(agent_positions)

        for i in range(n):
            xi, yi = agent_positions[i]

            for j in range(i + 1, n):
                xj, yj = agent_positions[j]

                dist = abs(xi - xj) + abs(yi - yj)

                # penalizza solo se vicini ma non nella stessa cella
                if 1 <= dist <= threshold:
                    penalty += weight * (threshold - dist + 1)

        return penalty
    
    def get_patch_coords(self, x, y, radius=1):
        coords = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                nx, ny = x + i, y + j
                coords.append((nx, ny))
        return coords

    def compute_cell_alignment(self, nx, ny, agent_id):
        alpha = self.grid_angles[nx, ny]

        plant_vec_dy = np.cos(alpha)
        plant_vec_dx = np.sin(alpha)

        drone_dx = self.last_move_vector[agent_id][0]
        drone_dy = self.last_move_vector[agent_id][1]

        norm_drone = np.sqrt(drone_dx ** 2 + drone_dy ** 2)

        if norm_drone <= 1e-9:
            return 0.0

        dot_product = (drone_dx * plant_vec_dx) + (drone_dy * plant_vec_dy)
        alignment = abs(dot_product / norm_drone)

        return float(np.clip(alignment, 0.0, 1.0))