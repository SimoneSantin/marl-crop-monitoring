import numpy as np
import random


class MCTSNode:
    def __init__(self, pos, belief, depth, planner, parent=None, action=None):
        self.pos = pos
        self.belief = belief
        self.depth = depth

        self.parent = parent
        self.action = action

        self.children = {}
        self.visits = 0
        self.value = 0.0

        self.planner = planner
        self.untried_actions = planner.actions.copy()

    def is_terminal(self):
        return self.depth >= self.planner.horizon

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c=1.4):
        best_score = -np.inf
        best_child_node = None

        for child in self.children.values():
            exploit = child.value / (child.visits + 1e-9)
            explore = c * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-9))
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child_node = child

        return best_child_node

    def expand(self):
        action = self.untried_actions.pop()

        new_pos = self.planner.simulate_move(self.pos, action)
        new_belief = self.belief.copy()

        # Reward atteso della patch osservata dopo l'azione
        immediate_reward = self.planner.compute_ig_patch(
            belief=new_belief,
            center_pos=new_pos,
            action=action,
            angles=self.planner.env.grid_angles
        )

        # Update belief atteso sulla patch
        new_belief = self.planner.update_belief_patch_expected(
            belief=new_belief,
            center_pos=new_pos,
            action=action,
            angles=self.planner.env.grid_angles,
            gamma=self.planner.gamma
        )

        child = MCTSNode(
            pos=new_pos,
            belief=new_belief,
            depth=self.depth + 1,
            planner=self.planner,
            parent=self,
            action=action
        )

        # Salviamo la reward immediata nel nodo figlio per poterla usare nel rollout
        child.immediate_reward = immediate_reward

        self.children[action] = child
        return child

    def backprop(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class DecMCTSPlanner:
    def __init__(
        self,
        env,
        horizon=5,
        iterations=300,
        discount_factor=0.95,
        gamma=3.0,
        ucb_c=1.4
    ):
        self.env = env
        self.sensor = env.scalar_sensor
        self.num_classes = 10
        self.horizon = horizon
        self.iterations = iterations
        self.discount_factor = discount_factor
        self.gamma = gamma
        self.ucb_c = ucb_c

        self.actions = [0, 1, 2, 3]  # up, down, left, right

    def entropy(self, p):
        return -np.sum(p * np.log(p + 1e-9))

    def simulate_move(self, pos, action):
        x, y = pos

        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        x = np.clip(x, 0, self.env.field_size - 1)
        y = np.clip(y, 0, self.env.field_size - 1)

        return (x, y)

    def action_to_vector(self, action):
        if action == 0:
            return (-1, 0)
        elif action == 1:
            return (1, 0)
        elif action == 2:
            return (0, -1)
        elif action == 3:
            return (0, 1)
        else:
            raise ValueError(f"Unknown action: {action}")

    def get_alignment(self, pos, move_vector, angles):
        x, y = pos
        alpha = angles[x, y]

        plant_vec_dy = np.cos(alpha)
        plant_vec_dx = np.sin(alpha)

        drone_dx, drone_dy = move_vector
        norm_drone = np.sqrt(drone_dx ** 2 + drone_dy ** 2)

        if norm_drone > 0:
            dot = (drone_dx * plant_vec_dx) + (drone_dy * plant_vec_dy)
            alignment = abs(dot / norm_drone)
        else:
            alignment = 0.0

        return float(np.clip(alignment, 0.0, 1.0))

    def get_noise_intensity(self, pos, move_vector, angles):
        alignment = self.get_alignment(pos, move_vector, angles)
        return 1.0 - alignment

    def compute_ig_cell(self, belief, pos, noise):
        x, y = pos
        prior = belief[x, y]

        h_before = self.entropy(prior)
        expected_h_after = 0.0

        for k in range(self.num_classes):
            likelihood = self.sensor.observe(k, noise)

            posterior_k = prior * likelihood
            posterior_k /= (posterior_k.sum() + 1e-9)

            h_after = self.entropy(posterior_k)
            expected_h_after += prior[k] * h_after

        return h_before - expected_h_after

    def expected_posterior_multiclass(self, prior, noise):
        """
        Expected posterior multiclass:
        E[p(z | observation)] mediato sui possibili true class k
        con peso prior[k].
        """
        expected_post = np.zeros_like(prior)

        for k in range(self.num_classes):
            likelihood = self.sensor.observe(k, noise)

            posterior_k = prior * likelihood
            posterior_k /= (posterior_k.sum() + 1e-9)

            expected_post += prior[k] * posterior_k

        expected_post /= (expected_post.sum() + 1e-9)
        return expected_post

    def compute_ig_patch(self, belief, center_pos, action, angles):
        """
        Reward atteso della patch 3x3.
        """
        cx, cy = center_pos
        move_vector = self.action_to_vector(action)

        total_ig = 0.0

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = cx + i, cy + j

                if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                    patch_pos = (nx, ny)

                    alignment = self.get_alignment(patch_pos, move_vector, angles)
                    noise = 1.0 - alignment

                    ig = self.compute_ig_cell(belief, patch_pos, noise)
                    total_ig += ig

        return total_ig

    def update_belief_patch_expected(self, belief, center_pos, action, angles, gamma=3.0):
        """
        Update atteso del belief sulla patch 3x3.
        Coerente con il tuo update:
        - alignment -> noise
        - alignment^gamma -> reliability del mixing prior/posterior
        """
        cx, cy = center_pos
        move_vector = self.action_to_vector(action)

        new_belief = belief.copy()

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = cx + i, cy + j

                if 0 <= nx < self.env.field_size and 0 <= ny < self.env.field_size:
                    prior = new_belief[nx, ny].copy()

                    alignment = self.get_alignment((nx, ny), move_vector, angles)
                    noise = 1.0 - alignment

                    expected_post = self.expected_posterior_multiclass(prior, noise)

                    reliability = alignment ** gamma

                    updated = (1.0 - reliability) * prior + reliability * expected_post
                    updated /= (updated.sum() + 1e-9)

                    new_belief[nx, ny] = updated

        return new_belief

    def greedy_rollout_action(self, pos, belief, angles):
        """
        Rollout policy greedy sulla IG patch.
        Molto meglio di random.choice.
        """
        best_action = None
        best_score = -np.inf

        for action in self.actions:
            next_pos = self.simulate_move(pos, action)

            score = self.compute_ig_patch(
                belief=belief,
                center_pos=next_pos,
                action=action,
                angles=angles
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def rollout(self, node):
        pos = node.pos
        belief = node.belief.copy()
        total_reward = 0.0
        discount = 1.0
        depth = node.depth

        while depth < self.horizon:
            action = self.greedy_rollout_action(
                pos=pos,
                belief=belief,
                angles=self.env.grid_angles
            )

            next_pos = self.simulate_move(pos, action)

            reward = self.compute_ig_patch(
                belief=belief,
                center_pos=next_pos,
                action=action,
                angles=self.env.grid_angles
            )

            belief = self.update_belief_patch_expected(
                belief=belief,
                center_pos=next_pos,
                action=action,
                angles=self.env.grid_angles,
                gamma=self.gamma
            )

            total_reward += discount * reward
            discount *= self.discount_factor

            pos = next_pos
            depth += 1

        return total_reward

    def choose_action(self, position, belief_map, angles):
        root = MCTSNode(
            pos=position,
            belief=belief_map.copy(),
            depth=0,
            planner=self
        )

        for _ in range(self.iterations):
            node = root

            # SELECTION
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(c=self.ucb_c)

            # EXPANSION
            if not node.is_terminal():
                node = node.expand()

            # SIMULATION
            reward = 0.0

            # reward immediata del nodo espanso, se presente
            if hasattr(node, "immediate_reward"):
                reward += node.immediate_reward

            reward += self.rollout(node)

            # BACKPROP
            node.backprop(reward)

        # BEST ACTION: scegli per valore medio, non per visite
        best_action = None
        best_value = -np.inf

        for action, child in root.children.items():
            avg_value = child.value / (child.visits + 1e-9)

            if avg_value > best_value:
                best_value = avg_value
                best_action = action

        return best_action