from operator import pos

import numpy as np
import random
import copy


def entropy(p):
    eps = 1e-9
    return -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))


class MCTSNode:

    def __init__(self, pos, belief, depth, planner, parent=None, action=None):

        self.pos = pos
        self.belief = belief
        self.depth = depth

        self.parent = parent
        self.action = action

        self.children = {}

        self.visits = 0
        self.value = 0

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
            # Riduci child.value a uno scalare usando la media
            if isinstance(child.value, np.ndarray):
                exploit = np.mean(child.value) / (child.visits + 1e-6)
            else:
                exploit = child.value / (child.visits + 1e-6)

            explore = c * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))

            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child_node = child

        return best_child_node
    
    def expand(self):

        action = self.untried_actions.pop()

        new_pos = self.planner.simulate_move(self.pos, action)

        new_belief = self.belief.copy()

        child = MCTSNode(
            new_pos,
            new_belief,
            self.depth + 1,
            self.planner,
            parent=self,
            action=action
        )

        self.children[action] = child

        return child

    def backprop(self, reward):

        node = self

        while node is not None:

            node.visits += 1
            node.value += reward

            node = node.parent

class DecMCTSPlanner:

    def __init__(self, env, horizon=8, iterations=500):

        self.env = env
        self.sensor = env.scalar_sensor
        self.num_classes = 10
        self.horizon = horizon
        self.iterations = iterations

        self.actions = [0,1,2,3]   # up down left right

    def simulate_move(self, pos, action):

        x,y = pos

        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        x = np.clip(x,0,self.env.field_size-1)
        y = np.clip(y,0,self.env.field_size-1) 

        return (x,y)

    def compute_ig(self, belief, pos, noise):

        x, y = pos

        prior = belief[x, y]

        H_before = -np.sum(prior * np.log(prior + 1e-9))

        expected_H_after = 0

        for k in range(self.num_classes):

            likelihood = self.sensor.observe(k, noise) 

            posterior = prior * likelihood
            posterior /= (posterior.sum() + 1e-9)

            H_after = -np.sum(posterior * np.log(posterior + 1e-9))

            expected_H_after += prior[k] * H_after

        return H_before - expected_H_after


    def update_belief(self, belief, pos, noise):

        x, y = pos

        prior = belief[x, y]

        likelihood = self.env.scalar_sensor.observe(
            np.argmax(prior), noise
        ) 

        posterior = prior * likelihood
        posterior /= (posterior.sum() + 1e-9)

        belief[x, y] = posterior

        return belief


    def rollout(self, node):

        pos = node.pos
        belief = node.belief.copy()

        total_reward = 0

        depth = node.depth

        while depth < self.horizon:

            action = random.choice(self.actions)

            pos = self.simulate_move(pos, action)

            move_vec = self.action_to_vector(action)

            noise = self.get_noise_intensity(
                pos,
                move_vec,
                self.env.grid_angles
            )

            ig = self.compute_ig(belief, pos, noise)

            belief = self.update_belief(belief, pos, noise)

            total_reward += ig

            depth += 1

        return total_reward


    def choose_action(self, position, belief_map, angles):

        root = MCTSNode(
            position,
            belief_map.copy(),
            depth=0,
            planner=self
        )

        for _ in range(self.iterations):

            node = root

            # SELECTION
            while node.is_fully_expanded() and not node.is_terminal():

                node = node.best_child()

            # EXPANSION
            if not node.is_terminal():

                node = node.expand()

            # SIMULATION
            reward = self.rollout(node)

            # BACKPROP
            node.backprop(reward)

        # BEST ACTION

        best_action = None
        best_visits = -1

        for action,child in root.children.items():

            if child.visits > best_visits:

                best_visits = child.visits
                best_action = action

        return best_action
    
    def get_noise_intensity(self, pos, move_vector, angles):

        x, y = pos

        alpha = angles[x, y]

        plant_vec_dy = np.cos(alpha)
        plant_vec_dx = np.sin(alpha)

        drone_dx, drone_dy = move_vector

        norm_drone = np.sqrt(drone_dx**2 + drone_dy**2)

        if norm_drone > 0:
            dot = (drone_dx * plant_vec_dx) + (drone_dy * plant_vec_dy)
            alignment = abs(dot / norm_drone)
        else:
            alignment = 0.0

        noise_intensity = 1.0 - alignment

        return noise_intensity

    def action_to_vector(self, action):

        if action == 0:
            return (-1, 0)

        elif action == 1:
            return (1, 0)

        elif action == 2:
            return (0, -1)

        elif action == 3:
            return (0, 1)