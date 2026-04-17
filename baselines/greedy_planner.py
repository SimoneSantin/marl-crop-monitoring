import numpy as np


class GreedyIGPlannerScalar:
    """
    Greedy 1-step Information Gain planner coerente con osservazione 3x3.

    Per ogni azione:
    - simula la prossima posizione
    - considera la patch 3x3 centrata sulla prossima posizione
    - per ogni cella calcola alignment locale
    - usa alignment sia per il noise del sensore sia per il peso del contributo
    """

    def __init__(self, field_size, num_classes, scalar_sensor, gamma=3.0):
        self.field_size = field_size
        self.num_classes = num_classes
        self.sensor = scalar_sensor
        self.actions = [0, 1, 2, 3]
        self.gamma = gamma

    def entropy(self, p):
        eps = 1e-9
        return -np.sum(p * np.log2(p + eps))

    def simulate_position(self, position, action):
        x, y = position

        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.field_size - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.field_size - 1, y + 1)

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

    def compute_ig_cell(self, belief, position, noise_intensity):
        x, y = position

        prior = belief[x, y]
        h_before = self.entropy(prior)

        expected_h_after = 0.0

        for k in range(self.num_classes):
            likelihood = self.sensor.observe(k, noise_intensity)

            posterior = prior * likelihood
            posterior /= (posterior.sum() + 1e-9)

            h_after = self.entropy(posterior)
            expected_h_after += prior[k] * h_after

        ig = h_before - expected_h_after
        return ig

    def compute_ig_patch(self, center_pos, belief, move_vector, angles):
        cx, cy = center_pos
        total_score = 0.0

        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = cx + i, cy + j

                if 0 <= nx < self.field_size and 0 <= ny < self.field_size:
                    patch_pos = (nx, ny)

                    # alignment -> noise
                    alignment = self.get_alignment(patch_pos, move_vector, angles)
                    noise = 1.0 - alignment

                    # IG già incorpora il noise
                    ig = self.compute_ig_cell(belief, patch_pos, noise)

                    total_score += ig

        return total_score
  

    def choose_action(self, current_position, belief, angles):
        best_action = 0
        best_score = -np.inf

        for action in self.actions:
            next_pos = self.simulate_position(current_position, action)
            move_vector = self.action_to_vector(action)

            score = self.compute_ig_patch(
                center_pos=next_pos,
                belief=belief,
                move_vector=move_vector,
                angles=angles
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action