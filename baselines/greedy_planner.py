import numpy as np


class GreedyIGPlannerScalar:
    """
    Greedy 1-step Information Gain planner
    """

    def __init__(self, field_size, num_classes, scalar_sensor):
        self.field_size = field_size
        self.num_classes = num_classes
        self.sensor = scalar_sensor
        self.actions = [0, 1, 2, 3]

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

    def compute_ig_cell(self, belief, position, noise_intensity):
        x, y = position

        prior = belief[x, y]
        H_before = self.entropy(prior)

        expected_H_after = 0.0

        # Possibili veri valori (classi)
        for k in range(self.num_classes):

            # Simuliamo che il true_val sia k
            likelihood = self.sensor.observe(k, noise_intensity)

            posterior = prior * likelihood
            posterior /= (posterior.sum() + 1e-9)

            H_after = self.entropy(posterior)

            # Peso = prior[k]
            expected_H_after += prior[k] * H_after

        IG = H_before - expected_H_after
        return IG
    
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
    
 
    def choose_action(self, current_position, belief, angles):
        best_action = 0
        best_ig = -np.inf

        for action in self.actions:

            next_pos = self.simulate_position(current_position, action)

            move_vector = self.action_to_vector(action)

            noise = self.get_noise_intensity(current_position, move_vector, angles)

            ig = self.compute_ig_cell(belief, next_pos, noise)

            if ig > best_ig:
                best_ig = ig
                best_action = action
        return best_action
    
    def action_to_vector(self, action):

        if action == 0:
            return (-1, 0)

        elif action == 1:
            return (1, 0)

        elif action == 2:
            return (0, -1)

        elif action == 3:
            return (0, 1)