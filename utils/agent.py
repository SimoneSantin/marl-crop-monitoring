import numpy as np

class Agent:

    def __init__(self, env, num_classes, agent_id, planner):

        self.env = env
        self.planner = planner
        self.agent_id = agent_id

        self.field_size = env.field_size
        self.num_classes = num_classes
        self.belief_map = None

        self.reset()

    def reset(self):

        self.belief_map = np.ones(
            (self.field_size, self.field_size, self.num_classes)
        ) / self.num_classes


    def update_belief(self, sensor_dist):
        
        x, y = self.env.agent_pos[self.agent_id]

        self.belief_map[x, y] *= sensor_dist
        self.belief_map[x, y] /= (self.belief_map[x, y].sum() + 1e-9)


    def choose_action(self, obs):

        pos = self.env.agent_pos
        angles = self.env.grid_angles

        # planner tipo Greedy o MCTS
        if hasattr(self.planner, "choose_action"):

            return self.planner.choose_action(
                pos[0],
                self.belief_map,
                angles
            )

        # planner RL (MAPPO / PPO / ecc)
        elif hasattr(self.planner, "act"):

            action, log_prob = self.planner.act(obs, self.agent_id)

            return action, log_prob

        else:
            raise ValueError("Planner interface not recognized")



    def get_prediction_map(self):

        return np.argmax(self.belief_map, axis=2)

    def compute_accuracy(self):

        pred_map = self.get_prediction_map()
        true_map = self.env.grid_counts

        accuracy = (pred_map == true_map).mean()

        return accuracy