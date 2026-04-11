import os
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.agent import Agent
from env.custom_map import CustomMapEnv
from marl.mappo_policy import MAPPOPlannerMultiAgent
from training.mappo_training import MAPPOTrainer
from utils.constants import COUNT_MARKER


class MAPPOTest:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = None
        self.run_dir = None
        self.plots_dir = None
        self.models_dir = None

        self.env = None
        self.agents = None
        self.planner = None
        self.trainer = None

    def build_default_config(self):
        return {
            "algorithm": "MAPPO",
            "experiment_name": "CNN-LSTM",
            "env": {
                "field_size": 40,
                "num_agents": 3,
                "max_steps": 2000
            },
            "reward": {
                "type": "collision_aware",
                "formula": "reward = new_cell * 0.2 - 0.01 + localAccuracy * 5.0, completion_bonus",
                "new_cell_weight": 0.5,
                "collision_weight": 2.0,
                "step_penalty": 0.01,
                "alignment_weight": 2.0,
                "completion_bonus": 0.0,
                "completion_threshold": 0.95,
                "accuracy_weight": 1.0
            },
            "training": {
                "num_episodes": 500,
                "lr": 0.0001,
                "gamma": 0.99,
                "clip_eps": 0.1,
                "lam": 0.95,
                "epochs": 5,
                "mini_batch_size": 16
            }
        }

    def load_config(self, path):
        with open(path, "r") as f:
            self.config = json.load(f)

    def save_config(self):
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def create_run_dir(self):
        algo = self.config["algorithm"]
        exp_name = self.config["experiment_name"]
        field_size = self.config["env"]["field_size"]
        num_agents = self.config["env"]["num_agents"]
        reward_type = self.config["reward"]["type"]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_name = f"{algo}_{exp_name}_fs{field_size}_agents{num_agents}_{reward_type}_{timestamp}"

        self.run_dir = os.path.join("results", run_name)
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.models_dir = os.path.join(self.run_dir, "models")

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def moving_average(self, data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def plot_training(self, metrics):
        rewards = metrics["rewards"]
        coverage = metrics["coverage"]
        lengths = metrics["lengths"]
        new_cells = metrics["new_cells"]
        collisions = metrics["collisions"]
        efficiency = metrics["efficiency"]
        accuracy = metrics["accuracy"]
        alignment = metrics["alignment"]

        #check metriche
        terms = metrics["terms"]
        new_cells_terms = [t["new_cells"] for t in terms]
        collisions_terms = [t["collisions"] for t in terms]
        #step_terms = [t["step"] for t in terms]
        alignment_terms = [t["alignment"] for t in terms]
        accuracy_terms = [t["accuracy"] for t in terms]
  

        plt.figure()
        plt.plot(alignment, alpha=0.3, label="alignment")
        ma = self.moving_average(alignment)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Episode Alignment")
        plt.xlabel("Episode")
        plt.ylabel("Alignment")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "alignment_plot.png"))
        plt.close()
        plt.figure()

        plt.plot(rewards, alpha=0.3, label="reward")
        ma = self.moving_average(rewards)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "reward_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(coverage, alpha=0.3, label="coverage")
        ma = self.moving_average(coverage)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Map Coverage")
        plt.xlabel("Episode")
        plt.ylabel("Visited ratio")
        plt.savefig(os.path.join(self.plots_dir, "coverage_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(lengths, alpha=0.3, label="lengths")
        ma = self.moving_average(lengths)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Episode Length")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.savefig(os.path.join(self.plots_dir, "episode_length_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(new_cells, alpha=0.3, label="new_cells")
        ma = self.moving_average(new_cells)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("New Cells per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Cells discovered")
        plt.savefig(os.path.join(self.plots_dir, "new_cells_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(collisions, alpha=0.3, label="collisions")
        ma = self.moving_average(collisions)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Collisions per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Collisions")
        plt.savefig(os.path.join(self.plots_dir, "collisions_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(accuracy, alpha=0.3, label="accuracy")
        ma = self.moving_average(accuracy)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Accuracy per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(self.plots_dir, "accuracy_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(efficiency, alpha=0.3, label="efficiency")
        ma = self.moving_average(efficiency)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Efficiency per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Efficiency")
        plt.savefig(os.path.join(self.plots_dir, "efficiency_plot.png"))
        plt.close()

        plt.figure(figsize=(6, 6))
        normalized = metrics["visit_heatmap"] / np.max(metrics["visit_heatmap"])
        plt.imshow(normalized, cmap="viridis", interpolation="nearest", origin="lower")
        plt.title("Visit Heatmap")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar()
        plt.title("Cumulative Visited Heatmap")
        plt.savefig(os.path.join(self.plots_dir, "visit_heatmap.png"))
        plt.close()

        plt.figure(figsize=(6, 6))
        for i, path in enumerate(metrics["episode_paths"]):
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            plt.plot(xs, ys, marker="o", markersize=2, label=f"Agent {i}")

        plt.xlim(0, self.config["env"]["field_size"] - 1)
        plt.ylim(0, self.config["env"]["field_size"] - 1)
        plt.title("Agent Trajectories")
        plt.xlabel("Y")
        plt.ylabel("X")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(self.plots_dir, "agent_trajectories.png"))
        plt.close()

        plt.figure()

        plt.plot(self.moving_average(new_cells_terms), label="new_cells")
        plt.plot(self.moving_average(collisions_terms), label="collisions")
        plt.plot(self.moving_average(alignment_terms), label="alignment")
        plt.plot(self.moving_average(accuracy_terms), label="accuracy")

        plt.title("Reward Terms Contribution (Smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Average contribution per step")
        plt.legend()

        plt.savefig(os.path.join(self.plots_dir, "reward_terms_plot.png"))
        plt.close()
        print(f"Plots saved in {self.plots_dir}")

    def save_models(self):
        
        torch.save(
            self.planner.actor.state_dict(),
            os.path.join(self.models_dir, f"actor_.pt")
        )

        torch.save(
            self.planner.critic.state_dict(),
            os.path.join(self.models_dir, "critic.pt")
        )

        print(f"Models saved in {self.models_dir}")

    def setup(self):
        env_cfg = self.config["env"]
        train_cfg = self.config["training"]
        reward_cfg = self.config["reward"]

        self.env = CustomMapEnv(
            field_size=env_cfg["field_size"],
            num_agents=env_cfg["num_agents"],
            max_steps=env_cfg["max_steps"],
            reward_config=reward_cfg,
            algorithm=self.config["algorithm"]
        )

        self.env.reset()

        obs_dim = self.env.obs_dim
        action_dim = self.env.action_space.n
        num_agents = self.env.num_agents

        self.agents = [
            Agent(self.env, COUNT_MARKER, agent_id=i, planner=None)
            for i in range(num_agents)
        ]

        if self.config["algorithm"] == "MAPPO":
            self.planner = MAPPOPlannerMultiAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                agents=self.agents,
                field_size=self.env.field_size,
                num_classes=COUNT_MARKER,
                lr=train_cfg["lr"],
                gamma=train_cfg["gamma"],
                clip_eps=train_cfg["clip_eps"],
                lam=train_cfg["lam"],
                epochs=train_cfg["epochs"],
                mini_batch_size=train_cfg["mini_batch_size"]
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config['algorithm']}")

        for agent in self.agents:
            agent.planner = self.planner

        self.trainer = MAPPOTrainer(
            env=self.env,
            planner=self.planner,
            num_episodes=train_cfg["num_episodes"],
            reward_weights={
                "accuracy": reward_cfg["accuracy_weight"]
            }
        )

    def run(self):
        if self.config_path is not None:
            self.load_config(self.config_path)
        else:
            self.config = self.build_default_config()

        self.create_run_dir()
        self.save_config()
        self.setup()

        metrics = self.trainer.train()
        self.plot_training(metrics)
        self.save_models()


def main():
    experiment = MAPPOTest(config_path=None)
    experiment.run()


if __name__ == "__main__":
    main()