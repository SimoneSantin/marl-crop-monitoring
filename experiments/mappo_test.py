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
    def __init__(self, config_path=None, config=None):
        self.config_path = config_path
        self.config = config
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
            "experiment_name": "FINALMAPPO4",
            "env": {
                "field_size": 40,
                "num_agents": 3,
                "max_steps": 2000
            },
            "reward": {
                "type": "best_reward",
                "formula": "reward = new_cell * 0.2 - 0.01 + localAccuracy * 5.0, completion_bonus",
                "new_cell_weight": 1.0,
                "collision_weight": 1.0,
                "step_penalty": 0.01,
                "alignment_weight": 0.5,
                "completion_bonus": 1.0,
                "completion_threshold": 0.95,
                "accuracy_weight": 1.0
            },
            "training": {
                "num_episodes": 500,
                "lr": 5e-05,
                "gamma": 0.99,
                "clip_eps": 0.1,
                "lam": 0.95,
                "epochs": 3,
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

        self.run_dir = os.path.join("results3", run_name)
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.models_dir = os.path.join(self.run_dir, "models")

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def moving_average(self, data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def plot_training(self, metrics):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        rewards = metrics["rewards"]
        coverage = metrics["coverage"]
        lengths = metrics["lengths"]
        collisions = metrics["collisions"]
        accuracy = metrics["accuracy"]
        visited_accuracy = metrics.get("visited_accuracy", None)
        unvisited_accuracy = metrics.get("unvisited_accuracy", None)
        accuracy_traces = metrics["accuracy_traces"]

        os.makedirs(self.plots_dir, exist_ok=True)

        # 1. Final accuracy per episode
        plt.figure()
        plt.plot(accuracy, alpha=0.35, label="global accuracy")
        ma = self.moving_average(accuracy)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Global Accuracy per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "global_accuracy_per_episode.png"))
        plt.close()

        # 2. Visited accuracy per episode
        if visited_accuracy is not None:
            plt.figure()
            plt.plot(visited_accuracy, alpha=0.35, label="visited accuracy")
            ma = self.moving_average(visited_accuracy)
            plt.plot(range(len(ma)), ma, label="moving avg")
            plt.title("Visited-Cell Accuracy per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, "visited_accuracy_per_episode.png"))
            plt.close()

        # 3. Global vs visited accuracy
        if visited_accuracy is not None:
            plt.figure()
            plt.plot(accuracy, alpha=0.35, label="global accuracy")
            plt.plot(visited_accuracy, alpha=0.35, label="visited accuracy")
            plt.title("Global Accuracy vs Visited-Cell Accuracy")
            plt.xlabel("Episode")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, "global_vs_visited_accuracy.png"))
            plt.close()

        # 4. Coverage per episode
        plt.figure()
        plt.plot(coverage, alpha=0.35, label="coverage")
        ma = self.moving_average(coverage)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Coverage per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Visited ratio")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "coverage_per_episode.png"))
        plt.close()

        # 5. Collisions per episode
        plt.figure()
        plt.plot(collisions, alpha=0.35, label="collisions")
        ma = self.moving_average(collisions)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Collisions per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Collisions")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "collisions_per_episode.png"))
        plt.close()

        # 6. Accuracy at equal coverage
        plt.figure()
        plt.scatter(coverage, accuracy, s=12, alpha=0.5)
        plt.title("Accuracy at Equal Coverage")
        plt.xlabel("Coverage")
        plt.ylabel("Global accuracy")
        plt.savefig(os.path.join(self.plots_dir, "accuracy_vs_coverage.png"))
        plt.close()

        # 7. Accuracy traces during selected episodes
        plt.figure()
        for episode, trace in accuracy_traces.items():
            steps = np.arange(len(trace))
            plt.plot(steps, trace, label=f"episode {episode}")

        plt.title("Accuracy During Selected Episodes")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "accuracy_traces.png"))
        plt.close()

        # 8. AUC accuracy during selected episodes
        auc_values = {}
        for episode, trace in accuracy_traces.items():
            trace = np.asarray(trace, dtype=np.float32)
            if len(trace) > 1:
                auc_values[episode] = np.trapz(trace) / (len(trace) - 1)
            else:
                auc_values[episode] = np.nan

        if len(auc_values) > 0:
            plt.figure()
            episodes = list(auc_values.keys())
            aucs = list(auc_values.values())
            plt.bar([str(e) for e in episodes], aucs)
            plt.title("Accuracy AUC During Selected Episodes")
            plt.xlabel("Episode")
            plt.ylabel("AUC")
            plt.savefig(os.path.join(self.plots_dir, "accuracy_auc_selected_episodes.png"))
            plt.close()

        # 9. Episodes needed to reach target accuracy
        thresholds = [0.75, 0.80]
        reached = []

        for threshold in thresholds:
            acc_arr = np.asarray(accuracy, dtype=np.float32)
            idx = np.where(acc_arr >= threshold)[0]

            if len(idx) > 0:
                reached.append(idx[0])
            else:
                reached.append(np.nan)

        plt.figure()
        plt.bar([str(t) for t in thresholds], reached)
        plt.title("Episodes Needed to Reach Target Accuracy")
        plt.xlabel("Target accuracy")
        plt.ylabel("Episode")
        plt.savefig(os.path.join(self.plots_dir, "episodes_to_target_accuracy.png"))
        plt.close()

        # 10. Episode length
        plt.figure()
        plt.plot(lengths, alpha=0.35, label="episode length")
        ma = self.moving_average(lengths)
        plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Episode Length")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "episode_length.png"))
        plt.close()

        # 11. Agent trajectories
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

        # grafico unvisited accuracy
        if unvisited_accuracy is not None:
            plt.figure()
            plt.plot(unvisited_accuracy, alpha=0.35, label="unvisited accuracy")
            ma = self.moving_average(unvisited_accuracy)
            plt.plot(range(len(ma)), ma, label="moving avg")
            plt.title("Unvisited-Cell Accuracy per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, "unvisited_accuracy_per_episode.png"))
            plt.close()

        # grafico tutte e tre le accuracy insieme
        if visited_accuracy is not None and unvisited_accuracy is not None:
            plt.figure()
            plt.plot(accuracy,            alpha=0.35, label="global accuracy")
            plt.plot(visited_accuracy,    alpha=0.35, label="visited accuracy")
            plt.plot(unvisited_accuracy,  alpha=0.35, label="unvisited accuracy")
            plt.title("Global vs Visited vs Unvisited Accuracy")
            plt.xlabel("Episode")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(os.path.join(
                self.plots_dir, "global_vs_visited_vs_unvisited_accuracy.png"
            ))
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

        self.env.reset(
            seed=self.config.get("env_seed", None)
        )
        obs_dim = self.env.obs_dim
        action_dim = self.env.action_space.n
        num_agents = self.env.num_agents
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_gaussian      = self.config.get("use_gaussian", False)
        use_lstm          = self.config.get("use_lstm", False)
        use_cell_lstm     = self.config.get("use_cell_lstm", False)
        use_random_policy = self.config.get("use_random_policy", False)  # ← AGGIUNTO

        self.agents = [
            Agent(
                self.env, COUNT_MARKER, agent_id=i, planner=None,
                device=device,
                use_gaussian=use_gaussian,
                use_lstm=use_lstm,
                use_cell_lstm=use_cell_lstm,
                cell_lstm_path="./LSTM/models/cell_observer_lstm.pth",
                cell_lstm_hidden=128,
                use_random_policy=use_random_policy,   # ← AGGIUNTO
            )
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
            },
            config=self.config
        )
        
    def run(self, env_seed=None):
        if self.config is not None:
            pass
        elif self.config_path is not None:
            self.load_config(self.config_path)
        else:
            self.config = self.build_default_config()

        if env_seed is not None:
            np.random.seed(env_seed)
            torch.manual_seed(env_seed)
            self.config["env_seed"] = env_seed

        self.create_run_dir()
        self.save_config()
        self.setup()

        metrics = self.trainer.train()
        self.plot_training(metrics)
        self.save_models()

        return metrics

def main():
    experiment = MAPPOTest(config_path=None)
    experiment.run()


if __name__ == "__main__":
    main()