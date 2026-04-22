from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from utils.constants import COUNT_MARKER
from env.custom_map import CustomMapEnv
from utils.agent import Agent
from baselines.greedy_planner import GreedyIGPlannerScalar
from baselines.dec_mcts_planner import DecMCTSPlanner


def moving_average(values, window=10):
    if not values:
        return []
    if len(values) < window:
        window = len(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def create_experiment_dirs(experiment_type):
    base_dir = os.path.join("results", f"MCTS_{experiment_type}")
    greedy_dir = os.path.join(base_dir, "greedyIG")
    mcts_dir = os.path.join(base_dir, "DecMCTS")

    os.makedirs(greedy_dir, exist_ok=True)
    os.makedirs(mcts_dir, exist_ok=True)

    return base_dir, greedy_dir, mcts_dir


def save_plots(metrics, plots_dir, field_size=None):
    coverage = metrics.get("coverage")
    lengths = metrics.get("lengths")
    accuracy = metrics.get("accuracy")
    alignment = metrics.get("alignment")
    episode_paths = metrics.get("episode_paths")

    if alignment:
        plt.figure()
        plt.plot(alignment, alpha=0.3, label="alignment")
        ma = moving_average(alignment)
        if len(ma) > 0:
            plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Episode Alignment")
        plt.xlabel("Step")
        plt.ylabel("Alignment")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "alignment_plot.png"))
        plt.close()

    if coverage:
        plt.figure()
        plt.plot(coverage, alpha=0.3, label="coverage")
        ma = moving_average(coverage)
        if len(ma) > 0:
            plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Map Coverage")
        plt.xlabel("Step")
        plt.ylabel("Visited ratio")
        plt.savefig(os.path.join(plots_dir, "coverage_plot.png"))
        plt.close()

    if lengths:
        plt.figure()
        plt.plot(lengths, alpha=0.3, label="lengths")
        ma = moving_average(lengths)
        if len(ma) > 0:
            plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Episode Length")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.savefig(os.path.join(plots_dir, "episode_length_plot.png"))
        plt.close()

    if accuracy:
        plt.figure()
        plt.plot(accuracy, alpha=0.3, label="accuracy")
        ma = moving_average(accuracy)
        if len(ma) > 0:
            plt.plot(range(len(ma)), ma, label="moving avg")
        plt.title("Accuracy per Episode")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(plots_dir, "accuracy_plot.png"))
        plt.close()

    if episode_paths and field_size is not None:
        plt.figure(figsize=(6, 6))
        for i, path in enumerate(episode_paths):
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            plt.plot(xs, ys, marker="o", markersize=2, label=f"Agent {i}")

        plt.xlim(0, field_size - 1)
        plt.ylim(0, field_size - 1)
        plt.title("Agent Trajectories")
        plt.xlabel("Y")
        plt.ylabel("X")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plots_dir, "agent_trajectories.png"))
        plt.close()


    print(f"Plots saved in {plots_dir}")


def run_greedy_ig(env, plots_dir):
    planner = GreedyIGPlannerScalar(
        field_size=env.field_size,
        num_classes=COUNT_MARKER,
        scalar_sensor=env.scalar_sensor
    )

    agent = Agent(env, COUNT_MARKER, agent_id=0, planner=planner)
    obs, _ = env.reset()
    agent.reset()

    accuracy_history = []
    alignment_history = []
    total_new_cells = 0
    coverage_history = []
    length_history = []
    episode_paths = []
    step = 0
    done = False

    while not done:
        obs_i = obs[0]

        action = agent.choose_action(obs_i)

        next_obs, reward, terminated, truncated, info = env.step([action])

        next_obs_i = next_obs[0]

        alignment_patch = next_obs_i[:9]
        sensor_patch_flat = next_obs_i[9:9 + 9 * COUNT_MARKER]
        sensor_patch = sensor_patch_flat.reshape(9, COUNT_MARKER)

        agent.update_belief_patch(
            sensor_patch=sensor_patch,
            alignment_patch=alignment_patch,
            gamma=3.0
        )

        accuracy = agent.compute_accuracy()
        accuracy_history.append(accuracy)

        alignment_history.append(info["reward_terms"]["alignment"])

        total_new_cells += info["new_cells"]
        visited_ratio = total_new_cells / (env.field_size * env.field_size)
        coverage_history.append(visited_ratio)

        obs = next_obs
        step += 1
        length_history.append(step)
        pos = env.agent_pos[0]
        episode_paths.append(tuple(pos))

        done = terminated or truncated

    metrics = {
        "accuracy": accuracy_history,
        "alignment": alignment_history,
        "lengths": length_history,
        "coverage": coverage_history,
        "episode_paths": [episode_paths],
    }

    save_plots(metrics, plots_dir, field_size=env.field_size)
    return metrics


def run_dec_mcts(env, plots_dir):
    planner = DecMCTSPlanner(env)

    agent = Agent(env, COUNT_MARKER, agent_id=0, planner=planner)
    obs, _ = env.reset()
    agent.reset()

    accuracy_history = []
    alignment_history = []
    total_new_cells = 0
    coverage_history = []
    length_history = []
    episode_paths = []
    step = 0
    done = False

    while not done:
        obs_i = obs[0]
        if step % 20 == 0:
            print(f"Step mcts {step}")
        action = agent.choose_action(obs_i)

        next_obs, reward, terminated, truncated, info = env.step([action])

        next_obs_i = next_obs[0]

        alignment_patch = next_obs_i[:9]
        sensor_patch_flat = next_obs_i[9:9 + 9 * COUNT_MARKER]
        sensor_patch = sensor_patch_flat.reshape(9, COUNT_MARKER)

        agent.update_belief_patch(
            sensor_patch=sensor_patch,
            alignment_patch=alignment_patch,
            gamma=3.0
        )

        accuracy = agent.compute_accuracy()
        accuracy_history.append(accuracy)

        alignment_history.append(info["reward_terms"]["alignment"])

        total_new_cells += info["new_cells"]
        visited_ratio = total_new_cells / (env.field_size * env.field_size)
        coverage_history.append(visited_ratio)

        obs = next_obs
        step += 1
        length_history.append(step)
        pos = env.agent_pos[0]
        episode_paths.append(tuple(pos))

        done = terminated or truncated

    metrics = {
        "accuracy": accuracy_history,
        "alignment": alignment_history,
        "lengths": length_history,
        "coverage": coverage_history,
        "episode_paths": [episode_paths],
    }

    save_plots(metrics, plots_dir, field_size=env.field_size)
    return metrics


if __name__ == "__main__":
    experiment_type = "20x20"

    base_dir, greedy_dir, mcts_dir = create_experiment_dirs(experiment_type)

    env = CustomMapEnv(20)

    greedy_metrics = run_greedy_ig(env, greedy_dir)
    mcts_metrics = run_dec_mcts(env, mcts_dir)

