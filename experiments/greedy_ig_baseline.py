import numpy as np
import matplotlib.pyplot as plt

from utils.constants import COUNT_MARKER
from env.custom_map import CustomMapEnv

from baselines.greedy_planner import GreedyIGPlannerScalar
from baselines.dec_mcts_planner import DecMCTSPlanner


def plot_belief_heatmap(belief, title, step=None, cls=None):
    if cls is None:
        plot_map = np.max(belief, axis=2)
        label = "Probabilità Massima"

    plt.figure(figsize=(6,6))
    plt.imshow(plot_map, cmap='viridis', origin='lower', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label=label)

    if step is not None:
        plt.title(f"{title} (Step {step})")
        filename = f"results/plots/{title} belief_step_{step:04d}.png"
    else:
        plt.title(title)
        filename = f"results/plots/{title.replace(' ','_')}.png"

    plt.xlabel("Colonna")
    plt.ylabel("Riga")
    plt.savefig(filename)
    plt.close()


env = CustomMapEnv()

planner = GreedyIGPlannerScalar(
    field_size=env.field_size,
    num_classes=COUNT_MARKER,
    scalar_sensor=env.scalar_sensor
)

obs, _ = env.reset()

belief_map = np.ones(
    (env.field_size, env.field_size, COUNT_MARKER)
) / COUNT_MARKER

accuracy_history_greedy = []

done = False
step = 0

while not done:

    if step % 200 == 0 or step == 999:
        plot_belief_heatmap(belief_map, title="Belief Map Greedy", step=step)
    sensor_dist = obs[9:]

    x, y = env.agent_pos

    # Bayesian update
    belief_map[x, y] *= sensor_dist
    belief_map[x, y] /= belief_map[x, y].sum()

    pred_map = np.argmax(belief_map, axis=2)
    true_map = env.grid_counts
    angles = env.grid_angles
    accuracy = (pred_map == true_map).mean()
    accuracy_history_greedy.append(accuracy)

    action = planner.choose_action(env.agent_pos, belief_map, angles)

    obs, reward, terminated, truncated, _ = env.step(action)
    step += 1
    done = terminated or truncated


planner = DecMCTSPlanner(env)

obs, _ = env.reset()

belief_map = np.ones(
    (env.field_size, env.field_size, COUNT_MARKER)
) / COUNT_MARKER

accuracy_history_mcts = []

done = False
step = 0
while not done:

    if step % 200 == 0 or step == 999:
        plot_belief_heatmap(belief_map, title="Belief Map Dec-MCTS", step=step)

    sensor_dist = obs[9:]

    x, y = env.agent_pos

    # Bayesian update
    belief_map[x, y] *= sensor_dist
    belief_map[x, y] /= belief_map[x, y].sum()

    pred_map = np.argmax(belief_map, axis=2)
    true_map = env.grid_counts

    angles = env.grid_angles
    accuracy = (pred_map == true_map).mean()
    accuracy_history_mcts.append(accuracy)

    action = planner.choose_action(env.agent_pos, belief_map, angles)

    obs, reward, terminated, truncated, _ = env.step(action)
    print("Action:", action, "Reward:", reward, "Accuracy:", accuracy) 
    step += 1
    done = terminated or truncated


plt.plot(accuracy_history_greedy, label="Greedy IG")
plt.plot(accuracy_history_mcts, label="Dec-MCTS")

plt.xlabel("Steps")
plt.ylabel("Map Accuracy")

plt.title("Greedy IG vs Dec-MCTS")

plt.legend()

plt.savefig("results/plots/accuracy_comparison.png")

