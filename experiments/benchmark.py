import cProfile
import pstats
import io
import numpy as np
import time
 
# simula un episodio con le stesse dimensioni del tuo sistema
# per trovare dove va il tempo senza girare tutto MAPPO
 
FIELD_SIZE = 40
NUM_CLASSES = 10
NUM_AGENTS = 3
MAX_STEPS = 2000
INFERENCE_RADIUS = 5
SIGMA = 2.5
 
 
def precompute_kernel(inference_radius, sigma):
    r = inference_radius
    offsets = []
    weights = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            dist2 = dx * dx + dy * dy
            if dist2 <= r * r and dist2 > 0:
                w = np.exp(-dist2 / (2 * sigma ** 2))
                offsets.append([dx, dy])
                weights.append(w)
    return np.array(offsets, dtype=np.int32), np.array(weights, dtype=np.float32)
 
 
def gaussian_field_step(belief_map, observed_mask, observed_cells, offsets, weights):
    H, W, C = belief_map.shape
    accumulator = np.zeros((H, W, C), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
 
    for ox, oy in observed_cells:
        center_belief = belief_map[ox, oy]
        neighbors = np.array([ox, oy]) + offsets
 
        in_grid = (
            (neighbors[:, 0] >= 0) & (neighbors[:, 0] < H) &
            (neighbors[:, 1] >= 0) & (neighbors[:, 1] < W)
        )
        neighbors = neighbors[in_grid]
        w = weights[in_grid]
 
        if len(neighbors) == 0:
            continue
 
        nx = neighbors[:, 0]
        ny = neighbors[:, 1]
 
        not_observed = observed_mask[nx, ny] == 0
        nx = nx[not_observed]
        ny = ny[not_observed]
        w = w[not_observed]
 
        if len(nx) == 0:
            continue
 
        accumulator[nx, ny] += w[:, np.newaxis] * center_belief
        weight_sum[nx, ny] += w
 
    has_contrib = (weight_sum > 0) & (observed_mask == 0)
    if np.any(has_contrib):
        inferred = accumulator[has_contrib] / (weight_sum[has_contrib, np.newaxis] + 1e-9)
        inferred /= (inferred.sum(axis=1, keepdims=True) + 1e-9)
        belief_map[has_contrib] = inferred
 
 
def bayesian_update(belief_map, observed_mask, x, y, sensor_patch, alignment_patch, gamma=3.0):
    observed_cells = []
    idx = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            cx, cy = x + dx, y + dy
            if 0 <= cx < FIELD_SIZE and 0 <= cy < FIELD_SIZE:
                sensor_dist = sensor_patch[idx]
                alignment = alignment_patch[idx]
                reliability = alignment ** gamma
                likelihood = sensor_dist ** reliability
                likelihood /= (likelihood.sum() + 1e-9)
                old = belief_map[cx, cy]
                updated = old * likelihood
                updated /= (updated.sum() + 1e-9)
                belief_map[cx, cy] = updated
                observed_mask[cx, cy] = 1
                observed_cells.append((cx, cy))
            idx += 1
    return observed_cells
 
 
def simulate_episode():
    offsets, weights = precompute_kernel(INFERENCE_RADIUS, SIGMA)
 
    belief_maps = [
        np.ones((FIELD_SIZE, FIELD_SIZE, NUM_CLASSES), dtype=np.float32) / NUM_CLASSES
        for _ in range(NUM_AGENTS)
    ]
    observed_masks = [
        np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        for _ in range(NUM_AGENTS)
    ]
 
    # timing per sezione
    t_bayes = 0.0
    t_gf = 0.0
    t_uncertainty = 0.0
    t_enrich = 0.0
 
    for step in range(MAX_STEPS):
        for agent_id in range(NUM_AGENTS):
            # posizione casuale (simula movimento)
            x = np.random.randint(1, FIELD_SIZE - 1)
            y = np.random.randint(1, FIELD_SIZE - 1)
 
            # osservazioni casuali
            sensor_patch = np.random.dirichlet(
                np.ones(NUM_CLASSES), size=9
            ).astype(np.float32)
            alignment_patch = np.random.rand(9).astype(np.float32)
 
            # --- Bayesian update ---
            t0 = time.perf_counter()
            observed_cells = bayesian_update(
                belief_maps[agent_id], observed_masks[agent_id],
                x, y, sensor_patch, alignment_patch
            )
            t_bayes += time.perf_counter() - t0
 
            # --- Gaussian Field ---
            t0 = time.perf_counter()
            gaussian_field_step(
                belief_maps[agent_id], observed_masks[agent_id],
                observed_cells, offsets, weights
            )
            t_gf += time.perf_counter() - t0
 
        # --- Shared uncertainty map (compute_shared_uncertainty_map) ---
        t0 = time.perf_counter()
        beliefs = np.stack(belief_maps, axis=0)
        entropy = -np.sum(beliefs * np.log(beliefs + 1e-9), axis=-1)
        entropy /= np.log(NUM_CLASSES)
        shared_uncertainty = np.mean(entropy, axis=0)
        t_uncertainty += time.perf_counter() - t0
 
        # --- Enrich obs (per agente) ---
        t0 = time.perf_counter()
        for agent_id in range(NUM_AGENTS):
            x = np.random.randint(1, FIELD_SIZE - 1)
            y = np.random.randint(1, FIELD_SIZE - 1)
 
            entropy_patch = np.zeros((3, 3), dtype=np.float32)
            agents_patch = np.zeros((3, 3), dtype=np.float32)
 
            for i in range(-1, 2):
                for j in range(-1, 2):
                    nx, ny = x + i, y + j
                    if 0 <= nx < FIELD_SIZE and 0 <= ny < FIELD_SIZE:
                        belief = belief_maps[agent_id][nx, ny]
                        ent = -np.sum(belief * np.log(belief + 1e-9))
                        ent /= np.log(NUM_CLASSES)
                        entropy_patch[i+1, j+1] = ent
 
            # downsample 4x4
            coarse_size = 4
            block = FIELD_SIZE // coarse_size
            coarse = np.zeros((coarse_size, coarse_size), dtype=np.float32)
            for bi in range(coarse_size):
                for bj in range(coarse_size):
                    xs = bi * block
                    xe = (bi + 1) * block if bi < coarse_size - 1 else FIELD_SIZE
                    ys = bj * block
                    ye = (bj + 1) * block if bj < coarse_size - 1 else FIELD_SIZE
                    coarse[bi, bj] = np.mean(shared_uncertainty[xs:xe, ys:ye])
 
        t_enrich += time.perf_counter() - t0
 
    print(f"\n=== PROFILING RISULTATI (1 episodio, {MAX_STEPS} step, {NUM_AGENTS} agenti) ===")
    print(f"Bayesian update:       {t_bayes:.2f}s  ({t_bayes/MAX_STEPS*1000:.1f}ms/step)")
    print(f"Gaussian Field:        {t_gf:.2f}s  ({t_gf/MAX_STEPS*1000:.1f}ms/step)")
    print(f"Shared uncertainty:    {t_uncertainty:.2f}s  ({t_uncertainty/MAX_STEPS*1000:.1f}ms/step)")
    print(f"Enrich obs:            {t_enrich:.2f}s  ({t_enrich/MAX_STEPS*1000:.1f}ms/step)")
    total = t_bayes + t_gf + t_uncertainty + t_enrich
    print(f"Totale misurato:       {total:.2f}s")
    print(f"(il resto e MAPPO, env step, ecc.)")
 
 
if __name__ == "__main__":
    t_start = time.perf_counter()
    simulate_episode()
    t_total = time.perf_counter() - t_start
    print(f"\nTempo totale simulazione: {t_total:.2f}s")