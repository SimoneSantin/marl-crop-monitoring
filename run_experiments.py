"""
run_experiments.py

Avvia tutti gli esperimenti in sequenza automaticamente.
Per ogni configurazione gira 3 seed x 1 run = 3 run totali.

Uso:
    python run_experiments.py --computer 1   # computer 1: config 1,2,3
    python run_experiments.py --computer 2   # computer 2: config 4,5
    python run_experiments.py --computer 3   # computer 3: config 6,7
    python run_experiments.py --computer all # tutti (un solo computer)
    python run_experiments.py --config 3_mappo_bayesian  # solo una config
    python run_experiments.py --computer 1 --episodes 200  # episodi ridotti
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.mappo_test import MAPPOTest

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────

ENV_SEEDS    = [42, 123, 456]
NUM_EPISODES = 500  # riduci a 200 per test rapidi

def build_config(experiment_name, use_belief=True, use_gf=True,
                 use_lstm=True, use_random=False):
    return {
        "algorithm":         "MAPPO",
        "experiment_name":   experiment_name,
        "use_belief":        use_belief,
        "use_gf":            use_gf,
        "use_lstm":          use_lstm,
        "use_random_policy": use_random,
        "env": {
            "field_size":  40,
            "num_agents":  3,
            "max_steps":   2000
        },
        "reward": {
            "type": experiment_name,
            "new_cell_weight":      1.0,
            "collision_weight":     1.0,
            "step_penalty":         0.01,
            "alignment_weight":     0.5,
            "completion_bonus":     1.0,
            "completion_threshold": 0.95,
            "accuracy_weight":      1.0
        },
        "training": {
            "num_episodes":    NUM_EPISODES,
            "lr":              5e-5,
            "gamma":           0.99,
            "clip_eps":        0.1,
            "lam":             0.95,
            "epochs":          3,
            "mini_batch_size": 16
        }
    }

ALL_CONFIGS = {
    "1_random_policy":   build_config("random_policy",
                                      use_belief=False, use_gf=False,
                                      use_lstm=False, use_random=True),
    "2_mappo_no_belief": build_config("mappo_no_belief",
                                      use_belief=False, use_gf=False,
                                      use_lstm=False),
    "3_mappo_bayesian":  build_config("mappo_bayesian",
                                      use_belief=True, use_gf=False,
                                      use_lstm=False),
    "4_mappo_bayes_gf":  build_config("mappo_bayes_gf",
                                      use_belief=True, use_gf=True,
                                      use_lstm=False),
    "5_mappo_lstm":      build_config("mappo_lstm",
                                      use_belief=True, use_gf=False,
                                      use_lstm=True),
    "6_mappo_gf_lstm":   build_config("mappo_gf_lstm",
                                      use_belief=True, use_gf=True,
                                      use_lstm=True),
    "7_oracle_conf":     build_config("oracle_confidence",
                                      use_belief=True, use_gf=False,
                                      use_lstm=False),
}

COMPUTER_CONFIGS = {
    "1":   ["1_random_policy", "2_mappo_no_belief", "3_mappo_bayesian"],
    "2":   ["4_mappo_bayes_gf", "5_mappo_lstm"],
    "3":   ["6_mappo_gf_lstm", "7_oracle_conf"],
    "all": list(ALL_CONFIGS.keys()),
}

# ─────────────────────────────────────────────
# FUNZIONI DI SUPPORTO
# ─────────────────────────────────────────────

def get_save_path(config_name, env_seed):
    return os.path.join(
        "results", config_name,
        f"seed_{env_seed}.json"
    )

def is_completed(config_name, env_seed):
    return os.path.exists(get_save_path(config_name, env_seed))

def save_results(config_name, env_seed, metrics):
    save_path = get_save_path(config_name, env_seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    acc_hist = metrics["accuracy"]
    cov_hist = metrics["coverage"]
    col_hist = metrics["collisions"]

    ep_to_075 = next(
        (i for i, a in enumerate(acc_hist) if a >= 0.75),
        len(acc_hist)
    )
    ep_to_080 = next(
        (i for i, a in enumerate(acc_hist) if a >= 0.80),
        len(acc_hist)
    )
    acc_at_cov07 = float(np.mean(
        [a for a, c in zip(acc_hist, cov_hist) if c >= 0.7]
    )) if any(c >= 0.7 for c in cov_hist) else 0.0

    results = {
        "config_name":        config_name,
        "env_seed":           env_seed,
        "final_accuracy":     float(acc_hist[-1]),
        "final_coverage":     float(cov_hist[-1]),
        "mean_collisions":    float(np.mean(col_hist)),
        "auc_accuracy":       float(np.trapz(acc_hist) / len(acc_hist)),
        "episodes_to_075":    int(ep_to_075),
        "episodes_to_080":    int(ep_to_080),
        "acc_at_coverage_07": float(acc_at_cov07),
        "accuracy_history":   [float(x) for x in acc_hist],
        "coverage_history":   [float(x) for x in cov_hist],
        "collisions_history": [float(x) for x in col_hist],
        "alignment_history":  [float(x) for x in metrics.get("alignment", [])],
    }

    if "visited_accuracy" in metrics:
        results["accuracy_visited_history"] = [
            float(x) for x in metrics["visited_accuracy"]
        ]
    if "accuracy" in metrics:
        results["accuracy_unvisited_history"] = [
            float(x) for x in metrics["accuracy"]
        ]

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

def count_missing(config_names):
    return sum(
        1
        for cname in config_names
        for seed in ENV_SEEDS
        if not is_completed(cname, seed)
    )

def print_status(config_name, env_seed,
                 total_configs, config_idx,
                 total_missing, done_so_far,
                 start_time):
    elapsed = time.time() - start_time
    avg_per_run = elapsed / max(done_so_far, 1)
    remaining   = (total_missing - done_so_far) * avg_per_run

    print(f"\n{'='*60}")
    print(f"Config {config_idx}/{total_configs}: {config_name}")
    print(f"Seed: {env_seed}")
    print(f"Completate: {done_so_far}/{total_missing}")
    print(f"Tempo trascorso:  {elapsed/3600:.1f}h")
    if done_so_far > 0:
        print(f"Tempo stimato rimasto: {remaining/3600:.1f}h")
    print(f"{'='*60}")

# ─────────────────────────────────────────────
# RUNNER PRINCIPALE
# ─────────────────────────────────────────────

def run_single(config_name, config, env_seed):
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)

    config = config.copy()
    config["env_seed"] = env_seed

    exp = MAPPOTest(config=config)
    metrics = exp.run(env_seed=env_seed)

    return metrics

def run_all_experiments(config_names):
    """
    Gira tutte le configurazioni in config_names.
    Per ognuna: 3 seed x 1 run = 3 run totali.
    Salta automaticamente le run gia completate.
    """
    start_time    = time.time()
    total_missing = count_missing(config_names)
    done_so_far   = 0

    print(f"\n{'='*60}")
    print(f"AVVIO ESPERIMENTI")
    print(f"Configurazioni: {len(config_names)}")
    print(f"Seed: {ENV_SEEDS}")
    print(f"Run per seed: 1")
    print(f"Run totali mancanti: {total_missing}")
    print(f"Episodi per run: {NUM_EPISODES}")
    print(f"{'='*60}")

    completed = 0
    skipped   = 0
    errors    = 0

    for config_idx, config_name in enumerate(config_names, start=1):
        config = ALL_CONFIGS[config_name]

        for env_seed in ENV_SEEDS:

            if is_completed(config_name, env_seed):
                print(f"[SKIP] {config_name} | seed={env_seed}")
                skipped += 1
                continue

            print_status(
                config_name, env_seed,
                len(config_names), config_idx,
                total_missing, done_so_far,
                start_time
            )

            try:
                metrics = run_single(config_name, config, env_seed)
                results = save_results(config_name, env_seed, metrics)

                print(f"\n✓ Completata: "
                      f"acc={results['final_accuracy']:.3f} | "
                      f"cov={results['final_coverage']:.3f} | "
                      f"auc={results['auc_accuracy']:.3f}")

                completed   += 1
                done_so_far += 1

            except Exception as e:
                print(f"\n✗ ERRORE in {config_name} "
                      f"seed={env_seed}: {e}")

                # salva log errore senza bloccare tutto
                err_path = get_save_path(
                    config_name, env_seed
                ).replace(".json", "_ERROR.txt")
                os.makedirs(os.path.dirname(err_path), exist_ok=True)
                with open(err_path, "w") as f:
                    import traceback
                    f.write(traceback.format_exc())

                errors += 1
                continue

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TUTTO COMPLETATO")
    print(f"Run completate:           {completed}")
    print(f"Run saltate (gia fatte):  {skipped}")
    print(f"Errori:                   {errors}")
    print(f"Tempo totale:             {elapsed/3600:.1f}h")
    print(f"{'='*60}")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esegui gli esperimenti MAPPO"
    )
    parser.add_argument(
        "--computer",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Quale computer (1=config1-3, 2=config4-5, 3=config6-7)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Gira solo una configurazione (es. 3_mappo_bayesian)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help=f"Episodi per run (default {NUM_EPISODES})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Mostra le configurazioni disponibili ed esci"
    )
    args = parser.parse_args()

    # mostra lista configurazioni
    if args.list:
        print("\nConfigurazioni disponibili:")
        for name in ALL_CONFIGS:
            done = sum(1 for s in ENV_SEEDS if is_completed(name, s))
            print(f"  {name:30s} [{done}/{len(ENV_SEEDS)} completate]")
        sys.exit(0)

    # override episodi
    NUM_EPISODES = args.episodes
    for cfg in ALL_CONFIGS.values():
        cfg["training"]["num_episodes"] = NUM_EPISODES

    # seleziona configurazioni
    if args.config is not None:
        if args.config not in ALL_CONFIGS:
            print(f"Config non trovata: {args.config}")
            print(f"Usa --list per vedere le configurazioni disponibili")
            sys.exit(1)
        configs_to_run = [args.config]
    else:
        configs_to_run = COMPUTER_CONFIGS[args.computer]

    print(f"\nConfigurazioni da girare: {configs_to_run}")
    run_all_experiments(configs_to_run)
