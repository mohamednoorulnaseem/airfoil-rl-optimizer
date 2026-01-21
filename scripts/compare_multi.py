import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO

try:
    from src.aerodynamics.legacy_eval import aero_score_multi
    from src.optimization.single_objective_env import AirfoilEnv
except ImportError:
    # Just in case
    from src.aerodynamics.legacy_eval import aero_score_multi
    from src.optimization.single_objective_env import AirfoilEnv

def find_best_params(model, env, alphas, n_episodes=10):
    best_params = None
    best_ld_mean = -1.0
    best_lds = None

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        # after episode, use env.params as candidate
        m, p, t = env.params
        cls, cds, lds = aero_score_multi(m, p, t, alphas=alphas)
        ld_mean = float(np.mean(lds))

        if ld_mean > best_ld_mean:
            best_ld_mean = ld_mean
            best_params = env.params.copy()
            best_lds = lds

    return best_params, best_ld_mean, best_lds


def main():
    # Define angles of attack (deg) here instead of using env.alphas
    alphas = np.array([0.0, 4.0, 8.0], dtype=np.float32)

    env = AirfoilEnv()
    # Path to model might need adjustment
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ppo_airfoil_fake")
    if not os.path.exists(model_path + ".zip"):
         model_path = "ppo_airfoil_fake" # Try local if above fails
         
    try:
        model = PPO.load(model_path, env=env)
    except:
        print(f"Could not load model at {model_path}")
        return

    # Baseline params (like NACA 2412-ish)
    base_params = np.array([0.02, 0.4, 0.12], dtype=np.float32)
    base_cls, base_cds, base_lds = aero_score_multi(
        base_params[0], base_params[1], base_params[2], alphas=alphas
    )

    # RL-optimized params
    best_params, best_ld_mean, best_lds = find_best_params(model, env, alphas, n_episodes=10)

    print("Alphas (deg):", alphas)
    print("Baseline params [m, p, t]:", base_params)
    print("Optimized params [m, p, t]:", best_params)
    print()

    print("AoA |  L/D baseline  |  L/D optimized")
    print("----------------------------------------")
    for alpha, ld_b, ld_o in zip(alphas, base_lds, best_lds):
        print(f"{alpha:4.1f} | {ld_b:12.2f} | {ld_o:13.2f}")

    print()
    print(f"Mean L/D baseline:  {np.mean(base_lds):.2f}")
    print(f"Mean L/D optimized: {best_ld_mean:.2f}")


if __name__ == "__main__":
    main()
