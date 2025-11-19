import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from airfoil_env import AirfoilEnv
from airfoil_gen import naca4


def run_episodes(model, env, n_episodes=20):
    best_ld = -1.0
    best_params = None

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            ld = info["L/D"]
            if ld > best_ld:
                best_ld = ld
                best_params = env.params.copy()

        print(
            f"Episode {ep}: total_reward={total_reward:.2f}, "
            f"best L/D so far={best_ld:.2f}"
        )

    return best_params, best_ld


def plot_airfoils(best_params, best_ld):
    # baseline NACA-like
    base_m, base_p, base_t = 0.02, 0.4, 0.12
    xu_b, yu_b, xl_b, yl_b = naca4(base_m, base_p, base_t)

    # RL-optimized
    m, p, t = best_params
    xu_o, yu_o, xl_o, yl_o = naca4(float(m), float(p), float(t))

    plt.figure()
    plt.plot(xu_b, yu_b, label=f"Baseline upper")
    plt.plot(xl_b, yl_b, label=f"Baseline lower")
    plt.plot(xu_o, yu_o, "--", label=f"Optimized upper")
    plt.plot(xl_o, yl_o, "--", label=f"Optimized lower")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Baseline vs RL-optimized airfoil\n"
        f"Optimized params: m={m:.4f}, p={p:.4f}, t={t:.4f}, best L/D={best_ld:.2f}"
    )
    plt.legend()
    plt.savefig("optimized_airfoil.png", dpi=300)
    print("Saved optimized_airfoil.png")


def main():
    env = AirfoilEnv()
    model = PPO.load("ppo_airfoil_fake", env=env)

    best_params, best_ld = run_episodes(model, env, n_episodes=20)
    print("======== BEST FOUND ========")
    print(f"Best params [m, p, t] = {best_params}")
    print(f"Best L/D = {best_ld:.2f}")

    plot_airfoils(best_params, best_ld)


if __name__ == "__main__":
    main()
