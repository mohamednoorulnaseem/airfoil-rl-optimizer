from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from airfoil_env import AirfoilEnv


def main():
    env = AirfoilEnv()

    # Optional: check env for common issues
    check_env(env, warn=True)

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=None,
    )

    # Train the agent
    model.learn(total_timesteps=50000)

    model.save("ppo_airfoil_fake")

    # Test the trained agent
    obs, info = env.reset()
    total_reward = 0.0
    for step in range(30):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"Step {step}: reward={reward:.2f}, "
            f"m={env.params[0]:.4f}, p={env.params[1]:.4f}, t={env.params[2]:.4f}, "
            f"L/D={info['L/D']:.2f}"
        )
        if terminated or truncated:
            break

    print("Total test reward:", total_reward)


if __name__ == "__main__":
    main()
