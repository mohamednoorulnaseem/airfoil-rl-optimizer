from airfoil_env import AirfoilEnv

env = AirfoilEnv()

obs, info = env.reset()
print("Initial obs:", obs)

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: reward={reward:.2f}, info={info}")
    if terminated or truncated:
        print("Episode finished.")
        break
