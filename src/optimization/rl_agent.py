"""
RL Agent Module

PPO agent wrapper with training utilities and model management.

Author: Mohamed Noorul Naseem
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        return True


class AirfoilRLAgent:
    """
    PPO agent for airfoil optimization.
    
    Features:
    - Easy training interface
    - Model checkpointing
    - Performance tracking
    - Hyperparameter management
    """
    
    def __init__(
        self,
        env,
        model_path: str = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1
    ):
        self.env = env
        self.model_path = model_path
        self.verbose = verbose
        
        self.hyperparams = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef
        }
        
        if model_path and Path(model_path).exists():
            self.model = PPO.load(model_path, env=env)
            print(f"Loaded model from {model_path}")
        else:
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                **self.hyperparams
            )
        
        self.callback = TrainingCallback()
        self.training_history = []
    
    def train(self, total_timesteps: int = 100000, save_path: str = None) -> Dict:
        """Train the agent."""
        print(f"Training for {total_timesteps} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True
        )
        
        if save_path:
            self.save(save_path)
        
        return {
            'total_timesteps': total_timesteps,
            'episodes': len(self.callback.episode_rewards),
            'mean_reward': np.mean(self.callback.episode_rewards[-100:]),
            'best_reward': max(self.callback.episode_rewards) if self.callback.episode_rewards else 0
        }
    
    def predict(self, obs, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action for given observation."""
        return self.model.predict(obs, deterministic=deterministic)
    
    def optimize(self, max_steps: int = None) -> Dict:
        """Run optimization episode and return best result."""
        if max_steps is None:
            max_steps = self.env.max_steps
        
        obs, info = self.env.reset()
        best_ld = -np.inf
        best_params = None
        trajectory = []
        
        for step in range(max_steps):
            action, _ = self.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Extract L/D from objectives in info
            objectives = info.get('objectives', {})
            ld = objectives.get('ld_mean', 0)
            
            if ld > best_ld:
                best_ld = ld
                best_params = self.env.params.copy()
            
            trajectory.append({
                'step': step,
                'params': list(self.env.params),
                'reward': reward,
                'ld': ld
            })
            
            if terminated or truncated:
                break
        
        return {
            'best_ld': best_ld,
            'best_params': best_params,
            'trajectory': trajectory,
            'final_info': info
        }
    
    def save(self, path: str):
        """Save model to file."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def get_training_history(self) -> Dict:
        """Get training statistics."""
        return {
            'episode_rewards': self.callback.episode_rewards,
            'episode_lengths': self.callback.episode_lengths,
            'hyperparams': self.hyperparams
        }


def create_agent(env, **kwargs) -> AirfoilRLAgent:
    """Factory function to create agent."""
    return AirfoilRLAgent(env, **kwargs)


if __name__ == "__main__":
    print("Testing RL Agent Module...")
    
    from airfoil_env import AirfoilEnv
    env = AirfoilEnv()
    agent = AirfoilRLAgent(env, model_path="models/ppo_airfoil_fake.zip")
    
    result = agent.optimize()
    print(f"\nOptimization Result:")
    print(f"  Best L/D: {result['best_ld']:.2f}")
    print(f"  Best params: {result['best_params']}")
