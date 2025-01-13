import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
from interface import Agent

class ProjectAgent(Agent):
    def __init__(self):
        self.model = None

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def save(self, path: str):
        self.model.save(path)

    def load(self):
        model_path = "model.zip"
        if not os.path.exists(model_path):
            model_path = "src/" + model_path
                
        self.model = DQN.load(model_path, device="cpu")
        print(f"Loaded model from {model_path}")

class EvaluationCallback(BaseCallback):
    """
    Custom callback for evaluating the agent every N steps
    """
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            temp_agent = ProjectAgent(self.model)
            
            score_single = evaluate_HIV(agent=temp_agent, nb_episode=1)
            
            score_population = evaluate_HIV_population(agent=temp_agent, nb_episode=5)
            
            if score_population > self.best_mean_reward and score_population > 2e+10 and score_single > 2e+10:
                self.best_mean_reward = score_population
                self.model.save("best_model2e10")
                print("Saved model 2e+10")
                
            self.last_mean_reward = score_population
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}")
                print(f"Single environment score: {score_single:.2e}")
                print(f"Population score: {score_population:.2e}")
                print(f"Best population score so far: {self.best_mean_reward:.2e}")
                print("-" * 50)
                
        return True

def train_dqn():
    env = TimeLimit(
        env=HIVPatient(domain_randomization=True),
        max_episode_steps=200
    )
    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=800,
        batch_size=800,
        tau=1.0,
        gamma=0.99,
        train_freq=1,
        gradient_steps=3,
        target_update_interval=600,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        policy_kwargs={
            "net_arch": [256, 256, 512, 256, 256]
        },
        verbose=1,
        device="auto",
        tensorboard_log="./dqn_hiv_tensorboard/"
    )
    
    # Setup callback
    # Evaluate every 5 episodes (5 * 200 steps = 1000 steps)
    callback = EvaluationCallback(eval_freq=1000)
    
    TOTAL_TIMESTEPS = 40_000
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )
    
if __name__ == "__main__":
    train_dqn()