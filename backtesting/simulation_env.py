import gym
import numpy as np
import pandas as pd
from gym import spaces


class TradingEnv(gym.Env):
    """
    A trading simulation environment for reinforcement learning.
    Compatible with OpenAI Gym.
    """

    def __init__(self, data, initial_cash=1_000_000, max_steps=None):
        super(TradingEnv, self).__init__()

        self.data = data 
        self.initial_cash = initial_cash
        self.max_steps = max_steps or len(data) - 1

       
        self.action_space = spaces.Discrete(3)

        
        num_features = data.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features + 2,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.position = 0
        self.current_step = 0
        self.done = False
        self.trade_log = []
        self.entry_price = 0
        return self._get_observation()

    def _get_observation(self):
        obs = self.data.iloc[self.current_step].values
        return np.concatenate((obs, [self.cash], [self.position]), axis=None)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, {}

        current_price = self.data.iloc[self.current_step]['close']
        reward = 0

        if action == 1:  # buy
            if self.position == 0:
                self.position = self.cash / current_price
                self.cash = 0
                self.entry_price = current_price

        elif action == 2:  # sell
            if self.position > 0:
                proceeds = self.position * current_price
                reward = proceeds - (self.position * self.entry_price)
                self.cash += proceeds
                self.position = 0
                self.entry_price = 0

        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.done = True

        next_obs = self._get_observation()
        return next_obs, reward, self.done, {}

    def render(self, mode='human'):
        current_price = self.data.iloc[self.current_step]['close']
        net_worth = self.cash + self.position * current_price
        print(f"Step: {self.current_step}, Price: {current_price:.2f}, Cash: {self.cash:.2f}, Position: {self.position:.4f}, Net Worth: {net_worth:.2f}")

    def seed(self, seed=None):
        np.random.seed(seed)


if __name__ == '__main__':
    df = pd.read_csv('data/processed/SPY.csv', parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    env = TradingEnv(df)
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
