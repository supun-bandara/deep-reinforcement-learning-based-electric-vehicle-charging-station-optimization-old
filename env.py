import sys
import os
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import pathlib
from gym.utils import seeding
from GetData import get_data
from SimulateActions import simulate_actions
from SimulateStation import simulate_station
from WriteToCSV import write_to_csv

class ChargingEnv(gym.Env):
    def __init__(self):
        self.number_of_chargers = 20
        self.maximum_demand = 10 # kWh
        self.done = False
        low = np.array(np.zeros(2*self.number_of_chargers+2), dtype=np.float32)
        high = np.array(np.ones(2*self.number_of_chargers+2), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.number_of_chargers,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.seed

    def step(self, actions):
        #print('env - step - actions', actions)
        #print()
        self.timestep = self.timestep + 1
        # print(self.timestep)
        [reward, cost_2, cost_3, grid, self.res_charging_demand, self.res_parking_time] = simulate_actions(self, actions)
        conditions = self.get_obs()
        results = {'timestep': self.timestep, 'price': self.price, 'maximum_demand': self.maximum_demand, 'new_evs': self.new_evs, 'actions': actions, 'res_charging_demand': self.res_charging_demand, 'res_parking_time': self.res_parking_time, 'leave': self.leave, 'reward': -reward, 'cost_2': cost_2, 'cont_3': cost_3, 'grid': grid}
        write_to_csv(results)
        '''if self.timestep == 200000: # change this according to the number of timesteps
            self.done = True    
            self.timestep = 0'''
        self.info = {}
        #print('env - step - conditions', conditions, 'reward', -reward, 'done', self.done, 'info', self.info)
        #print()
        return conditions, -reward, self.done, False, self.info

    def reset(self, seed=None, reset_flag=1):
        #print('env - reset')
        #print()
        self.timestep = 1
        self.chargers = np.zeros(self.number_of_chargers)
        self.res_parking_time = np.zeros(self.number_of_chargers)
        self.res_charging_demand = np.zeros(self.number_of_chargers)
        self.done = False
        self.df_station = pd.read_csv('EVCS.csv')
        self.df_price = pd.read_csv('Price.csv')
        self.info = {}
        #print('env - reset - obs', self.get_obs(), 'info', self.info)
        #print()
        return self.get_obs(), self.info

    def get_obs(self):
        #print('env - get_obs')
        #print()
        [self.price, self.new_evs] = get_data(self)
        [self.leave, self.chargers, self.res_parking_time, self.res_charging_demand] = simulate_station(self)
        states = np.concatenate((np.array([self.price, self.maximum_demand]), np.array(self.res_charging_demand), np.array(self.res_parking_time)),axis=None)
        observations = np.concatenate((states),axis=None)
        observations = observations.astype(np.float32)
        #print('env - get_obs - observations', observations)
        #print()
        return observations

    def seed(self, seed=None):
        #print('env - seed')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return 0
'''
env = ChargingEnv()
episodes = 1
for episode in range(episodes):
	done = False
	obs = env.reset()
	while not done:
		random_action = env.action_space.sample()
		obs, reward, done, finish, info = env.step(random_action)
'''