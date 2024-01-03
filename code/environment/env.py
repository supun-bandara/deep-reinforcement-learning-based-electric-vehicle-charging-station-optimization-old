import torch
import operator
import numpy as np
import pandas as pd
from scipy.io import loadmat

class ENV:
    def __init__(self):
        self.df = pd.read_excel('data/CAISO Average Price 2017.xlsx')
        self.m = pd.read_excel('data/Electric_Vehicle_Charging_Station_Data.xlsx')
        self.out1 = np.random.randint(0, 4, size=4000)
        self.ISO_eprice = self.df['price'].values
        self.beta1 = -1
        self.beta2 = 6
        self.deadline = 6  # Unit: 5
        self.theta1 = 0.1
        self.theta2 = 0.9
        self.ISO_eprice = self.ISO_eprice.astype('float64')
        self.eprice_mean = np.mean(self.ISO_eprice)

    def env(self, action, residual_demand, iternum):
        print(f"env - action: {action}, residual_demand: {residual_demand}, iternum: {iternum}")
        if action[1] > residual_demand.shape[0]:
            print(f"env - residual_demand.shape: {residual_demand.shape}")
            action[1] = residual_demand.shape[0]
            print(f"env - action[1]: {action[1]}")

        # Charging Station Start to Charge
        if residual_demand.shape[0] > 0.5:
            least = residual_demand[:, 1] - residual_demand[:, 0]
            print(f"env - least: {least}")
            order = [operator.itemgetter(0)(t) - 1 for t in sorted(enumerate(least, 1), key=operator.itemgetter(1),
                                                                   reverse=True)]
            print(f"env - order: {order}")
            residual_demand[order[:action[1]], 0] = residual_demand[order[:action[1]], 0] - 1
            print(f"env - residual_demand[order[:action[1]],0]: {residual_demand[order[:action[1]],0]}")
            residual_demand[:, 1] = residual_demand[:, 1] - 1
            print(f"env - residual_demand[:,1]: {residual_demand[:,1]}")

        # EV Admission
        reward = 0
        for i in range(self.out1[iternum]):
            dem = self.beta1 * action[0] + self.beta2
            if dem < 0:
                dem = 0
            reward += dem * action[0]
            print(f"reward: {reward}, dem: {dem}, action[0]: {action[0]}")
            residual_demand = self.demand_update(residual_demand, np.array([dem, self.deadline]).reshape((1, 2)))
            print(f"env - for loop - dem: {dem}, reward: {reward}, residual_demand: {residual_demand}")

        if residual_demand.shape[0] < 0.5:
            return reward, residual_demand, torch.tensor([0, action[1], 0, 0, self.ISO_eprice[iternum + 1], self.out1[iternum + 1]])

        # Departure
        residual_demand_ = []
        for i in range(residual_demand.shape[0]):
            if residual_demand[i, 1] > 0.5 and residual_demand[i, 0] > 0.5:
                residual_demand_.append(residual_demand[i, :])
        residual_demand = np.array(residual_demand_)

        # Caculate Reward and Features
        f1 = reward
        f2 = action[1]
        try:
            reward_output = reward_output - action[1] * self.ISO_eprice[iternum]
        except:
            reward_output = reward - action[1] * self.ISO_eprice[iternum]
        f3 = 0
        f4 = 0
        for i in range(residual_demand.shape[0]):
            f3 = f3 - residual_demand[i, 0] * (
                        np.max(residual_demand[:, 1]) - residual_demand[i, 1]) * self.theta1
            f4 = f4 - residual_demand[i, 0] * np.power(self.theta2, residual_demand[i, 1])
        return reward_output, residual_demand, torch.tensor([f1, f2, max(f3, -20), max(f4, -20), 5 * self.ISO_eprice[iternum + 1] / self.eprice_mean,self.out1[iternum + 1]])

    def demand_update(self, current, new):
        if current.shape[0] < 0.5:
            output = new
        else:
            output = np.concatenate((current, new), axis=0)
        return output
