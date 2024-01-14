import numpy as np
import time

def simulate_actions(self,actions):
    #print('simulate_actions - simulate_clever_control - actions', actions)
    #print()
    chargers = self.chargers
    res_parking_time = self.res_parking_time
    res_charging_demand = self.res_charging_demand
    leave=self.leave

    P_charging=np.zeros(self.number_of_chargers)

    for car in range(self.number_of_chargers): 
        # in case action=[-100,100] P_charging[car] = actions[car]/100*max_charging_energy otherwise if action=[-1,1] P_charging[car] = 100*actions[car]/100*max_charging_energy

        if chargers[car] == 1:
            max_charging_energy = min([2, res_charging_demand[car]]) ########## defind maximum energy that the car can get
            P_charging[car] = actions[car]*max_charging_energy ######### check this again    # 
            res_charging_demand[car] = res_charging_demand[car] - P_charging[car] ############### change this back once other problems are done : P_charging[car]
            res_parking_time[car] = res_parking_time[car] - 1
        else:
            P_charging[car] = 0

    # Calculation of energy coming from Grid
    total_charging = sum(P_charging)

    # First cost index
    #energy_from_grid = max([total_charging, 0])
    energy_from_grid = total_charging
    cost_1 = energy_from_grid*self.price

    # Maximum demand of the station
    if total_charging > self.maximum_demand:
        cost_2 = (total_charging - self.maximum_demand)*1000
    elif total_charging > 0:
        cost_2 = (self.maximum_demand - total_charging)
    else:
        cost_2 = 0

    #Penalty of not fully charging the cars that leave
    cost_EV =[]
    for ii in range(len(leave)):
        cost_EV.append((ii*2)*100)
    cost_3 = sum(cost_EV)

    cost = cost_2 + cost_3

    #print('simulate_actions - simulate_clever_control - cost', cost, 'energy_from_grid', energy_from_grid, 'cost_3', cost_3, 'res_charging_demand', res_charging_demand)
    #print()
    return cost, cost_2, cost_3, energy_from_grid, res_charging_demand, res_parking_time