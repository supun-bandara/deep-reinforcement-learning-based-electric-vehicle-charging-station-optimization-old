import pandas as pd

def get_data(self):
    #print('Data')
    #print()
    ID = self.timestep
    price = self.df_price[self.df_price['ID'] == ID]['price'].values[0]

    new_evs = []

    if ID in self.df_station['Start_Time_Index'].values:
        rows = self.df_station.loc[self.df_station['Start_Time_Index'] == ID]
        for index, row in rows.iterrows():
            duration = row['Duration_Count']
            SOC = row['SOC_Level']
            Capacity = row['Battery_Capacity_kWh']
            energy_demand = (100-SOC)*Capacity/100
            new_evs.append([duration, energy_demand])
    
    #print('Data - price', price, 'new_evs', new_evs)
    #print()
    return price, new_evs