import argparse
import os
import numpy as np
import pandas as pd

def generate_vrp_data(vrp_size, alpha_inf, alpha_sup, beta_inf, beta_sup, base_inf, base_sup):
    node = np.arange(vrp_size)
    local_x = np.random.uniform(size=(vrp_size)).tolist()
    local_y = np.random.uniform(size=(vrp_size)).tolist()
    alpha = np.random.uniform(alpha_inf, alpha_sup, size=(vrp_size)).tolist()
    beta = np.random.uniform(beta_inf, beta_sup, size=(vrp_size)).tolist()
    base = np.random.uniform(base_inf, base_sup, size=(vrp_size)).tolist()
    alpha[0] = 0
    beta[0] = 0
    base[0] = 0
    velocity = [0.05]
    capacity = [120]
    UAV_num = [int(6)]
    dataframe = pd.concat([pd.DataFrame({'Node': node, 'Local_X': local_x, 'Local_Y': local_y, 'alpha': alpha, 'beta': beta, 'base': base}), pd.DataFrame({'Capacity': capacity, 'Velocity': velocity, 'UAV_num': UAV_num})], axis=1)
    return dataframe

if __name__ == "__main__":
    # data = generate_vrp_data(1, 40, 41, -1, 0)
    # data.to_csv("/Users/wanpengfu/PycharmProjects/Time-varying VRP/test.csv", index=False, sep=',')
    data = generate_vrp_data(30, 0.001, 0.005, 10, 30, 6, 10)
    data.to_csv("/Users/wanpengfu/PycharmProjects/UVA/test.csv", index=False, sep=',')