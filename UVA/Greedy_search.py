import os
import numpy as np
import pandas as pd
import math
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import Function as fc


class Greedy:
    def __init__(self, file_name):
        Node, Local_X, Local_Y, alpha, beta, base, Capacity, Velocity, UAV_num, travel_time = self.get_data(file_name)
        self.Node = Node
        self.Local_X = Local_X
        self.Local_Y = Local_Y
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.Capacity = Capacity
        self.Velocity = Velocity
        self.UAV_num = int(UAV_num)
        self.travel_time = travel_time

    def get_data(self, file_name):
        dir_name = os.path.dirname(os.path.realpath('__file__'))
        print(dir_name)
        file_name = os.path.join(dir_name, file_name)
        df = pd.read_csv(file_name, encoding='latin1')
        customer_size = df.shape[0] - 1
        Node = [df['Node'][i] for i in range(customer_size + 1)]
        Local_X = [df['Local_X'][i] for i in range(customer_size + 1)]
        Local_Y = [df['Local_Y'][i] for i in range(customer_size + 1)]
        alpha = [df['alpha'][i] for i in range(customer_size + 1)]
        beta = [df['beta'][i] for i in range(customer_size + 1)]
        base = [df['base'][i] for i in range(customer_size + 1)]
        Capacity = df['Capacity'][0]
        Velocity = df['Velocity'][0]
        UAV_num = df['UAV_num'][0]
        customer_number = [i for i in range(1, customer_size + 1)]
        node_number = [0] + customer_number
        df2 = df.iloc[:, 1:3]
        dist_matrix = pd.DataFrame(distance_matrix(df2.values, df2.values), index=df2.index, columns=df2.index)
        dis = {(i, j): dist_matrix[i][j] for i in node_number for j in node_number}
        travel_time = {(i, j): dist_matrix[i][j] / Velocity for i in node_number for j in node_number}
        return Node, Local_X, Local_Y, alpha, beta, base, Capacity, Velocity, UAV_num, travel_time

    # single decision
    def greedy_jud(self, unserviced_node, routes, arrive_time, service_time, value):
        # record uva, node, arrive_time, last_service_time, last_value
        temp_record = (0, 0, 0, 0, 0)
        for uav in range(int(self.UAV_num)):
            node1 = routes[uav][-1]
            for unserviced in unserviced_node:
                node2 = unserviced
                # constraint
                left_time = self.Capacity - arrive_time[uav][-1] - self.travel_time[node1, node2] - self.travel_time[node2, 0]
                if left_time < 0:
                    continue
                if node1 == 0:
                    b = fc.value_decay(self.travel_time[node1, node2], self.alpha[node2], self.beta[node2])
                    temp_value = fc.information_value(0, self.base[node2], b)
                    if temp_record[4] < temp_value:
                        temp_record = (uav, node2, self.travel_time[node1, node2], 0, temp_value)
                else:
                    temp_t, temp_value = fc.calculate_t(fc.value_decay, fc.information_value, fc.information_value_cum, self.base[node1], self.base[node2], self.alpha[node1], self.alpha[node2], self.beta[node1], self.beta[node2], arrive_time[uav][-1], left_time, self.travel_time[node1, node2])
                    if left_time - temp_t < 0 or temp_t < 0:
                        continue
                    if temp_record[4] < temp_value[0]:
                        temp_record = (uav, node2, arrive_time[uav][-1] + temp_t + self.travel_time[node1, node2], temp_t, temp_value[0])
        if temp_record[1] == 0:
            return unserviced_node, routes, arrive_time, service_time, value
        unserviced_node.remove(temp_record[1])
        routes[temp_record[0]].append(temp_record[1])
        arrive_time[temp_record[0]].append(temp_record[2])
        # record last inf
        if temp_record[3] != 0:
            service_time[temp_record[0]].append(temp_record[3])
            value[temp_record[0]].append(temp_record[4])
        return unserviced_node, routes, arrive_time, service_time, value

    def greedy_alg(self):
        unserviced = list(range(1, len(self.Node)))
        routes = []
        arrive_time = []
        service_time = []
        value = []
        for uav in range(int(self.UAV_num)):
            routes.append([0])
            arrive_time.append([0])
            service_time.append([0])
            value.append([0])
        count = 0
        counts = len(unserviced)
        while count < counts:
            unserviced, routes, arrive_time, service_time, value = self.greedy_jud(unserviced, routes, arrive_time, service_time, value)
            count += 1
        for uav in range(int(self.UAV_num)):
            node = routes[uav][-1]
            routes[uav].append(0)
            temp_t, temp_value = fc.calculate_final_t(fc.value_decay, fc.information_value, fc.information_value_cum, self.alpha[node], self.beta[node], self.base[node], arrive_time[uav][-1], self.Capacity-arrive_time[uav][-1]-self.travel_time[node, 0])
            arrive_time[uav].append(arrive_time[uav][-1] + temp_t + self.travel_time[node, 0])
            service_time[uav].append(temp_t)
            service_time[uav].append(0)
            value[uav].append(temp_value[0])
            value[uav].append(0)
        return unserviced, routes, arrive_time, service_time, value

    def make_route(self, routes):
        for i in range(self.UAV_num):
            for j in range(1, len(routes[i])):
                plt.plot([self.Local_X[self.Node[routes[i][j - 1]]], self.Local_X[self.Node[routes[i][j]]]],
                         [self.Local_Y[self.Node.index(routes[i][j - 1])], self.Local_Y[self.Node.index(routes[i][j])]],
                         color='r')
                plt.scatter(
                    [self.Local_X[self.Node[routes[i][j - 1]]], self.Local_X[self.Node[routes[i][j]]]],
                    [self.Local_Y[self.Node[routes[i][j - 1]]], self.Local_Y[self.Node[routes[i][j]]]],
                    color='b')

if __name__ == '__main__':
    file_name = 'test.csv'
    greedy = Greedy(file_name)
    unserviced, routes, arrive_time, service_time, value = greedy.greedy_alg()
    print(unserviced)
    print(routes)
    print(arrive_time)
    print(service_time)
    print(value)

