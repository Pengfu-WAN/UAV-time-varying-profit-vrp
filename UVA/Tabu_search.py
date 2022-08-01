from Greedy_search import Greedy
import Function as fc
import numpy as np
import copy
from tkinter import _flatten
from tqdm import trange
import time
import matplotlib.pyplot as plt

class Tabu:
    def __init__(self, file_name):
        gy = Greedy(file_name)
        Node, Local_X, Local_Y, alpha, beta, base, Capacity, Velocity, UAV_num, travel_time = gy.get_data(file_name)
        self.gy = gy
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

    def get_initial_solution(self):
        unserviced, routes, arrive_time, service_time, value = self.gy.greedy_alg()
        return unserviced, routes, arrive_time, service_time, value

    def get_total_value(self, routes):
        # record new route
        unserviced = []
        route = [0]
        arrive_time = [0]
        service_time = [0]
        value = [0]
        route.append(routes[1])
        arrive_time.append(self.travel_time[routes[0], routes[1]])
        for i in range(1, len(routes) - 2):
            node1 = routes[i]
            node2 = routes[i + 1]
            left_time = self.Capacity - arrive_time[-1] - self.travel_time[node1, node2] - self.travel_time[
                node2, 0]
            if left_time < 0:
                continue
            temp_t, temp_value = fc.calculate_t(fc.value_decay, fc.information_value, fc.information_value_cum,
                                                self.base[node1], self.base[node2], self.alpha[node1],
                                                self.alpha[node2], self.beta[node1], self.beta[node2],
                                                arrive_time[-1], left_time, self.travel_time[node1, node2])
            if left_time - temp_t < 0 or temp_t < 0:
                continue
            route.append(node2)
            arrive_time.append(arrive_time[-1] + temp_t + self.travel_time[node1, node2])
            service_time.append(temp_t)
            value.append(temp_value[0])
        node = route[-1]
        route.append(0)
        temp_t, temp_value = fc.calculate_final_t(fc.value_decay, fc.information_value, fc.information_value_cum,
                                                  self.alpha[node], self.beta[node], self.base[node],
                                                  arrive_time[-1],
                                                  self.Capacity - arrive_time[-1] - self.travel_time[node, 0])
        arrive_time.append(arrive_time[-1] + temp_t + self.travel_time[node, 0])
        service_time.append(temp_t)
        service_time.append(0)
        value.append(temp_value[0])
        value.append(0)

        for node in routes:
            if node not in route:
                unserviced.append(node)

        return unserviced, route, arrive_time, service_time, value

    def neighbor_add(self, unserviced, routes):
        if unserviced is None:
            return None
        neighbors = []
        neighbors_value = []
        original_value = np.zeros(self.UAV_num)
        for uav in range(self.UAV_num):
            original_value[uav] = sum(self.get_total_value(routes[uav])[4])
        original_value_sum = sum(original_value)
        for node in unserviced:
            for uav in range(self.UAV_num):
                for place in range(1, len(routes[uav])):
                    temp_neighbor = copy.deepcopy(routes)
                    temp_neighbor[uav].insert(place, node)
                    unserviced, route, arrive_time, service_time, value = self.get_total_value(temp_neighbor[uav])
                    temp_neighbor_value = original_value_sum - original_value[uav] + sum(value)
                    temp_neighbor[uav] = route
                    neighbors.append(temp_neighbor)
                    neighbors_value.append(temp_neighbor_value)
        return neighbors, neighbors_value

    def neighbor_reduce(self, routes):
        neighbors = []
        neighbors_value = []
        original_value = np.zeros(self.UAV_num)
        for uav in range(self.UAV_num):
            original_value[uav] = sum(self.get_total_value(routes[uav])[4])
        original_value_sum = sum(original_value)
        for uav in range(self.UAV_num):
            for node in routes[uav][1:-1]:
                temp_neighbor = copy.deepcopy(routes)
                temp_neighbor[uav].remove(node)
                unserviced, route, arrive_time, service_time, value = self.get_total_value(temp_neighbor[uav])
                temp_neighbor_value = original_value_sum - original_value[uav] + sum(value)
                temp_neighbor[uav] = route
                neighbors.append(temp_neighbor)
                neighbors_value.append(temp_neighbor_value)
        return neighbors, neighbors_value

    def get_neighbour(self, routes, unserviced):
        add_neighbors, add_neighbors_value = self.neighbor_add(unserviced, routes)
        reduce_neighbors, reduce_neighbors_value = self.neighbor_reduce(routes)
        new_neighbour = add_neighbors + reduce_neighbors
        new_neighbour_value = add_neighbors_value + reduce_neighbors_value
        return new_neighbour, new_neighbour_value

    def tabu_search(self, tabu_len, iteration_count, routes):
        tabu_table = []
        tabu_table.append(routes)
        routes_best = routes
        value_best = 0
        for uav in range(self.UAV_num):
            unserviced, route, arrive_time, service_time, value = self.get_total_value(routes[uav])
            value_best += sum(value)
        routes_expect = routes_best
        value_expect = value_best
        expect = np.zeros(iteration_count)
        best = np.zeros(iteration_count)
        for iter in trange(iteration_count):
            time.sleep(1)
            unserviced = []
            for node in self.Node:
                if node not in list(_flatten(routes_best)):
                    unserviced.append(node)
            routes_new, value_new = self.get_neighbour(routes_best, unserviced)
            value_best = max(value_new)
            routes_best = routes_new[value_new.index(value_best)]
            if value_best > value_expect:
                value_expect = value_best
                routes_expect = routes_best
                if routes_best in tabu_table:
                    tabu_table.remove(routes_best)
                tabu_table.append(routes_best)
            else:
                while routes_best in tabu_table:
                    value_new.remove(value_best)
                    routes_new.remove(routes_best)
                    value_best = max(value_new)
                    routes_best = routes_new[value_new.index(value_best)]
                tabu_table.append(routes_best)
            if len(tabu_table) >= tabu_len:
                del tabu_table[0]
            expect[iter] = value_expect
            best[iter] = value_best
        # record routes_expect inf
        arrive_time_expect = []
        service_time_expect = []
        for uva in range(self.UAV_num):
            unserviced, route, arrive_time, service_time, value = self.get_total_value(routes_expect[uva])
            arrive_time_expect.append(arrive_time)
            service_time_expect.append(service_time)
        return value_expect, routes_expect, expect, best, arrive_time_expect, service_time_expect

if __name__ == '__main__':
    start = time.time()
    file_name = 'test.csv'
    tabu = Tabu(file_name)
    tabu_len = 100
    iteration_count = 300
    unserviced, routes, arrive_time, service_time, value = tabu.get_initial_solution()
    value_expect, routes_expect, expect, best, arrive_time_expect, service_time_expect = tabu.tabu_search(tabu_len, iteration_count, routes)
    end = time.time()
    print("Total Value:", value_expect)
    print("Final Route:", routes_expect)
    print("Arrive Time:", arrive_time_expect)
    print("Service Time:", service_time_expect)
    print("Run Time:", end-start)
    x = np.arange(iteration_count)
    plt.figure(figsize=(8, 8), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(211)
    tabu.gy.make_route(routes_expect)
    ax2 = plt.subplot(223)
    plt.plot(x, expect)
    ax3 = plt.subplot(224)
    plt.plot(x, best)
    plt.savefig('Tabu_test.png')
    result = []
    result.append(value_expect)
    result.append(routes_expect)
    result.append(arrive_time_expect)
    result.append(service_time_expect)
    result.append(end-start)
    data = np.array(result, dtype=object)
    n = np.savetxt("tabu_search_result.txt", data, fmt="%s")
    plt.show()

    """
    file_name = 'test.csv'
    tabu = Tabu(file_name)
    unserviced, route, arrive_time, service_time, value = tabu.get_total_value([0, 29, 0])
    print(route)
    print(arrive_time)
    print(service_time)
    print(value)
    unserviced, route, arrive_time, service_time, value = tabu.get_total_value([0, 29, 1, 0])
    print(route)
    print(arrive_time)
    print(service_time)
    print(value)
    """


