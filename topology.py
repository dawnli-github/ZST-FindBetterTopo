from algorithm import *
import random
import numpy as np
import torch
import itertools
import copy

# Reproducibility


def seed_all(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


# Generate DME nodes(Sinks)


def genRandomNode(num=8, x_bound=1000, y_bound=1000):
    nodes = []
    for _ in range(num):
        x = random.randint(0, x_bound)
        y = random.randint(0, y_bound)
        nodes.append(DMENode(Point(x, y)))
    return nodes


def knnTopo(nodes):
    while len(nodes) > 1:
        min_pair = None
        min_dist = float('inf')
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                dist = manhattanDistance(nodes[i].ms, nodes[j].ms)
                delta_wl = abs(nodes[i].sub_wl - nodes[j].sub_wl)
                cost = dist if dist - delta_wl > 0 else delta_wl
                # cost = dist - delta_wl
                # cost = dist
                if cost > 0 and (cost < abs(min_dist) or min_dist < 0):
                    min_dist = cost
                    min_pair = (i, j)
                if min_dist == float('inf') or (min_dist < 0
                                                and min_dist < cost):
                    min_dist = cost
                    min_pair = (i, j)

        node_i = nodes[min_pair[0]]
        node_j = nodes[min_pair[1]]
        nodes.remove(node_i)
        nodes.remove(node_j)
        father = dmeMerge(node_i, node_j)
        nodes.append(father)
    return nodes[0]


def orderTopo(nodes):
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) == 2:
        return dmeMerge(nodes[0], nodes[1])
    mid = len(nodes) // 2
    node1 = orderTopo(nodes[:mid])
    node2 = orderTopo(nodes[mid:])
    return dmeMerge(node1, node2)


def enumTopo(nodes, sub_list=[], total_list=[]):
    test_nodes = copy.deepcopy(nodes)
    cb_list = itertools.combinations(test_nodes, 2)
    for combination in cb_list:
        node1 = combination[0]
        node2 = combination[1]
        test_nodes.remove(node1)
        test_nodes.remove(node2)
        loop_nodes = copy.deepcopy(test_nodes)
        test_nodes.append(node1)
        test_nodes.append(node2)
        father = dmeMerge(node1, node2)
        if len(test_nodes) == 2:
            if len(sub_list) == 0 or sub_list[-1] > father.sub_wl:
                sub_list.append(father.sub_wl)
            if len(total_list) == 0 or total_list[-1] > father.total_wl:
                total_list.append(father.total_wl)
        else:
            loop_nodes.append(father)
            enumTopo(loop_nodes, sub_list, total_list)


# Test

seed = 0
seed_all(seed)
node_num = 8
x_bound = node_num * 50
y_bound = node_num * 50

nodes = genRandomNode(node_num, x_bound, y_bound)
knn_test_nodes = copy.deepcopy(nodes)
root = knnTopo(knn_test_nodes)
print("KNN sub: ", root.sub_wl)
print("KNN total: ", root.total_wl)

min_sub_wl = float('inf')
min_total_wl = float('inf')

while True:
    seed += 1
    random.seed(seed)
    test_nodes = copy.deepcopy(nodes)
    while len(test_nodes) > 1:
        random.shuffle(test_nodes)
        node1 = test_nodes.pop()
        random.shuffle(test_nodes)
        node2 = test_nodes.pop()
        father = dmeMerge(node1, node2)
        test_nodes.append(father)
    root = test_nodes[0]
    if (root.sub_wl < min_sub_wl):
        min_sub_wl = root.sub_wl
        print("Order sub: ", min_sub_wl)
    if (root.total_wl < min_total_wl):
        min_total_wl = root.total_wl
        print("Order total: ", min_total_wl)
        if min_total_wl == 831.0:
            plotDME(root, x_bound, y_bound)
            dot = plotTree(root)
            break

# min_sub_wl = []
# min_total_wl = []
# enumDME(nodes,min_sub_wl,min_total_wl)
# print("Enum sub: ", min(min_sub_wl))
# print("Enum total: ", min(min_total_wl))