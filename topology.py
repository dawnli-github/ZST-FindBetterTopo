from algorithm import *
import random
import numpy as np
import torch
import itertools
import copy
import math

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


def genTestNode():
    nodes = [
        DMENode(Point(0, 20)),
        DMENode(Point(40, 40)),
        DMENode(Point(90, 90)),
        DMENode(Point(120, 80)),
        DMENode(Point(120, 60)),
        DMENode(Point(40, 120)),
        DMENode(Point(70, 0)),
        DMENode(Point(60, 40)),
        DMENode(Point(130, 30)),
        DMENode(Point(40, 0))
    ]
    return nodes


def genRandomNode(num=16, x_bound=20, y_bound=20):
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
                cost = dist if dist > delta_wl else delta_wl
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


def clusterKnnTopo(nodes, num_limit=16):
    while len(nodes) > num_limit:
        new_nodes = []
        for cluster_nodes in dividNodes(nodes, num_limit):
            root = knnTopo(cluster_nodes)
            new_nodes.append(root)
        nodes = new_nodes
    return knnTopo(nodes)


def biPartitionTopo(nodes):
    if len(nodes) == 1:
        return nodes[0]
    else:
        octagon = findOctagon(nodes)
        oct_nodes = boundNodes(octagon, nodes)
        node_num = len(oct_nodes)
        half_num = math.ceil(len(oct_nodes) / 2)
        for i in range(half_num):
            oct_nodes.append(oct_nodes[i])
        left_set = []
        right_set = []
        partition_cost = float("inf")
        for i in range(node_num):
            ref_set = oct_nodes[i:i + half_num]
            weight_map = {}
            for node in nodes:
                dist_list = []
                for ref_node in ref_set:
                    dist_list.append(manhattanDistance(ref_node.ms, node.ms))
                weight = max(dist_list) + min(dist_list)
                weight_map[node] = weight
            sorted_pair = sorted(weight_map.items(), key=lambda x: x[1])
            sorted_node = [node for node, _ in sorted_pair]
            s_l = sorted_node[:len(sorted_node) // 2]
            s_r = sorted_node[len(sorted_node) // 2:]
            cur_cost = boundDiameter(s_l) + boundDiameter(s_r)
            if partition_cost > cur_cost:
                partition_cost = cur_cost
                left_set = s_l
                right_set = s_r
        left_node = biPartitionTopo(left_set)
        right_node = biPartitionTopo(right_set)
        return dmeMerge(left_node, right_node)


def topClusterTopo(nodes):
    if len(nodes) > 2:
        cluster_nodes = biPartitionNodes(nodes)
        left_node = topClusterTopo(cluster_nodes[0])
        right_node = topClusterTopo(cluster_nodes[1])
        return topoMerge(left_node, right_node)
    if len(nodes) == 2:
        return topoMerge(nodes[0], nodes[1])
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


def testDME(nodes, x_bound, y_bound, msg, func, need_DME=False):
    timer = Timer()
    timer.start()
    test_nodes = copy.deepcopy(nodes)
    root = func(test_nodes)
    if (need_DME):
        DME(root)
    root.log(msg)
    # plotDME(root, x_bound, y_bound, msg)
    timer.printTime(msg)

# Test
seed = 0
seed_all(seed)
node_num = 25000
x_bound = node_num * 150
y_bound = node_num * 150

nodes = genRandomNode(node_num, x_bound, y_bound)
# nodes = genTestNode()

# Cluster-KNN Topo
testDME(nodes, x_bound, y_bound,"Cluster Knn",clusterKnnTopo)

# Bi-Partition Topo
testDME(nodes, x_bound, y_bound,"Bi-Partition",biPartitionTopo)

# Top-Cluster Topo
testDME(nodes, x_bound, y_bound,"Bi-Cluster",topClusterTopo,True)
