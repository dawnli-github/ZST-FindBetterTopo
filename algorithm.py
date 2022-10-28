from data import *
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
from graphviz import Digraph
from sklearn.cluster import KMeans
import numpy as np
from sklearn import svm

# Polygon Method


def slope(line: LineString):
    p1 = Point(line.coords[0])
    p2 = Point(line.coords[1])
    return (p2.y - p1.y) / (p2.x - p1.x)


def makeRectangle(line: LineString):
    a = Point(line.coords[0])
    b = Point(line.coords[1])
    return Polygon([(a.x, a.y), (a.x, b.y), (b.x, b.y), (b.x, a.y)])


def findOctagon(nodes):
    x_p = -1 * float('inf')
    x_m = float('inf')
    y_p = -1 * float('inf')
    y_m = float('inf')
    ymx_p = -1 * float('inf')
    ymx_m = float('inf')
    ypx_p = -1 * float('inf')
    ypx_m = float('inf')
    for node in nodes:
        x, y = node.ms.coords[0]
        x_p = max(x, x_p)
        x_m = min(x, x_m)
        y_p = max(y, y_p)
        y_m = min(y, y_m)
        ymx_p = max(y - x, ymx_p)
        ymx_m = min(y - x, ymx_m)
        ypx_p = max(y + x, ypx_p)
        ypx_m = min(y + x, ypx_m)
    return Polygon([(y_p - ymx_p, y_p), (ypx_p - y_p, y_p), (x_p, ypx_p - x_p),
                    (x_p, x_p + ymx_m), (y_m - ymx_m, y_m), (ypx_m - y_m, y_m),
                    (x_m, ypx_m - x_m), (x_m, x_m + ymx_p)]).convex_hull


def boundNodes(octagon, nodes):
    if isinstance(octagon, Point):
        return nodes
    bound_nodes = []
    coords = []
    for node in nodes:
        if node.ms.intersects(octagon.boundary):
            bound_nodes.append(node)
            coords.append(node.ms.xy)
    atan_map = {}
    cen_x, cen_y = np.mean(coords, axis=0)
    for i in range(len(bound_nodes)):
        node = bound_nodes[i]
        o_x = coords[i][0] - cen_x
        o_y = coords[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        atan_map[node] = atan2
    sorted_pair = sorted(atan_map.items(), key=lambda x: x[1])
    sorted_bound_nodes = [node for node, _ in sorted_pair]
    return sorted_bound_nodes


def boundDiameter(nodes):
    octagon = findOctagon(nodes)
    bound_nodes = boundNodes(octagon, nodes)
    max_dist = 0
    for i in range(len(bound_nodes)):
        for j in range(i, len(bound_nodes)):
            max_dist = max(
                manhattanDistance(bound_nodes[i].ms, bound_nodes[j].ms),
                max_dist)
    return max_dist


def diameter(nodes):
    max_dist = 0
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            max_dist = max(manhattanDistance(nodes[i].ms, nodes[j].ms),
                           max_dist)
    return max_dist


def extend(polygon, ridius):
    coords = []
    for i in range(len(polygon.coords)):
        x = polygon.coords[i][0]
        y = polygon.coords[i][1]
        coords.append((x + ridius, y))
        coords.append((x - ridius, y))
        coords.append((x, y + ridius))
        coords.append((x, y - ridius))
    return Polygon(coords).convex_hull


def PMD(a: Point, b: Point):
    return abs(a.x - b.x) + abs(a.y - b.y)


def PLMD(p: Point, line: LineString):
    if line.coords[0] == line.coords[1]:
        return PMD(p, Point(line.coords[0]))
    rect = makeRectangle(line)
    if rect.intersects(p):
        k = slope(line)
        t_p = Point(line.coords[0])
        v_dist = abs(p.y - (t_p.y + k * (p.x - t_p.x)))
        h_dist = abs(p.x - (t_p.x + (p.y - t_p.y) / k))
        return min(v_dist, h_dist)
    return min(PMD(p, Point(line.coords[0])), PMD(p, Point(line.coords[1])))


def LMD(a: LineString, b: LineString):
    if (a.intersects(b)):
        return 0
    return min(PLMD(Point(b.coords[0]), a), PLMD(Point(b.coords[1]), a),
               PLMD(Point(a.coords[0]), b), PLMD(Point(a.coords[1]), b))


def manhattanDistance(a, b):
    if isinstance(a, Point) and isinstance(b, Point):
        return PMD(a, b)
    if isinstance(a, LineString) and isinstance(b, Point):
        return PLMD(b, a)
    if isinstance(a, Point) and isinstance(b, LineString):
        return PLMD(a, b)
    if isinstance(a, LineString) and isinstance(b, LineString):
        return LMD(a, b)
    raise TypeError('Must be Point or LineString')


def midPoint(node):
    if isinstance(node.ms, Point):
        return np.array(node.ms.coords[0])
    if isinstance(node.ms, LineString):
        return (np.array(node.ms.coords[0]) + np.array(node.ms.coords[1])) / 2
    if isinstance(node.ms, Polygon):
        ex_coords = node.ms.exterior._set_coords
        return np.sum(np.array(ex_coords[:-1]), axis=0) / len(ex_coords)


# Cluster


def dividNodes(nodes, num_limit=16):
    cluster_num = len(nodes) // num_limit
    locations = []
    for node in nodes:
        locations.append(midPoint(node))
    kmeans = KMeans(n_clusters=cluster_num).fit(locations)
    cluster_nodes = [[] for _ in range(cluster_num)]
    for i in range(len(nodes)):
        label = kmeans.labels_[i]
        cluster_nodes[label].append(nodes[i])
    return cluster_nodes


def biPartitionNodes(nodes):
    cluster_num = 2
    locations = []
    for node in nodes:
        locations.append(midPoint(node))
    kmeans = KMeans(n_clusters=cluster_num).fit(locations)
    cluster_nodes = [[] for _ in range(cluster_num)]
    for i in range(len(nodes)):
        label = kmeans.labels_[i]
        cluster_nodes[label].append(nodes[i])
    return cluster_nodes


# CTS Method


def DME(node: DMENode):
    if node.left is not None and node.right is not None:
        left = node.left
        right = node.right
        DME(left)
        DME(right)
        dist = manhattanDistance(left.ms, right.ms)
        delta_wl = left.sub_wl - right.sub_wl
        if dist >= abs(delta_wl):
            left_radius = (dist - delta_wl) / 2
            right_radius = (dist + delta_wl) / 2
            left_region = extend(left.ms, left_radius)
            right_region = extend(right.ms, right_radius)
            ms = left_region & right_region
            sub_wl = (left.sub_wl + right.sub_wl + dist) / 2
            merge_cost = dist
        else:
            if left.sub_wl < right.sub_wl:
                ms = extend(left.ms, -delta_wl) & right.ms
            else:
                ms = extend(right.ms, delta_wl) & left.ms
            sub_wl = max(left.sub_wl, right.sub_wl)
            merge_cost = abs(delta_wl)
        node.ms = ms
        node.sub_wl = sub_wl
        node.total_wl = left.total_wl + right.total_wl + merge_cost


def dmeMerge(left: DMENode, right: DMENode):
    dist = manhattanDistance(left.ms, right.ms)
    delta_wl = left.sub_wl - right.sub_wl
    if dist >= abs(delta_wl):
        left_radius = (dist - delta_wl) / 2
        right_radius = (dist + delta_wl) / 2
        left_region = extend(left.ms, left_radius)
        right_region = extend(right.ms, right_radius)
        ms = left_region & right_region
        sub_wl = (left.sub_wl + right.sub_wl + dist) / 2
        merge_cost = dist
    else:
        if left.sub_wl < right.sub_wl:
            ms = extend(left.ms, -delta_wl) & right.ms
        else:
            ms = extend(right.ms, delta_wl) & left.ms
        sub_wl = max(left.sub_wl, right.sub_wl)
        merge_cost = abs(delta_wl)
    father = DMENode(ms, left, right, sub_wl,
                     left.total_wl + right.total_wl + merge_cost)
    left.father = father
    right.father = father
    return father


def topoMerge(left: DMENode, right: DMENode):
    father = DMENode(left=left, right=right)
    left.father = father
    right.father = father
    return father


def topDown(root: DMENode):
    if root.father is not None:
        _, root.ms = nearest_points(root.father.ms, root.ms)
    else:
        if isinstance(root.ms, Point) == False:
            root.ms = Point(root.ms.xy[0])
    if root.left is not None:
        topDown(root.left)
    if root.right is not None:
        topDown(root.right)


# Plotter


def plotMS(root: DMENode):
    if root.left is not None:
        plotMS(root.left)
    if root.right is not None:
        plotMS(root.right)
    if root.left is None and root.right is None:
        plt.plot(*root.ms.xy, 'b.')
    elif root.father is None:
        if isinstance(root.ms, Point):
            plt.plot(*root.ms.xy, 'r*', markersize=5)
        else:
            plt.plot(*root.ms.xy[0], 'r*', markersize=5)
    else:
        plt.plot(*root.ms.xy, color='red')


def plotFlyLine(root: DMENode):
    if root.left is not None:
        plotFlyLine(root.left)
    if root.right is not None:
        plotFlyLine(root.right)
    if root.father is not None:
        plt.plot([*root.ms.xy[0], *root.father.ms.xy[0]],
                 [*root.ms.xy[1], *root.father.ms.xy[1]],
                 color='green')


def nameInit(root: DMENode, i=0):
    root.id = i
    root.name = root.ms.wkt
    if root.left is not None:
        nameInit(root.left, 2 * i + 1)
    if root.right is not None:
        nameInit(root.right, 2 * i + 2)


def plotTree(root: DMENode, file_name="plotTree"):
    nameInit(root)
    dot = Digraph(comment='The DME Tree')
    node_list = [root]
    while (len(node_list) > 0):
        node = node_list.pop()
        dot.node(str(node.id), node.name)
        if node.father is not None:
            dot.edge(str(node.father.id), str(node.id))
        if node.left is not None:
            node_list.append(node.left)
        if node.right is not None:
            node_list.append(node.right)
    dot.render(filename=file_name, directory="./output/tree/")
    return dot


def plotDME(root: DMENode, x_bound, y_bound, file_name="topo"):
    # plot png
    plotMS(root)
    topDown(root)
    plotFlyLine(root)
    plt.xlim(-10, x_bound + 10)
    plt.ylim(-10, y_bound + 10)
    plt.savefig("./output/picture/" + file_name + ".png", dpi=300)
    plt.close()
    # plot tree dot
    plotTree(root, file_name)


#