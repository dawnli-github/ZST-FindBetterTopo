from data import *
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
from graphviz import Digraph

# Polygon Method


def slope(line: LineString):
    p1 = Point(line.coords[0])
    p2 = Point(line.coords[1])
    return (p2.y - p1.y) / (p2.x - p1.x)


def makeRectangle(line: LineString):
    a = Point(line.coords[0])
    b = Point(line.coords[1])
    return Polygon([(a.x, a.y), (a.x, b.y), (b.x, b.y), (b.x, a.y)])


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


# CTS Method


def DME(node: DMENode):
    if node.left is not None and node.right is not None:
        left = node.left
        right = node.right
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
    return father


def topDown(root: DMENode):
    if root.father is not None:
        _, root.ms = nearest_points(root.father.ms, root.ms)
    else:
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


def plotDME(root: DMENode, x_bound, y_bound, file_name="topo"):
    # plot png
    plotMS(root)
    topDown(root)
    plotFlyLine(root)
    plt.xlim(0, x_bound)
    plt.ylim(0, y_bound)
    plt.savefig("./output/picture/" + file_name + ".png", dpi=300)
    # plot tree dot
    plotTree(root)


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
    dot.render(filename=file_name, directory="./output/")
    return dot


#