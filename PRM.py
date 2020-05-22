"""
ANP hw 4 hands on part.
Submitters:
 Ori Sztyglic
 Asaf Danieli
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
from copy import copy
from typing import Tuple
from typing import NamedTuple


class Obs:
    def __init__(self, x=0.0, y=0.0, nx=0.0, ny=0.0):
        self.x = x
        self.y = y
        self.nx = nx
        self.ny = ny

    def IsObs(self, query_x, query_y):
        if self.x <= query_x <= self.x + self.nx and \
                self.y <= query_y <= self.y + self.ny:
            return True
        else:
            return False


class ObsList:
    def __init__(self, obsList=None):
        if obsList is None:
            obsList = []
        self.obsList = copy(obsList)

    def is_in_obs_set(self, query_x, query_y):
        for obs in self.obsList:
            if obs.IsObs(query_x, query_y):
                return True
        return False

    def push_obs(self, obs: Obs):
        assert obs
        self.obsList.append(obs)

def SampleNode(Cobs: ObsList, N: int) -> Tuple[float, float]:
    loc = np.random.uniform(low=0.0, high=float(N), size=(2, 1))
    while Cobs.is_in_obs_set(loc[0], loc[1]):
        loc = np.random.uniform(low=0.0, high=float(N), size=(2, 1))
    return loc


# A Python3 program to find if 2 given line segments intersect or not

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Given three colinear points p, q, r, the function checks if


# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False
# code from: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# by Ansh Riyal
"""
example:
p1 = Point(1, 1) 
q1 = Point(10, 1) 
p2 = Point(1, 2) 
q2 = Point(10, 2) 
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
"""

def IsEdgeIntersectObs(node1, node2, Cobs: ObsList) -> bool:

    for obs in Cobs.obsList:
        p1 = Point(node1[0], node1[1])
        q1 = Point(node2[0], node2[1])

        p2 = Point(obs.x, obs.y)
        q2 = Point(obs.x + obs.nx, obs.y)

        p3 = Point(obs.x, obs.y)
        q3 = Point(obs.x, obs.y + obs.ny)

        p4 = Point(obs.x, obs.y+obs.ny)
        q4 = Point(obs.x + obs.nx, obs.y + obs.ny)

        p5 = Point(obs.x + obs.nx, obs.y)
        q5 = Point(obs.x + obs.nx, obs.y + obs.ny)
        if doIntersect(p1, q1, p2, q2) or doIntersect(p1, q1, p3, q3) or \
                doIntersect(p1, q1, p4, q4) or doIntersect(p1, q1, p5, q5):
            return True
    #  doesn't intersect with all obstacles.
    return False

class GraphPRM(NamedTuple):
    edge_matrix: np.ndarray
    nodes_pos: dict


def FillEdgesMatrix(edge_matrix: np.ndarray, nodes_pos: dict, thresh: float,
                    Cobs: ObsList):
    assert edge_matrix is not None and nodes_pos is not None and thresh and Cobs
    num_of_nodes = len(nodes_pos)
    print("calculating edges...")
    for i in tqdm(range(num_of_nodes)):
        for j in range(i+1, num_of_nodes):
            dist = np.linalg.norm(np.array(nodes_pos[i])-np.array(nodes_pos[j]))
            if dist < thresh and IsEdgeIntersectObs(nodes_pos[i], nodes_pos[j],
                                                    Cobs) is False:
                edge_matrix[i][j] = dist
                edge_matrix[j][i] = edge_matrix[i][j]




# Q1 Probabilistic Road Map
# section (a) - implementing the function GeneratePRM
"""
Function: GeneratePRM
Input:
    N - the gird or 'world size' where we consider NXN grid
    thd - threshold parameter. Euclidian diastance under this threshold is
     considered close.
    Nnodes - number of nodes to the graph to have.
    Cobs - a set of obstacles. each item in the set is a struct with the fields
     x,y,nx,ny. where (x,y) is the bot left coordinate of the rectangle and
      nx,ny are his dimentions in the x and y axis' respectively.
output:
    G=(V,E) - a graph g made out of nodes V(vertex) and E(edges). the graph is
     represented a a connectivity matrix with each cell is a weight. and the
     positions of all nodes is a dictionary n->(x,y) meaning the dictionary
     maps natural numbers to 2-tuple location.   
"""


def GeneratePRM(N: int, thd: float, Nnodes: int, Cobs: ObsList):
    assert N > 0 and thd > 0 and Nnodes > 0 and Cobs is not None
    size = (Nnodes, Nnodes)
    # init connectivity matrix. 'far away' nodes has infinity distance.
    edge_matrix = np.full(size, np.inf, dtype=float)
    nodes_pos = {}  # init the nodes holder
    #  sample nodes from the grid and fill the dictionary with the
    #  sampled locations
    for i in range(Nnodes):
        location = SampleNode(Cobs, N)
        nodes_pos[i] = location
    FillEdgesMatrix(edge_matrix, nodes_pos, thd, Cobs)
    return GraphPRM(edge_matrix, nodes_pos)


def GenerateObs(n_obs: int, nx: float, ny: float, N: int) -> ObsList:
    assert n_obs and nx and ny and N
    # according to the size of the obstacles find correct range for sampling
    upperX = N - nx
    upperY = N - ny
    obs_list = ObsList()
    for i in range(n_obs):
        x = np.random.uniform(low=0.0, high=float(upperX))
        y = np.random.uniform(low=0.0, high=float(upperY))
        obs_list.push_obs(Obs(x, y, nx, ny))
    return obs_list


def plotPRM(graphPRM: GraphPRM, Cobs: ObsList, thd: float):
    edges = graphPRM.edge_matrix
    nodes = graphPRM.nodes_pos

    fig = plt.figure()
    # Get the current reference
    ax = fig.gca()
    # add rectangles
    for obs in Cobs.obsList:
        # Create a Rectangle patch
        rect = patches.Rectangle((obs.x, obs.y), obs.nx, obs.ny, linewidth=1,
                                 edgecolor='r', facecolor='r')
        # Add the patch to the Axes
        ax.add_patch(rect)
    # add edges
    edge_counter = 0
    tot_nodes_degree = 0
    print("plotting edges...")
    for i in tqdm(range(len(nodes))):
        node_edges = edges[i][:]  # e.g. [inf,19.1,in,in,5.0,inf...]
        node_degree = len(node_edges[node_edges < thd + 1])
        tot_nodes_degree += node_degree
        for j in range(i+1, len(nodes)):
            if edges[i][j] < thd + 1:
                edge_counter += 1
                x1 = (nodes[i])[0]
                x2 = (nodes[j])[0]
                y1 = (nodes[i])[1]
                y2 = (nodes[j])[1]
                plt.plot([x1, x2], [y1, y2], c='b', linewidth=0.2)
    mean_node_degree = tot_nodes_degree / len(nodes)
    # add the nodes
    for i in range(len(nodes)):
        loc = nodes[i]
        plt.scatter(loc[0], loc[1], c='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'#nodes={len(nodes)}, th={thd}, #edges={edge_counter},'
              f' mean node degree = {mean_node_degree}')
    plt.show()


def main():
    N = 100
    n_obs = 15
    nx = 15.0
    ny = 10.0
    Nnodes = [100, 500]
    thd = [20, 50]
    Cobs = GenerateObs(n_obs, nx, ny, N)
    graphPRM1 = GeneratePRM(N, thd[0], Nnodes[0], Cobs)  # 1000 nodes,  20 thresh
    plotPRM(graphPRM1, Cobs, thd[0])
    # graphPRM2 = GeneratePRM(N, thd[1], Nnodes[0], Cobs)  # 1000 nodes,  50 thresh
    # plotPRM(graphPRM2, Cobs, thd[1])
    # graphPRM3 = GeneratePRM(N, thd[0], Nnodes[1], Cobs)  # 5000 nodes,  20 thresh
    # plotPRM(graphPRM3, Cobs, thd[0])
    # graphPRM4 = GeneratePRM(N, thd[1], Nnodes[1], Cobs)  # 5000 nodes,  50 thresh
    # plotPRM(graphPRM4, Cobs, thd[1])


if __name__ == '__main__':
    main()
