# This is the file where should insert your own code.
#
# Exercise sheet 2:
# Shourya Verma: shourya.verma@stud.uni-heidelberg.de
# Marvin Bundschuh: marvin.bundschuh@stud.uni-heidelberg.de
# Almoatasem bellah Haggag: almoatasem.haggag@stud.uni-heidelberg.de

from itertools import chain
import pulp

import imageio
import numpy as np
import glob
from collections import namedtuple
import matplotlib.pyplot as plt
from tabulate import tabulate

from model_2_4 import ACYCLIC_MODELS, CYCLIC_MODELS

# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

from collections import namedtuple
import gzip
import re

Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


#
# Internal functions to read the input file format.
#

def tokenize_file(f):
    r = re.compile(r'[a-zA-Z0-9.]+')
    for line in f:
        line = line.rstrip('\r\n')
        for m in r.finditer(line):
            yield m.group(0)


def parse_uai(tokens):
    header = next(tokens)
    assert header == 'MARKOV'
    num_nodes = int(next(tokens))
    nodes = [None] * num_nodes
    edges = []
    for i in range(num_nodes):
        nodes[i] = Node(costs=[0] * int(next(tokens)))

    node_list = []
    num_costs = int(next(tokens))
    for i in range(num_costs):
        num_vars = int(next(tokens))
        node_list.append(tuple(int(next(tokens)) for j in range(num_vars)))

    cost_cache = {}
    for i in range(num_costs):
        size = int(next(tokens))
        if len(node_list[i]) == 1:
            u, = node_list[i]
            assert size == len(nodes[u].costs)
            for x_u in range(len(nodes[u].costs)):
                nodes[u].costs[x_u] = float(next(tokens))
        elif len(node_list[i]) == 2:
            u, v = node_list[i]
            costs = {}
            assert size == len(nodes[u].costs) * len(nodes[v].costs)
            for x_u in range(len(nodes[u].costs)):
                for x_v in range(len(nodes[v].costs)):
                    costs[(x_u, x_v)] = float(next(tokens))

            cache_key = repr(costs)
            try:
                costs = cost_cache[cache_key]
            except KeyError:
                cost_cache[cache_key] = costs

            edges.append(Edge(left=u, right=v, costs=costs))
        else:
            raise RuntimeError('Higher-order factors not supported.')
    return nodes, edges


def load_uai(filename):
    open_func = open
    if filename.endswith('.gz'):
        open_func = gzip.open

    with open_func(filename, 'rt') as f:
        return parse_uai(tokenize_file(f))


#
# Ready-to-use models for exercise.
#

ALL_MODEL_DOWNSAMPLINGS = [1, 2, 4, 8, 16, 32]


def load_downsampled_model(downsampling):
    assert downsampling in ALL_MODEL_DOWNSAMPLINGS
    filename = 'models/tsu_{:02d}.uai.gz'.format(downsampling)
    return load_uai(filename)


def all_models():
    for downsampling in reversed(ALL_MODEL_DOWNSAMPLINGS):
        yield load_downsampled_model(downsampling)


__all__ = ['ALL_MODEL_DOWNSAMPLINGS', 'load_downsampled_model', 'all_models']

def convert_to_lp(nodes, edges):
    lp = pulp.LpProblem('GM')
    # populate LP
    mu_N = []  # label pick vector Vertices
    mu_E = []  # label pick vector Edges
    theta = []  # cost-vector

    # first set cost-vector and optimization variables
    for id_node, node in enumerate(nodes):
        label_vars = pulp.LpVariable.dicts("node_{}_label".format(id_node),
                                           range(len(node.costs)),
                                           lowBound=0, upBound=1,
                                           cat=pulp.LpContinuous)
        mu_N.append(label_vars)
        theta += node.costs

    for id_edge, edge in enumerate(edges):
        edge_vars = pulp.LpVariable.dicts("edge_{}_label".format(id_edge),
                                          list(edge.costs.keys()),
                                          lowBound=0, upBound=1,
                                          cat=pulp.LpContinuous)
        mu_E.append(edge_vars)
        theta += list(edge.costs.values())

    # set up mu-vector from the dicts
    mu = list(chain(*[list(node_vars.values()) for node_vars in mu_N]))
    mu += list(chain(*[list(edge_vars.values()) for edge_vars in mu_E]))

    # set up optimization function
    lp += (pulp.lpDot(mu, theta), "optimization function")

    # set up simplex constraints
    for node_id, node_vars in enumerate(mu_N):
        lp += (pulp.lpSum(list(node_vars.values())) == 1, "node_{}_simplex_constraint".format(node_id))

    for edge_id, edge_vars in enumerate(mu_E):
        lp += (pulp.lpSum(list(edge_vars.values())) == 1, "edge_{}_simplex_constraint".format(edge_id))

    # set up coupling constraints
    for node_id, node in enumerate(nodes):
        for edge_id, edge in enumerate(edges):
            if edge.left == node_id:
                for var_key, node_var in mu_N[node_id].items():
                    lp += (pulp.lpSum(
                        [mu_E[edge_id][edge_key] for edge_key in mu_E[edge_id] if edge_key[0] == var_key]) == node_var,
                           "left_node_{}_to_right_node_{}_label{}_coupling_constraint".format(edge.left, edge.right,
                                                                                              var_key))
            elif edge.right == node_id:
                for var_key, node_var in mu_N[node_id].items():
                    lp += (pulp.lpSum(
                        [mu_E[edge_id][edge_key] for edge_key in mu_E[edge_id] if edge_key[1] == var_key]) == node_var,
                           "right_node_{}_to_left_node_{}_label{}_coupling_constraint".format(edge.right, edge.left,
                                                                                              var_key))
    # set up greater than 0 constraints
    for var_id, var in enumerate(mu):
        lp += (var >= 0, "greater_than_0_constraint_{}".format(var_id))

    return lp


def lp_to_labeling(nodes, edges, lp):
    labeling = [None] * len(nodes)
    # compute labeling
    lp.solve()
    solved_nodes = [node_solution for node_solution in lp.variables() if node_solution.name[:4] == "node"]
    current_node_number, current_node_label, max_value = -1, -1, -1
    for node_solution in solved_nodes:
        node_number, node_label = [int(text) for text in node_solution.name.split("_") if text.isdigit()]
        if current_node_number != node_number:
            labeling[current_node_number] = current_node_label
            current_node_number, current_node_label, max_value = node_number, node_label, -1
        if node_solution.varValue > max_value:
            current_node_label = node_label
            max_value = node_solution.varValue
    labeling[current_node_number] = current_node_label
    return labeling


def labelling_to_energy(nodes, edges, labeling):
    energy = 0
    for node_id, node in enumerate(nodes):
        energy += node.costs[labeling[node_id]]
    for edge_id, edge in enumerate(edges):
        energy += edge.costs[(labeling[edge.left], labeling[edge.right])]
    return energy


def convert_to_ilp(nodes, edges):
    ilp = pulp.LpProblem('GM')

    # populate ILP
    mu_N = []  # label pick vector Vertices
    mu_E = []  # label pick vector Edges
    theta = []  # cost-vector

    # first set cost-vector and optimization variables
    for id_node, node in enumerate(nodes):
        label_vars = pulp.LpVariable.dicts("node_{}_label".format(id_node),
                                           range(len(node.costs)),
                                           lowBound=0, upBound=1,
                                           cat=pulp.LpInteger)
        mu_N.append(label_vars)
        theta += node.costs

    for id_edge, edge in enumerate(edges):
        edge_vars = pulp.LpVariable.dicts("edge_{}_label".format(id_edge),
                                          list(edge.costs.keys()),
                                          lowBound=0, upBound=1,
                                          cat=pulp.LpInteger)
        mu_E.append(edge_vars)
        theta += list(edge.costs.values())

    # set up mu-vector from the dicts
    mu = list(chain(*[list(node_vars.values()) for node_vars in mu_N]))
    mu += list(chain(*[list(edge_vars.values()) for edge_vars in mu_E]))

    # set up optimization function
    ilp += (pulp.lpDot(mu, theta), "optimization function")

    # set up simplex constraints
    for node_id, node_vars in enumerate(mu_N):
        ilp += (pulp.lpSum(list(node_vars.values())) == 1, "node_{}_simplex_constraint".format(node_id))

    for edge_id, edge_vars in enumerate(mu_E):
        ilp += (pulp.lpSum(list(edge_vars.values())) == 1, "edge_{}_simplex_constraint".format(edge_id))

    # set up coupling constraints
    for node_id, node in enumerate(nodes):
        for edge_id, edge in enumerate(edges):
            if edge.left == node_id:
                for var_key, node_var in mu_N[node_id].items():
                    ilp += (pulp.lpSum(
                        [mu_E[edge_id][edge_key] for edge_key in mu_E[edge_id] if edge_key[0] == var_key]) == node_var,
                            "left_node_{}_to_right_node_{}_label{}_coupling_constraint".format(edge.left, edge.right,
                                                                                               var_key))
            elif edge.right == node_id:
                for var_key, node_var in mu_N[node_id].items():
                    ilp += (pulp.lpSum(
                        [mu_E[edge_id][edge_key] for edge_key in mu_E[edge_id] if edge_key[1] == var_key]) == node_var,
                            "right_node_{}_to_left_node_{}_label{}_coupling_constraint".format(edge.right, edge.left,
                                                                                               var_key))

    # set up greater than 0 constraints
    for var_id, var in enumerate(mu):
        ilp += (var >= 0, "greater_than_0_constraint_{}".format(var_id))

    return ilp


def ilp_to_labeling(nodes, edges, ilp):
    labeling = [None] * len(nodes)
    # compute labeling
    ilp.solve()
    solved_nodes = [node_solution for node_solution in ilp.variables() if node_solution.name[:4] == "node"]

    for node_solution in solved_nodes:
        if node_solution.varValue == 1:
            node_number, node_label = [int(text) for text in node_solution.name.split("_") if text.isdigit()]
            labeling[node_number] = node_label

    return labeling


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


# helper functions
def make_pairwise(shape):
    c = {}
    for x_u in range(shape[0]):
        for x_v in range(shape[1]):
            c[x_u, x_v] = 1 if x_u == x_v else 0
    return c


def make_graph():
    nodes = [Node(costs=[0.5, 0.5]),
             Node(costs=[0.0, 0.0]),
             Node(costs=[0.2, 0.2])]

    edges = []
    for u, v in ((0, 1), (0, 2), (1, 2)):
        shape = tuple(len(nodes[x].costs) for x in (u, v))
        edges.append(Edge(left=u, right=v, costs=make_pairwise(shape)))

    return nodes, edges


def exercise_2_3():
    nodes, edges = make_graph()
    lp = convert_to_lp(nodes, edges)
    res = lp.solve()
    labellling = lp_to_labeling(nodes, edges, lp)
    relaxed_energy = lp.objective.value()
    energy = labelling_to_energy(nodes, edges, labellling)
    assert res

    ilp = convert_to_ilp(nodes, edges)
    res2 = ilp.solve()
    labellling2 = ilp_to_labeling(nodes, edges, ilp)
    energy2 = ilp.objective.value()
    assert res2

    print("lp example")
    for var in lp.variables():
        print('{} -> {}'.format(var.name, var.value()))
    print("rounded solution ->  {}".format(labellling))
    print("relaxed energy -> {}".format(relaxed_energy))
    print("rounded energy -> {}".format(energy))
    print("----------")

    print("ilp example")
    for var in ilp.variables():
        print('{} -> {}'.format(var.name, var.value()))
    print("actual solution ->  {}".format(labellling2))
    print("actual energy -> {}".format(energy2))

# The ILP in this example has more than one optimal solution that is equally good. The optimal solutions are
# [1, 1, 0], [1,0,1] , [0, 1 , 1], [0, 0, 1], [0, 1, 0] and [1, 0, 0]. So there are as many optimal solution where
# node i is 0, as there are optimal solutions where i is labelled 1. This leads to a fractional result, where every
# node-label is equally plausibel (so 0.5). When rounding, we have to decide for one of the nodes, although it is
# impossible to say which label to choose, because they have the same possibility to be right. In short, we haven't
# won everything by using the ilp relaxation, because at this point randomly guessing is the best possible strategy.
# In our implementation of lp_to_labelling we just always take the first label with the max_label when there are two
# equally good labels. Which of course leads us to the solution [0, 0, 0] which is worse then one of the real optinal
# solutions.


def exercise_2_4():
    acyclic_table = {"model": [],
                     "LP optima": [],
                     "ILP optima": []}

    for model_id, (nodes, edges) in enumerate(ACYCLIC_MODELS):
        lp = convert_to_lp(nodes, edges)
        lp.solve()

        ilp = convert_to_ilp(nodes, edges)
        ilp.solve()

        acyclic_table["model"].append("acylic model {}".format(model_id + 1))
        acyclic_table["LP optima"].append(lp.objective.value())
        acyclic_table["ILP optima"].append(ilp.objective.value())

    cyclic_table = {"model": [],
                    "LP optima": [],
                    "ILP optima": []}

    for model_id, (nodes, edges) in enumerate(CYCLIC_MODELS):
        lp = convert_to_lp(nodes, edges)
        lp.solve()

        ilp = convert_to_ilp(nodes, edges)
        ilp.solve()

        cyclic_table["model"].append("cylic model {}".format(model_id + 1))
        cyclic_table["LP optima"].append(lp.objective.value())
        cyclic_table["ILP optima"].append(ilp.objective.value())

    print("acyclic table")
    print(tabulate(acyclic_table,
                   headers='keys',
                   tablefmt='fancy_grid'
                   )
          )

    print("cyclic table")
    print(tabulate(cyclic_table,
                   headers='keys',
                   tablefmt='fancy_grid'
                   )
          )


def exercise_2_5():
    table = {"Model": [],
             "Image Dimensions": [],
             "#vars": [],
             "#contrs": [],
             "LP optima": [],
             "ILP optima": [],
             "Inference Time": []}

    dimensions = [(384, 288), (192, 144), (96, 72), (48, 36), (24, 18), (12, 9)]
    files = [1, 2, 4, 8, 16, 32]

    for idx, model_id in enumerate(files):
        # load nodes and edges from all downsampled model files
        nodes, edges = load_downsampled_model(model_id)

        # set tolerance level to slice and skip values, set to 50 to run quick, change to 10 or 5 for longer better runs
        tolerance = int(50 / model_id)
        nodes = nodes[::tolerance]
        edges = edges[::tolerance]

        # convert and solve lp and ilp problem
        lp = convert_to_lp(nodes, edges)
        ilp = convert_to_ilp(nodes, edges)

        lp.solve()
        ilp.solve()

        # set hard coded dimensions since IDK how to get them dynamically from the model file
        x, y = dimensions[idx]
        table["Model"].append("model {}".format(model_id))
        table["Image Dimensions"].append('{} X {}'.format(x, y))
        table["#vars"].append(lp.numVariables())
        table["#contrs"].append(lp.numConstraints())
        table["LP optima"].append(lp.objective.value())
        table["ILP optima"].append(ilp.objective.value())
        table["Inference Time"].append(lp.solutionTime)

    print("table")
    print(tabulate(table,
                   headers='keys',
                   tablefmt='fancy_grid'
                   )
          )

# What we observed: The Optima of the LP and ILP of the acyclic graph are always the same.
#                   The Optima of the LP of the cyclic graph are always better than the ILP optima.


def exercise_2_6():
    # Read images and segments
    all_segments = []
    all_images = []
    for i, img_path in enumerate(glob.glob("./segments/segments/*.png")):
        im = imageio.v2.imread(img_path)
        all_images.append(im)
        mask = im != 0
        positions = np.transpose(np.nonzero(mask))
        for x, y in positions:
            all_segments.append((x, y, 1, 1))

    # Create the ILP model
    ilp = pulp.LpProblem("segments", pulp.LpMaximize)

    # Create variables
    vars = {}
    for i, (x, y, w, h) in enumerate(all_segments):
        vars[i] = pulp.LpVariable(str(i), 0, 1, pulp.LpInteger)

    # add objective to maximize the total covered area
    ilp += pulp.lpSum(vars[i] * w * h for i in range(len(all_segments)))

    # add constraints so all_segments do not overlap
    for i in range(len(all_segments)):
        for j in range(i + 1, len(all_segments)):
            x1, y1, w1, h1 = all_segments[i]
            x2, y2, w2, h2 = all_segments[j]
            if (x1 < x2 + w2) and (x2 < x1 + w1) and (y1 < y2 + h2) and (y2 < y1 + h1):
                ilp += vars[i] + vars[j] <= 1

    # solve ilp
    ilp.solve()

    solved_segments = []
    # get solution and plot final image
    for i, v in enumerate(ilp.variables()):
        if v.varValue == 1.0:
            solved_segments.append(all_images[i])

    plt.imshow(sum(solved_segments))
    plt.savefig('output.png')


if __name__ == "__main__":
    # 1 and 2 can be tested through the provided tests
    print("ecercise 3 -------------------------------------------------")
    exercise_2_3()
    print("------------------------------------------------------------")
    print("ecercise 4 -------------------------------------------------")
    exercise_2_4()
    print("------------------------------------------------------------")

    exercise_2_5()
    exercise_2_6()


