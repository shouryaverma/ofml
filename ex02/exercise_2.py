# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>
from itertools import chain

import pulp


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
    mu = list(chain(*[list(node_vars.values())for node_vars in mu_N]))
    mu += list(chain(*[list(edge_vars.values())for edge_vars in mu_E]))

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
                    ilp += (pulp.lpSum([mu_E[edge_id][edge_key] for edge_key in mu_E[edge_id] if edge_key[0] == var_key]) == node_var,
                            "left_node_{}_to_right_node_{}_label{}_coupling_constraint".format(edge.left, edge.right, var_key))
            elif edge.right == node_id:
                for var_key, node_var in mu_N[node_id].items():
                    ilp += (pulp.lpSum([mu_E[edge_id][edge_key] for edge_key in mu_E[edge_id] if edge_key[1] == var_key]) == node_var,
                            "right_node_{}_to_left_node_{}_label{}_coupling_constraint".format(edge.right, edge.left, var_key))

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
