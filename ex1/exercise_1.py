#!/usr/bin/env python3
# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>
import numpy as np
import itertools

# For exercise 1.2
def evaluate_energy(nodes, edges, assignment):
    c = [node.costs for node in nodes]

    energy_node = 0
    for id, assign in enumerate(assignment):
         energy_node+=c[id][assign]
    
    energy_edge = 0
    for edge in edges:
        energy_edge+=edge.costs[(assignment[edge.left],assignment[edge.right])]

    return energy_node+energy_edge

# For exercise 1.3
def bruteforce(nodes, edges):
    assignment = [0] * len(nodes)
    # TODO: implement brute-force algorithm here...
    possibilities = []
    all_energy = []
    for node in nodes:
        p = [i for i in range(len(node.costs))]
        possibilities.append(p)
    assignments = list(itertools.product(*possibilities))
    for assignment in assignments:
        energy = evaluate_energy(nodes, edges, assignment)
        all_energy.append(energy)
        
    return (assignments[np.argmin(all_energy)], np.min(all_energy))


# For exercise 1.4
def dynamic_programming(nodes, edges):
    F, ptr = None, None
    return F, ptr

def backtrack(nodes, edges, F, ptr):
    assignment = [0] * len(nodes)
    return assignment


# For exercise 1.5
def compute_min_marginals(nodes, edges):
    m = [[0 for l in n] for n in nodes]
    return m


# For execrise 1.6
def dynamic_programming_tree(nodes, edges):
    F, ptr = None, None
    return F, ptr

def backtrack_tree(nodes, edges, F, ptr):
    assignment = [0] * len(nodes)
    return assignment
