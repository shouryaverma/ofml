#!/usr/bin/env python3
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

import exercise_2 as student

from collections import namedtuple


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


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


def run_example():
    nodes, edges = make_graph()
    lp = student.convert_to_lp(nodes, edges)
    res = lp.solve()
    labellling = student.lp_to_labeling(nodes, edges, lp)
    relaxed_energy = lp.objective.value()
    energy = student.labelling_to_energy(nodes, edges, labellling)
    assert res

    ilp = student.convert_to_ilp(nodes, edges)
    res2 = ilp.solve()
    labellling2 = student.ilp_to_labeling(nodes, edges, ilp)
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


if __name__ == '__main__':
    run_example()

# The ILP in this example has more than one optimal solution that is equally good. The optimal solutions are
# [1, 1, 0], [1,0,1] , [0, 1 , 1], [0, 0, 1], [0, 1, 0] and [1, 0, 0]. So there are as many optimal solution where
# node i is 0, as there are optimal solutions where i is labelled 1. This leads to a fractional result, where every
# node-label is equally plausibel (so 0.5). When rounding, we have to decide for one of the nodes, although it is
# impossible to say which label to choose, because they have the same possibility to be right. In short, we haven't
# won everything by using the ilp relaxation, because at this point randomly guessing is the best possible strategy.
# In our implementation of lp_to_labelling we just always take the first label with the max_label when there are two
# equally good labels. Which of course leads us to the solution [0, 0, 0] which is worse then one of the real optinal
# solutions.