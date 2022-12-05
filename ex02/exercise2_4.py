from tabulate import tabulate

from model_2_4 import *
import exercise_2 as student

acyclic_table = {"model": [],
                 "LP optima": [],
                 "ILP optima": []}

for model_id, (nodes, edges) in enumerate(ACYCLIC_MODELS):
    lp = student.convert_to_lp(nodes, edges)
    lp.solve()

    ilp = student.convert_to_ilp(nodes, edges)
    ilp.solve()

    acyclic_table["model"].append("acylic model {}".format(model_id + 1))
    acyclic_table["LP optima"].append(lp.objective.value())
    acyclic_table["ILP optima"].append(ilp.objective.value())

cyclic_table = {"model": [],
                "LP optima": [],
                "ILP optima": []}

for model_id, (nodes, edges) in enumerate(CYCLIC_MODELS):
    lp = student.convert_to_lp(nodes, edges)
    lp.solve()

    ilp = student.convert_to_ilp(nodes, edges)
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

# What we observed: The Optima of the LP and ILP of the acyclic graph are always the same.
#                   The Optima of the LP of the cyclic graph are always better than the ILP optima.
# Reason: TODO!
