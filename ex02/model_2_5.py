# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

from tsukuba import *
from tsukuba_visualize import *

from tabulate import tabulate

from model_2_4 import *
import exercise_2 as student

# Nodes, Edges = load_downsampled_model(32)

table = {"model": [],
        "image dimensions":[],
        "#vars":[],
        "#contrs":[],
        "LP optima": [],
        "ILP optima": []}

for model_id in [1,2,4,8,16,32]:
    nodes, edges = load_downsampled_model(model_id)
    lp = student.convert_to_lp(nodes, edges)
    lp.solve()

    ilp = student.convert_to_ilp(nodes, edges)
    ilp.solve()

    table["model"].append("model {}".format(model_id))
    table["image dimensions"].append(str(model_id)+'x'+str(model_id))
    table["#vars"].append('vars')
    table["#contrs"].append('constr')
    table["LP optima"].append(lp.objective.value())
    table["ILP optima"].append(ilp.objective.value())

print("table")
print(tabulate(table,
               headers='keys',
               tablefmt='fancy_grid'
               )
      )



