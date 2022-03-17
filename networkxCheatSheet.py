import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#### undirected graph
G = nx.Graph()
G.add_nodes_from([
    (1, {"color": "red"}),
    (2, {"color": "green"}),
])
G.add_edge(1, 2)

#### directed graph
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
DG.out_degree(1, weight='weight')

#### generate graphs sic
ws = nx.watts_strogatz_graph(30, 3, 0.1)# Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])

#### saving graphs adn import
nx.write_gml(ws, "path.to.file")
mygraph = nx.read_gml("path.to.file")

#### graph drawing 
subax1 = plt.subplot(121)
nx.draw(ws, with_labels=True, font_weight='bold')
subax2 = plt.subplot(122)
nx.draw_shell(ws, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
plt.savefig("path.png")
plt.show()