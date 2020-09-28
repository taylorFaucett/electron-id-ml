from igraph import *

class graph_plot:
    def __init__(self, efp_ix):
        self.efp_ix = efp_ix

        efpset = ef.EFPSet("d<=7, measure="hadr", beta=1, normed=False, verbose=False)
        graph = efpset.graphs(efp_ix)
        n, _, d, v, _, c, p, _ = efpset.specs[efp_ix]

        g = Graph()
        g.add_vertices(n)
        g.add_edges(graph)
        plot(g, f"g{efp_ix}.svg", layout=g.layout("circle"), bbox=(350, 350), margin=50)