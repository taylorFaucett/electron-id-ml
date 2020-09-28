#! /home/tfaucett/miniconda3/envs/tf/bin/python

import energyflow as ef
import igraph
import string
from tqdm import trange

rainbow_colors = [
    "#9400D3",
    "#4B0082",
    "#0000FF",
    "#00FF00",
    "#FFFF00",
    "#FF7F00",
    "#FF0000",
]


def plot_graph(graph, n, d, k, plot_file):
    # Create and plot the graph with igraph
    g = igraph.Graph()
    g.add_vertices(n)
    g.add_edges(graph)
    # https://igraph.org/python/doc/igraph-pysrc.html#Graph.__plot__
    igraph.plot(
        g,
        plot_file,
        layout=g.layout("circular"),  # rt = tree
        bbox=(350, 350),
        margin=50,
        background=None,
        vertex_color="#E31A1C",  # rainbow_colors
        vertex_size=40,
        # vertex_label=alphabet_list[:n],
        vertex_label_size=22,
        vertex_label_color="white",
        vertex_label_dist=0,
        vertex_shape="circle",  # "square", "triangle", "triangle-down"
        vertex_frame_color="#333333",
        vertex_frame_width=2,
        edge_width=4,
        edge_color="#333333",
        # edge_curved=0,
    )


def generate_efp_graphs():
    # Grab graphs
    prime_d7 = ef.EFPSet("d<=7", "p==1")
    chrom_4 = ef.EFPSet("d<=8", "p==1", "c==4")
    efpsets = [prime_d7, chrom_4]
    for efpset in efpsets:
        graphs = efpset.graphs()
        for efp_ix, graph in enumerate(graphs):
            n, e, d, v, k, c, p, h = efpset.specs[efp_ix]
            plot_graph(graph, n, d, k, f"graphs/pdf/efp_{n}_{d}_{k}.pdf")
            plot_graph(graph, n, d, k, f"graphs/png/efp_{n}_{d}_{k}.png")
            plot_graph(graph, n, d, k, f"graphs/eps/efp_{n}_{d}_{k}.eps")

        # chrom_4 = ef.EFPSet("d<=8", "p==1", "c==4")
        # graphs = chrom_4.graphs()
        # for efp_ix, graph in enumerate(graphs):
        #     n, e, d, v, k, c, p, h = chrom_4.specs[efp_ix]
        #     pdf_file = f"graphs/pdf/efp_{n}_{d}_{k}.pdf"
        #     plot_graph(graph, n, d, k, pdf_file)
        #     png_file = f"graphs/png/efp_{n}_{d}_{k}.pdf"
        #     plot_graph(graph, n, d, k, png_file)


if __name__ == "__main__":
    alphabet_string = string.ascii_lowercase
    alphabet_list = list(alphabet_string)
    generate_efp_graphs()
