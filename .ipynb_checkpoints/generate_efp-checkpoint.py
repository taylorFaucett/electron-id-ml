import h5py
import numpy as np
import pandas as pd
import energyflow as ef
import glob
import os
from tqdm import tqdm, trange
from itertools import product


def efp(data, graph, kappa, beta, normed):
    EFP_graph = ef.EFP(graph, measure="hadr", kappa=kappa, beta=beta, normed=normed)
    X = EFP_graph.batch_compute(data)
    return X


def generate_EFP():
    # Calculate HL variables from ET, eta, phi
    dtypes = ["et", "ht"]
    splits = ["test", "train", "valid"]
    for dtype in dtypes:
        for split in splits:
            print(f"Processing data type: {dtype}, {split}")
            X = pd.read_pickle(f"data/processed/{split}_{dtype}.pkl")["features"]
            y = pd.read_pickle(f"data/processed/y_{split}_{dtype}.pkl")["targets"]

            # Choose kappa, beta values
            kappas = [-1, 0, 0.5, 1, 2]
            betas = [0.5, 1, 2]

            # Grab graphs
            prime_d7 = ef.EFPSet("d<=7", "p==1")
            chrom_4 = ef.EFPSet("d<=8", "p==1", "c==4")
            efpsets = [prime_d7, chrom_4]
            for efpset in efpsets:
                graphs = efpset.graphs()
                t = tqdm(graphs)
                for efp_ix, graph in enumerate(t):
                    for kappa in kappas:
                        for beta in betas:
                            n, e, d, v, k, c, p, h = efpset.specs[efp_ix]
                            file_name = f"data/efp/{split}/{dtype}_efp_{n}_{d}_{k}_k_{kappa}_b_{beta}.feather"
                            if not os.path.exists(file_name):
                                graph = graphs[efp_ix]
                                t.set_description(
                                    f"Procesing: EFP[{n},{d},{k}](k={kappa},b={beta})"
                                )
                                efp_val = efp(
                                    data=X,
                                    graph=graph,
                                    kappa=kappa,
                                    beta=beta,
                                    normed=False,
                                )
                                efp_df = pd.DataFrame(
                                    {f"features": efp_val, f"targets": y}
                                )
                                efp_df.to_feather(file_name)
                            # else:
                            #     efp1 = pd.read_feather(file_name).head().features.values
                            #     efp2 = efp(
                            #         data=X.head(),
                            #         graph=graph,
                            #         kappa=kappa,
                            #         beta=beta,
                            #         normed=False,
                            #     )
                            #     if not (efp1 == efp2).all():
                            #         print(f"Discrepancy found in: {file_name}")


if __name__ == "__main__":
    generate_EFP()
