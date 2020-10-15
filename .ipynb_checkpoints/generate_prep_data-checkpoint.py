import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import energyflow as ef


def generate_prep_data():
    # Load jet-images
    df = h5py.File("data/raw/data.h5", mode="r")
    dtypes = ["et", "ht"]
    splits = ["test", "train", "valid"]
    data_types = [(x, y) for x in dtypes for y in splits]
    t = tqdm(data_types)
    for dtype, split in t:
        t.set_description(f"Processing {dtype} - {split}")
        t.refresh()
        # keys = ['ee', 'ee_mass', 'efpe', 'efph', 'et', 'et_mass', 'he', 'hl', 'ht', 'mass', 'y']
        print(dtype, split)
        y = df["y"][split][:]
        X = df[dtype][split][:]
        df0 = []
        if dtype == "et":
            eta_ = np.linspace(-0.3875, 0.3875, X.shape[2])
            phi_ = np.linspace(
                -(15.5 * np.pi) / 126.0, (15.5 * np.pi) / 126.0, X.shape[2]
            )
        elif dtype == "ht":
            eta_ = np.linspace(-0.4, 0.4, X.shape[2])
            phi_ = np.linspace(-(4 * np.pi) / 31.0, (4 * np.pi) / 31.0, X.shape[2])

        eta_phi = np.vstack([(x, y) for x in eta_ for y in phi_])
        eta_ = eta_phi[:, 0]
        phi_ = eta_phi[:, 1]
        for ix in trange(len(X)):
            et = X[ix].flatten()
            dfi = pd.DataFrame({"et": et, "eta": eta_, "phi": phi_})
            evt_out = dfi[(dfi[["et"]] != 0).all(axis=1)].to_numpy()
            evt_out[:, 0] /= np.sum(evt_out[:, 0])
            df0.append(evt_out)
        X0 = pd.DataFrame({"features": df0})
        y0 = pd.DataFrame({"targets": y})
        X0.to_pickle(f"data/processed/{split}_{dtype}.pkl")
        y0.to_pickle(f"data/processed/y_{split}_{dtype}.pkl")


if __name__ == "__main__":
    """ output: feature and target files
    
    generate_prep_data converts a jet image into [ET, eta, phi] format to be 
    processed by the energyflow package.
    """
    generate_prep_data()
