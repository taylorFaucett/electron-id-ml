import pandas as pd
import energyflow as ef
import glob
import numpy as np
import os
from tqdm import tqdm
import shutil
import itertools
from natsort import natsorted


def norm(x):
    normed = (x - min(x)) / (max(x) - min(x))
    return normed


def compare_dataframes():
    splits = ["test", "train", "valid"]
    cal_types = ["et", "ht"]
    for split in splits:
        for cal_type in cal_types:
            files = glob.glob(f"data/efp_data/{split}/*.feather")
            file_pairs = list(itertools.combinations(files, 2))
            t = tqdm(file_pairs)
            for p1, p2 in file_pairs:
                efp1 = pd.read_feather(p1).set_index("features").sort_index()
                efp2 = pd.read_feather(p2).set_index("features").sort_index()
                if efp1.equals(efp2):
                    print(p2)


def check_compression_errors(remove_broken=False):
    existing_graphs = natsorted(glob.glob("data/efp/**/*.feather"))
    comp_errors = []
    for efp in tqdm(existing_graphs):
        try:
            data = pd.read_feather(efp)
        except:
            print("loading error: " + efp)
            if remove_broken:
                os.remove(efp)


def efp_500():
    # Comparing pairs of EFPs is slow with complete datasets
    # So this function just generates a file with the first 500 events of each
    # EFP with the corresponding EFP label

    # We only care about either just et or just ht since the
    # duplicates will be based on EFP number, kappa and beta.
    # So we can just find duplicates in et and apply the rules to removing ht cases
    file_out = "data/processed/efp500.feather"
    existing_graphs = natsorted(glob.glob("data/efp/test/et_*.feather"))
    df0 = pd.DataFrame()
    for efp in tqdm(existing_graphs):
        file_name = efp.split("/")[-1].split(".feather")[0]
        data = norm(pd.read_feather(efp).head(500).features.values)
        dfi = pd.DataFrame({f"{file_name}": data})
        df0 = pd.concat([df0, dfi], axis=1)
    df0.to_feather(file_out)


def dup_search():
    # Get the first 500 entries for each EFP
    efp_500 = pd.read_feather("data/processed/efp500.feather")

    # Get all EFP labels
    efp_cols = list(efp_500.columns.values)

    # Get every combination of EFPs
    efp_pairs = list(itertools.combinations(efp_cols, 2))

    # Loop through pairs and find any pairs with identical values
    duplicates = []
    t = tqdm(efp_pairs)
    for ix, (p1, p2) in enumerate(t):
        efp1 = np.around(efp_500[p1].values, 7)
        efp2 = np.around(efp_500[p2].values, 7)
        if (efp1 == efp2).all():
            duplicates.append(str(p2))

    # Take just the unique EFPs (i.e. remove duplicates from the efp_cols list)
    uniques = list(set(efp_cols) - set(duplicates))

    # Remove 'et' label to generalize for both et and ht data types
    uniques = [x.split("et_")[-1] for x in uniques]
    uniques = pd.DataFrame({"efp": uniques})

    # Output unique EFPs
    file_out = "data/processed/uniques.feather"
    uniques.to_feather(file_out)

    # Remove 'et' label to generalize for both et and ht data types
    duplicates = [x.split("et_")[-1] for x in duplicates]
    duplicates = pd.DataFrame({"efp": duplicates})

    # Output duplicate EFPs
    file_out = "data/processed/duplicates.feather"
    duplicates.to_feather(file_out)


def move_duplicates(move_dupes=False):
    uniques = pd.read_feather("data/processed/uniques.feather").values.T[0]
    duplicates = pd.read_feather("data/processed/duplicates.feather").values.T[0]
    splits = ["test", "train", "valid"]
    cal_types = ["et", "ht"]
    print(f"Removing N={len(duplicates)} duplicates")
    for bad_file in duplicates:
        for split in splits:
            for cal_type in cal_types:
                source_file = f"data/efp/{split}/{cal_type}_{bad_file}.feather"
                destination_file = (
                    f"data/efp/duplicates/{split}/{cal_type}_{bad_file}.feather"
                )
                if os.path.exists(source_file):
                    print(f"moving {source_file} -> {destination_file}")
                    if move_dupes:
                        shutil.move(source_file, destination_file)


if __name__ == "__main__":
    check_compression_errors()
    # efp_500()
    # dup_search()
    # move_duplicates(move_dupes=False)
    print("Done!")
