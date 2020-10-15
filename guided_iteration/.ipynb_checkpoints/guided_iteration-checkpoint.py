#! /home/tfaucett/miniconda3/envs/tf/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from os import path, getcwd, mkdir, environ, chdir
from nn import nn
import h5py
import itertools
import random
from sklearn import preprocessing
import json
import shutil
import sys
from sklearn.metrics import roc_auc_score
import warnings

from visualization import visualization

warnings.simplefilter(action="ignore", category=FutureWarning)
np.seterr(divide="ignore")

# Suppress all the tensorflow debugging info for new networks
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

home = path.dirname(getcwd())

scaler = preprocessing.StandardScaler()
# scaler = preprocessing.MinMaxScaler()
# scaler = preprocessing.RobustScaler()


def ado_calc(fx, gx, y):
    # Combine the data into a single dataframe
    dfy = pd.DataFrame({"fx": fx, "gx": gx, "y": y})

    # Separate data into signal and background
    dfy_sb = dfy.groupby("y")

    # Set signal/background
    df0 = dfy_sb.get_group(0)
    df1 = dfy_sb.get_group(1)

    # grab the fx and gx values for those sig/bkg pairs
    fx0 = df0["fx"].values
    fx1 = df1["fx"].values
    gx0 = df0["gx"].values
    gx1 = df1["gx"].values

    # sig/bkg might be different sizes so trim the longer of the two to match sizes
    max_size = min(len(fx0), len(fx1))
    fx0 = fx0[:max_size]
    fx1 = fx1[:max_size]
    gx0 = gx0[:max_size]
    gx1 = gx1[:max_size]

    # find differently ordered pairs
    dos = do_calc(fx0=fx0, fx1=fx1, gx0=gx0, gx1=gx1)

    # Average of the diff-order results
    ado_val = np.mean(dos)
    if ado_val < 0.5:
        ado_val = 1.0 - ado_val
    return ado_val


def do_calc(fx0, fx1, gx0, gx1):
    def heaviside(x):
        return 0.5 * (np.sign(x) + 1)

    dfx = fx0 - fx1
    dgx = gx0 - gx1
    dos = heaviside(np.multiply(dfx, dgx))
    return dos


def data_grabber(selected_efps, split, data_dir, efp_dir, incl_hl=False, incl_mass=False, incl_pT=False, normalize=False):
    hl_file = h5py.File(f"{data_dir}/raw/data.h5", "r")
    hl = hl_file["hl"][split][:]
    y = hl_file["y"][split][:]
    if incl_hl:
        df = pd.DataFrame(hl)
    else:
        df = pd.DataFrame()
    if incl_mass:
        mass = np.hstack(np.load(f"{data_dir}/raw/mass_data_{split}.npy"))
        mass_df = pd.DataFrame({"mass": mass})
        df = pd.concat([df, mass_df], axis=1)
    if incl_mass:
        pT = np.hstack(np.load(f"{data_dir}/raw/pT_{split}_et.npy"))
        pT_df = pd.DataFrame({"pT": mass})
        df = pd.concat([df, pT_df], axis=1)
    for efp in selected_efps:
        efp_file = f"{efp_dir}/{split}/{efp}.feather"
        dfi = pd.read_feather(efp_file)["features"]
        dfi = pd.DataFrame({efp: dfi.values})
        df = pd.concat([df, dfi], axis=1)
    if normalize:
        for column in df:
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return scaler.fit_transform(df), y


def isolate_order(ix, N_pairs):
    if path.isfile(f"{pass_dir}/dif_order.feather"):
        print(f"Skipping isolate_order for pass {ix}")
        dif_data = pd.read_feather(f"{pass_dir}/dif_order.feather")
        idxp0 = dif_data["idx0"].values
        idxp1 = dif_data["idx1"].values
        return idxp0, idxp1
    # Get the LL predictions
    ll_y_test = np.hstack(np.load(f"{data_dir}/raw/et_and_ht_test_pred.npy"))

    # Get the predictions from the previous iteration
    hl_file = f"{pass_dir}/test_pred.feather"

    # sim_file = f"iteration_data/p{ix}/sim_order.feather"
    dif_file = f"{pass_dir}/dif_order.feather"
    hl_test = pd.read_feather(hl_file)
    hl_y_test = hl_test["prediction"].values
    y_true = hl_test["y"].values

    # Combine the data into a single dataframe
    dfy = pd.DataFrame({"ll": ll_y_test, "hl": hl_y_test, "y": y_true})

    # Separate data into signal and background
    dfy_sb = dfy.groupby("y")

    # Set signal/background
    df0 = dfy_sb.get_group(0)
    df1 = dfy_sb.get_group(1)

    # get the separate sig/bkg indices
    # randomly shuffle the indices so we don't re-use the same sub-set
    idx0 = random.sample(df0.index.values.tolist(), N_pairs)[:N_pairs]
    idx1 = random.sample(df1.index.values.tolist(), N_pairs)[:N_pairs]

    # generate a random set of sig/bkg pairs
    print(f"Generating (N={N_pairs**2:,}) sig/bkg pairs")
    idx_pairs = np.vstack(list(itertools.product(idx0, idx1)))
    idxp0 = idx_pairs[:, 0]
    idxp1 = idx_pairs[:, 1]

    # grab the ll and hl values for those sig/bkg pairs
    dfy0 = dfy.iloc[idxp0]
    dfy1 = dfy.iloc[idxp1]
    ll0 = dfy0["ll"].values
    ll1 = dfy1["ll"].values
    hl0 = dfy0["hl"].values
    hl1 = dfy1["hl"].values

    # find differently ordered pairs
    dos = do_calc(fx0=ll0, fx1=ll1, gx0=hl0, gx1=hl1)

    # let's put all of the data and decision-ordering in 1 data frame
    do_df = pd.DataFrame(
        {
            "idx0": idxp0,
            "idx1": idxp1,
            "ll0": ll0,
            "ll1": ll1,
            "hl0": hl0,
            "hl1": hl1,
            "dos": dos,
        }
    )

    # split the similar and differently ordered sets
    do_df_grp = do_df.groupby("dos")
    dif_df = do_df_grp.get_group(0)

    # save the differently ordered choices
    dif_df.reset_index().to_feather(dif_file)
    return idxp0, idxp1


def check_efps(ix):
    if path.isfile(f"{pass_dir}/dif_order_ado_comparison.csv"):
        print(f"Skipping check_efps for pass {ix}")
        return
    # Load the diff-ordering results
    dif_df = pd.read_feather(f"{pass_dir}/dif_order.feather")

    # Grab the dif-order indices and ll features corresponding to those
    idx0 = dif_df["idx0"].values
    idx1 = dif_df["idx1"].values
    ll0 = dif_df["ll0"].values
    ll1 = dif_df["ll1"].values

    print(f"Checking ADO on diff-order subset of size N = {len(dif_df):,}")

    if irc_safe:
        print("Using IRC (k=1) graphs only")
        efps = glob.glob(f"{efp_dir}/test/*k_1*.feather")
    else:
        print("Using all graphs")
        # get the efps to check against the dif_order results
        efps = glob.glob(f"{efp_dir}/test/*.feather")

    # Remove previously selected efps
    for selected_efp in selected_efps:
        print(f"removing efp: {selected_efp}")
        efps.remove(f"{efp_dir}/test/{selected_efp}.feather")

    ado_df = pd.DataFrame()
    t = tqdm(efps)
    for iy, efp in enumerate(t):
        # t.set_description(f"Processing {efp}")
        # t.refresh()
        # select the dif-order subset from dif_df for the efp
        efp_label = efp.split("/")[-1].split(".feather")[0]
        efp_df = pd.read_feather(efp)

        # Use the same diff-order sig/bkg pairs to compare with ll predictions
        efp0 = efp_df.iloc[idx0]["features"].values # was "nnify"
        efp1 = efp_df.iloc[idx1]["features"].values # was "nnify"

        # Calculate the ado
        ado_val = np.mean(do_calc(fx0=ll0, fx1=ll1, gx0=efp0, gx1=efp1))
        if ado_val < 0.5:
            ado_val = 1.0 - ado_val
        dfi = pd.DataFrame({"efp": efp_label, "ado": ado_val}, index=[iy])
        ado_df = pd.concat([ado_df, dfi], axis=0)
    ado_df = ado_df.sort_values(by=["ado"], ascending=False)
    ado_df.to_csv(f"{pass_dir}/dif_order_ado_comparison.csv")


def get_max_efp(ix):
    df = pd.read_csv(f"{pass_dir}/dif_order_ado_comparison.csv", index_col=0)

    # sort by max ado
    dfs = df.sort_values(by=["ado"], ascending=False)

    # get the max graph
    efp_max = dfs.iloc[0]["efp"]
    ado_max = dfs.iloc[0]["ado"]
    print(f"Maximum dif-order graph selected: {efp_max}")
    return efp_max, ado_max


def train_nn(ix):
    pred_file = f"{pass_dir}/test_pred.feather"
    layers = 3
    nodes = 50
    batch_size = 32

    # Finvnd the "first" EFP that is most similar to the NN(LL) predictions
    # Train a simple NN with this first choice
    X_train, y_train = data_grabber(
        selected_efps=selected_efps,
        split="train",
        data_dir=data_dir, 
        efp_dir=efp_dir,
        incl_hl=incl_hl,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=False,
    )
    X_test, y_test = data_grabber(
        selected_efps=selected_efps,
        split="test",
        data_dir=data_dir, 
        efp_dir=efp_dir,
        incl_hl=incl_hl,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=False,
    )
    X_val, y_val = data_grabber(
        selected_efps=selected_efps,
        split="valid",
        data_dir=data_dir, 
        efp_dir=efp_dir,
        incl_hl=incl_hl,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=False,
    )

    # Try different network designs according to number of hidden layers and units
    model_file = f"{model_dir}/model_l_{layers}_n_{nodes}_bs_{batch_size}.h5"
    if ix == 0:
        model_file = f"{model_dir}/model_0.h5"

    if not path.exists(model_file):
        model = nn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=1000,
            batch_size=batch_size,
            layers=layers,
            nodes=nodes,
            model_file=model_file,
            verbose=2,
        )
    else:
        model = tf.keras.models.load_model(model_file)
    test_pred = np.hstack(model.predict(X_test))
    auc_val = roc_auc_score(y_test, test_pred)
    print(f"    test-set AUC: {auc_val:.5}")
    test_df = pd.DataFrame({"prediction": test_pred, "y": y_test})
    test_df.to_feather(pred_file)
    return auc_val


if __name__ == "__main__":
    chdir(path.join(home, "guided_iteration"))
    run_name = "7HL_noNNIFY"
    incl_hl = True
    incl_mass = False
    incl_pT = False
    irc_safe = False
    it_dir = path.join("results", run_name)
    ll_benchmark = 0.9720

    data_dir = path.join(home, "data") 
    efp_dir = path.join(data_dir, "efp") 
    if not path.exists(it_dir):
        mkdir(it_dir)

    # make a copy of the nn for future reference
    copy_file = f"{it_dir}/nn_copy.py"
    print("     Making a copy of nn.py")
    shutil.copyfile("nn.py", copy_file)

    # make a copy of the guided_iteration settings for future reference
    copy_file = f"{it_dir}/guided_iteration_copy.py"
    print("     Making a copy of guided_iteration.py")
    shutil.copyfile("guided_iteration.py", copy_file)

    selected_efps, aucs, ados = [], [], []

    ix, ado_max, auc_val = 0, 0, 0
    while ado_max < 1 and auc_val < ll_benchmark:
        # Define data sub-directories
        pass_dir = f"{it_dir}/p{ix}"
        model_dir = f"{pass_dir}/model_files"

        # Setting the random seed to a predictable value (in this case iteration index)
        # Will make it easier to reproduce results in the future if necessary (despite shuffling diff-order pairs)
        random.seed(ix)

        # Create a sub-directory for the pass to store all relevant data
        if not path.exists(f"{it_dir}/p{ix}"):
            mkdir(pass_dir)
            mkdir(model_dir)

        # Train a NN using current EFP selections (or just HL when ix=0)
        auc_val = train_nn(ix)
        print(f"Iteration {ix} -> AUC: {auc_val:.4}")

        # Store the auc results
        aucs.append(auc_val)
        pass_list = ["hl6"] + selected_efps
        ados_list = [np.nan] + ados
        efp_df = pd.DataFrame({"efp": pass_list, "auc": aucs, "ado": ados_list})
        efp_df.to_csv(f"{it_dir}/selected_efps.csv")

        # Isolate random dif-order pairs
        idx0, idx1 = isolate_order(ix, 6000)

        # Check ado with each EFP for most similar DO on dif-order pairs
        check_efps(ix)

        # Get the max EFP and save it
        efp_max, ado_max = get_max_efp(ix)
        selected_efps.append(efp_max)
        print(selected_efps)
        ados.append(ado_max)

        # Make plots
        viz = visualization(it_dir, ix)
        viz.dif_order_hist_plots()
        # viz.nn_design_heatmap()
        viz.performance_plot()
        viz.clear_viz()
        ix += 1
