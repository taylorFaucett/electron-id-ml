#! /home/tfaucett/miniconda3/envs/tf/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
from nn import nn
import h5py
import itertools
import random
from sklearn import preprocessing
from average_decision_ordering import ADO
import shutil
import sys
from sklearn.metrics import roc_auc_score
import warnings
import tensorflow as tf
from viz_plots import viz

warnings.simplefilter(action="ignore", category=FutureWarning)

# Suppress all the tensorflow debugging info for new networks
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

scaler = preprocessing.MinMaxScaler()


def data_grabber(selected_efps, split, normalize=False):
    hl_file = h5py.File(f"{data_dir}/data/data.h5", "r")
    hl = hl_file["hl"][split][:]
    y = hl_file["y"][split][:]
    df = pd.DataFrame(hl)
    for efp in selected_efps:
        efp_file = f"{data_dir}/efp_data/{split}/{efp}.csv"
        dfi = pd.read_csv(efp_file, compression="gzip",
                          index_col=0)["features"]
        dfi = pd.DataFrame({efp: dfi.values})
        df = pd.concat([df, dfi], axis=1)
    if normalize:
        for column in df:
            df[column] = scaler.fit_transform(df[column].values.reshape(-1,1))
    return scaler.fit_transform(df), y


def train_nn(selected_efps):
    # Find the "first" EFP that is most similar to the NN(LL) predictions
    # Train a simple NN with this first choice
    X_train, y_train = data_grabber(selected_efps=selected_efps, split="train", normalize=True)
    X_test, y_test = data_grabber(selected_efps=selected_efps, split="test", normalize=True)
    X_val, y_val = data_grabber(selected_efps=selected_efps, split="valid", normalize=True)
    

    # Try different network designs according to number of hidden layers and units
    model_file = f"{model_dir}/model_l_{layers}_n_{nodes}.h5"
    model_file_name = model_file.split("/")[-1]
    model = nn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=1000,
        batch_size=32,
        layers=layers,
        nodes=nodes,
        ix=ix,
        model_file=model_file,
        verbose=0,
    )
    test_pred = np.hstack(model.predict(X_test))
    auc_val = roc_auc_score(y_test, test_pred)
    print(f"    AUC: {auc_val:.4}")


if __name__ == "__main__":
    efps = glob.glob("../efp_data/train/*.csv")
    selected_efps = random.choices(efps)
    print(selected_efps)
    exit()
    train_nn(selected_efps)
    
