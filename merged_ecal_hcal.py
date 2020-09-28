#! /home/tfaucett/miniconda3/envs/tf/bin/python

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd
import energyflow as ef
import glob


def nn(X_train, y_train, X_val, y_val):
    model = tf.keras.Sequential()
    nodes = 25
    layers = 3
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    model.add(tf.keras.layers.Dense(nodes, input_dim=X_train.shape[-1]))
    for _ in range(layers):
        model.add(
            tf.keras.layers.Dense(
                nodes,
                kernel_initializer="normal",
                activation="relu",
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
            )
        )
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"],
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="auto", verbose=0, patience=5
    )

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=500,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=[es],
    )
    return model


def merged_ecal_hcal():
    scaler = preprocessing.RobustScaler()

    # Calculate HL variables from ET, eta, phi
    efps = glob.glob("data/efp/test/et_*.feather")
    efps = [os.path.basename(x).split("et_")[-1].split(".feather")[0] for x in efps]
    t = tqdm(efps)
    for efp in t:
        efp_file = f"data/efp_merged/test/{efp}.feather"
        if not os.path.exists(efp_file):
            h_train = pd.read_feather(f"data/efp/train/ht_{efp}.feather")
            e_train = pd.read_feather(f"data/efp/train/et_{efp}.feather")
            h_test = pd.read_feather(f"data/efp/test/ht_{efp}.feather")
            e_test = pd.read_feather(f"data/efp/test/et_{efp}.feather")
            h_val = pd.read_feather(f"data/efp/valid/ht_{efp}.feather")
            e_val = pd.read_feather(f"data/efp/valid/et_{efp}.feather")

            X_train = scaler.fit_transform(
                np.vstack((e_train.features.values, h_train.features.values)).T
            )
            y_train = h_train.targets.values

            X_test = scaler.fit_transform(
                np.vstack((e_test.features.values, h_test.features.values)).T
            )
            y_test = h_test.targets.values

            X_val = scaler.fit_transform(
                np.vstack((e_val.features.values, h_val.features.values)).T
            )
            y_val = h_val.targets.values

            model = nn(X_train, y_train, X_val, y_val)
            train_out = np.hstack(model.predict(X_train))
            test_out = np.hstack(model.predict(X_test))
            val_out = np.hstack(model.predict(X_val))

            train_df = pd.DataFrame(
                {"features": train_out, "nnify": train_out, "targets": y_train}
            )
            test_df = pd.DataFrame(
                {"features": test_out, "nnify": test_out, "targets": y_test}
            )
            val_df = pd.DataFrame(
                {"features": val_out, "nnify": val_out, "targets": y_val}
            )
            
            auc_val = roc_auc_score(y_test, test_out)
            train_df.to_feather(f"data/efp_merged/train/{efp}.feather")
            val_df.to_feather(f"data/efp_merged/valid/{efp}.feather")
            test_df.to_feather(f"data/efp_merged/test/{efp}.feather")
            print(f"Last Finished: [AUC = {auc_val:.4}] - {efp}")


if __name__ == "__main__":
    merged_ecal_hcal()
