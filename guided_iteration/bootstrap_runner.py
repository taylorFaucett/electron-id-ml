import os
from nn import nn
from guided_iteration import data_grabber
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

# Suppress all the tensorflow debugging info for new networks
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def grab_and_mix_data(selected_efps):
    X_train, y_train = data_grabber(
        selected_efps=selected_efps,
        split="train",
        data_dir=data_dir,
        efp_dir=efp_dir,
        incl_hl=True,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=False,
    )
    X_test, y_test = data_grabber(
        selected_efps=selected_efps,
        split="test",
        data_dir=data_dir,
        efp_dir=efp_dir,
        incl_hl=True,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=False,
    )
    X_val, y_val = data_grabber(
        selected_efps=selected_efps,
        split="valid",
        data_dir=data_dir,
        efp_dir=efp_dir,
        incl_hl=True,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=False,
    )

    X = np.vstack(([X_train, X_test, X_val]))
    y = np.hstack(([y_train, y_test, y_val]))
    return X, y


def bootstrap_runner(run_name):
    selected_efps = pd.read_csv(
        os.path.join(home, "guided_iteration", "results", run_name, "selected_efps.csv"), index_col=0
    )
    selected_efps = selected_efps.efp.tolist()[1:3]
    X, y = grab_and_mix_data(selected_efps)

    n = len(y)
    n_train = int(0.85 * n)
    n_test = int(0.15 * n)
    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    rs.get_n_splits(X)

    ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    straps = []
    aucs = []
    bs_count = 0
    for train_index, test_index in rs.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]
        model_file = f"{bs_model_dir}/bs-{bs_count}.h5"
        if not os.path.isfile(model_file):
            model = nn(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                layers=layers,
                nodes=nodes,
                model_file=model_file,
                verbose=2,
            )
        else:
            model = tf.keras.models.load_model(model_file)

        auc_val = roc_auc_score(y_val, np.hstack(model.predict(X_val)))
        # print(f"    test-set AUC: {auc_val:.5}")
        straps.append(bs_count)
        aucs.append(auc_val)
        results = pd.DataFrame({"bs": straps, "auc": aucs})
        results.to_csv(os.path.join(home, "guided_iteration", "results", run_name, "bootstrap_results.csv"))
        bs_count += 1
        auc_mean = np.average(aucs)
        auc_std = np.std(aucs)
        print(f"AUC = {auc_mean:.5f} +/- {auc_std:.5f}")


if __name__ == "__main__":
    # Set directory references
    home = os.getcwd()
    data_dir = os.path.join(home, "data")
    efp_dir = os.path.join(data_dir, "efp")
    n_splits = 50  # number of bootstrap passes

    # Run information
    run_names = ["7HL", "7HL_mass", "7HL_ircSafe", "7HL_mass_ircSafe"]
    for run_name in run_names:
        if "mass" in run_name:
            incl_mass = True
        else:
            incl_mass = False
        # Include pT or not
        incl_pT = False
        # Create a directory for boot-strap models if it doesn't exist
        bs_model_dir = os.path.join(home, "guided_iteration", "results", run_name, "bs_models")
        if not os.path.exists(bs_model_dir):
            os.mkdir(bs_model_dir)

        # NN settings
        layers = 3
        nodes = 50
        batch_size = 64
        epochs = 250

        # Run bootstrap
        bootstrap_runner(run_name)
