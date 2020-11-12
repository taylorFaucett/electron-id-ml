from guided_iteration import data_grabber
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

def scheduler(epoch):
    if epoch < 10:
        return 0.01
    else:
        return 0.01 * tf.math.exp(0.1 * (10 - epoch))


def nn(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    layers,
    nodes,
    ix,
    model_file,
    verbose,
):

    # print("    Training a new model at: " + model_file)
    model = tf.keras.Sequential()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    model.add(tf.keras.layers.Flatten(input_dim=X_train.shape[1]))
    for lix in range(layers):
        model.add(
            tf.keras.layers.Dense(
                nodes,
                kernel_initializer="normal",
                activation="relu",
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
                bias_constraint=tf.keras.constraints.MaxNorm(3),
            )
        )
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",  # binary_crossentropy
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Accuracy(name="acc"),
        ],
    )
    mc = tf.keras.callbacks.ModelCheckpoint(
        model_file,
        monitor="val_auc",
        verbose=verbose,
        save_best_only=True,
        mode="max",
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", verbose=verbose, patience=10
    )

    lrs = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=verbose)

    callbacks = [mc, es, lrs]

    if verbose > 0:
        print(model.summary())

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    return model


def k_fold_nn(run_name, n_folds):    
    model_dir = f"{it_dir}/k_models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Get selected efps
    selected_efps = pd.read_csv(f"{it_dir}/selected_efps.csv").efp.values[1:]
    X_train, y_train = data_grabber(
        selected_efps=selected_efps,
        split="train",
        data_dir=data_dir, 
        efp_dir=efp_dir,
        incl_hl=incl_hl,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=True,
    )
    X_test, y_test = data_grabber(
        selected_efps=selected_efps,
        split="test",
        data_dir=data_dir, 
        efp_dir=efp_dir,
        incl_hl=incl_hl,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=True,
    )
    X_val, y_val = data_grabber(
        selected_efps=selected_efps,
        split="valid",
        data_dir=data_dir, 
        efp_dir=efp_dir,
        incl_hl=incl_hl,
        incl_mass=incl_mass,
        incl_pT=incl_pT,
        normalize=True,
    )
    X = np.vstack((X_train, X_test, X_val))
    y = np.concatenate((y_train, y_test, y_val))
    kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)
    kf.get_n_splits(X)
    kix = 0
    aucs = np.zeros(n_folds)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        model_file = f"{model_dir}/k{kix}.h5"
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if not os.path.isfile(model_file):
            model = nn(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=epochs,
                batch_size=batch_size,
                layers=layers,
                nodes=nodes,
                ix=kix,
                model_file=model_file,
                verbose=2,
            )
        else:
            model = tf.keras.models.load_model(model_file)
        yhat = np.hstack(model.predict(X_test))
        auc_val = roc_auc_score(y_test, yhat)
        aucs[kix] = auc_val
        print(f"pass {kix} -> test-set AUC={auc_val:.4}")
        results = pd.DataFrame({"auc": aucs})
        results.to_csv(f"{it_dir}/k_fold_results.csv")
        kix += 1
    print(aucs)
    auc_avg = np.average(aucs)
    auc_std = np.std(aucs)
    print(f"AUC = {auc_avg} ± {auc_std}")

if __name__ == "__main__":
    # Define directories
    run_name = "b32_lrs_mass_std"
    home = "/home/tfaucett/Projects/electron-id"
    data_dir = f"{home}/data"
    efp_dir = f"{data_dir}/efp"
    it_dir = f"results/{run_name}"
    
    # set HL selections
    incl_hl = True
    incl_mass = True
    incl_pT = False
    
    # Run details
    n_folds = 5
    lr_init = 0.01
    batch_size = 32
    nodes = 50
    layers = 3
    dropout = 0.25
    epochs = 1000
    k_fold_nn(run_name, n_folds)