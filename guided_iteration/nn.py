import glob
import os

# Suppress all the tensorflow debugging info for new networks
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    if epoch < 10:
        return 0.005
    else:
        return 0.005 * tf.math.exp(0.1 * (10 - epoch))


def nn(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    layers,
    nodes,
    model_file,
    verbose,
):

    print("    Training a new model at: " + model_file)
    model = tf.keras.Sequential()

    #     optimizer = tf.keras.optimizers.SGD(
    #         learning_rate=0.5, momentum=0.0, nesterov=False, name="SGD"
    #     )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    initializer = tf.keras.initializers.GlorotUniform()
    model.add(tf.keras.layers.Flatten(input_dim=X_train.shape[1]))
    for lix in range(layers):
        model.add(
            tf.keras.layers.Dense(
                nodes,
                kernel_initializer=initializer,
                activation="relu",
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
                bias_constraint=tf.keras.constraints.MaxNorm(3),
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4),
                activity_regularizer=tf.keras.regularizers.l2(1e-5),
            )
        )
    #         if lix <= layers - 2:
    #             model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Accuracy(name="acc"),
        ],
    )
    mc = tf.keras.callbacks.ModelCheckpoint(
        model_file, monitor="val_auc", verbose=verbose, save_best_only=True, mode="max",
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", verbose=verbose, patience=25
    )

    lrs = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=verbose)

    callbacks = [mc, es]

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
