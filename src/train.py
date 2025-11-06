# src/train.py

import os as _os
import json
import pickle
import tensorflow as tf
from src.data import load_fashion_mnist, split_70_15_15, make_datasets, CLASS_NAMES
from src.model import build_cnn

tf.random.set_seed(42)

EPOCHS = 30
BATCH = 64
LR = 1e-3


def main():
    # 1) Load + split 70/15/15
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = split_70_15_15(
        x_train, y_train, x_test, y_test, seed=42
    )

    # 2) Build tf.data pipelines
    ds_tr, ds_val, ds_te = make_datasets(
        x_tr, y_tr, x_val, y_val, x_te, y_te, batch=BATCH
    )

    # 3) Build & compile CNN
    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4) Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_loss"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5),
    ]

    # 5) Train
    history = model.fit(ds_tr, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks)

    # 6) Save model + labels
    _os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    with open("models/class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f)

    # 7) Save training history
    _os.makedirs("outputs", exist_ok=True)
    with open("outputs/history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # 8) Evaluate on val & test
    val_loss, val_acc = model.evaluate(ds_val, verbose=0)
    test_loss, test_acc = model.evaluate(ds_te, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
