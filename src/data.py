
import tensorflow as tf

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Normalize to [0,1] and add channel dim
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)

def split_70_15_15(x_train, y_train, x_test, y_test, seed=42):
    # Combined total = 70k -> target splits: 49k/10.5k/10.5k
    total = x_train.shape[0] + x_test.shape[0]
    assert total == 70000, "Fashion-MNIST should have 70k images total."
    # We'll re-concatenate and shuffle to get exact 70/15/15
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    n_train = int(0.70 * len(x))
    n_val   = int(0.15 * len(x))
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train+n_val], y[n_train:n_train+n_val]
    x_te, y_te = x[n_train+n_val:], y[n_train+n_val:]
    return (x_tr, y_tr), (x_val, y_val), (x_te, y_te)

def make_datasets(x_tr, y_tr, x_val, y_val, x_te, y_te, batch=64, shuffle=2048):
    ds_tr = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(shuffle).batch(batch).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch).prefetch(tf.data.AUTOTUNE)
    ds_te = tf.data.Dataset.from_tensor_slices((x_te, y_te)).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds_tr, ds_val, ds_te

CLASS_NAMES = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]
