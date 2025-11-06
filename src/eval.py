import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.data import load_fashion_mnist, split_70_15_15, make_datasets, CLASS_NAMES

def plot_curves(history_dict, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    # Accuracy
    plt.figure()
    plt.plot(history_dict["accuracy"])
    plt.plot(history_dict["val_accuracy"])
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train","val"], loc="best")
    plt.savefig(os.path.join(outdir, "acc_curve.png"), bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history_dict["loss"])
    plt.plot(history_dict["val_loss"])
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train","val"], loc="best")
    plt.savefig(os.path.join(outdir, "loss_curve.png"), bbox_inches="tight")
    plt.close()

def main():
    # Recreate datasets with the same split
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = split_70_15_15(
        (x_train.astype("float32")/255.0)[...,None], y_train,
        (x_test.astype("float32")/255.0)[...,None], y_test, seed=42)
    ds_tr, ds_val, ds_te = make_datasets(x_tr, y_tr, x_val, y_val, x_te, y_te, batch=256)

    model = tf.keras.models.load_model("models/model.h5")

    # Predictions on test set
    y_true = []
    y_pred = []
    for xb, yb in ds_te:
        logits = model.predict(xb, verbose=0)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(np.argmax(logits, axis=1).tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

        # Metrics
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Save metrics
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.save("outputs/confusion_matrix.npy", cm)

    # Save confusion matrix as PNG
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300)
    plt.close()


    # Save pretty text report
    from sklearn.metrics import classification_report as cr_text
    with open("outputs/classification_report.txt", "w") as f:
        f.write(cr_text(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Plot curves if history exists
    hist_path = "outputs/history.pkl"
    if os.path.exists(hist_path):
        import pickle
        with open(hist_path, "rb") as f:
            history = pickle.load(f)
        plot_curves(history, outdir="outputs")

    # Quick preview image predictions
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(9):
        img = x_te[i]
        lbl = CLASS_NAMES[y_te[i]]
        pred = CLASS_NAMES[y_pred[i]]
        plt.subplot(3,3,i+1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"T:{lbl}\nP:{pred}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/sample_predictions.png", bbox_inches="tight")
    plt.close()

    print("Saved metrics and plots in outputs/")

if __name__ == "__main__":
    main()
