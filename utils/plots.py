import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_confusion_matrix(cm, class_labels, title):
    plt.figure(figsize=(5, 5))

    plt.imshow(cm)
    plt.colorbar()

    plt.xticks(list(range(len(class_labels))), ['Predicted '+ str(label) for label in class_labels]) 
    plt.yticks(list(range(len(class_labels))), ['True '+ str(label) for label in class_labels])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title(title)
    plt.show()

def plot_difference_loss(scratch_loss, torch_loss):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    loss_diff = np.array(scratch_loss) - np.array(torch_loss)

    axes[0].plot(scratch_loss, label="From scratch")
    axes[0].plot(torch_loss, label="PyTorch")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training loss curves")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(scratch_loss, label="From scratch")
    axes[1].plot(torch_loss, label="PyTorch")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Training loss curves - log scale")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(loss_diff)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Scratch loss - PyTorch loss")
    axes[2].set_title("Difference between training loss curves")
    axes[2].axhline(0, linestyle="--")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolor="k",
        alpha=0.7
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)

    plt.show()

def animate_mlp_decision_boundary(model, snapshots, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    snapshot_keys = list(snapshots.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    original_params = {
        "W1": model.W1.copy(),
        "b1": model.b1.copy(),
        "W2": model.W2.copy(),
        "b2": model.b2.copy(),
    }

    def update(frame_idx):
        ax.clear()

        iteration = snapshot_keys[frame_idx]
        params = snapshots[iteration]

        model.W1 = params["W1"]
        model.b1 = params["b1"]
        model.W2 = params["W2"]
        model.b2 = params["b2"]

        Z = model.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3)

        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            edgecolor="k",
            alpha=0.7,
        )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"MLP decision boundary - iteration {iteration}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(snapshot_keys),
        interval=150,
        repeat=True,
    )

    plt.close(fig)

    model.W1 = original_params["W1"]
    model.b1 = original_params["b1"]
    model.W2 = original_params["W2"]
    model.b2 = original_params["b2"]

    return anim