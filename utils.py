import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses):

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Loss during training")

    plt.legend()
    plt.grid()

    plt.show()


def plot_decision_boundary(model, X, y):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="k"
    )

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x_values = np.linspace(x_min, x_max, 100)

    if model.w[1] != 0:

        y_values = -(
            model.w[0] * x_values + model.b
        ) / model.w[1]

        plt.plot(
            x_values,
            y_values,
            label="Decision Boundary"
        )

    else:

        x_boundary = -model.b / model.w[0]

        plt.axvline(
            x=x_boundary,
            label="Decision Boundary"
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.title("Decision Boundary")

    plt.legend()
    plt.grid()

    plt.show()