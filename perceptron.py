import numpy as np


class Perceptron:

    def __init__(self, input_size=2, init_type="small_random"):
        self.input_size = input_size
        self.init_type = init_type

        self.w = self.initialize_weights()
        self.b = 0.0

        self.train_losses = []
        self.val_losses = []

    def initialize_weights(self):
        if self.init_type == "zeros":
            return np.zeros(self.input_size)

        if self.init_type == "small_random":
            return np.random.randn(self.input_size) * 0.01

        if self.init_type == "large_random":
            return np.random.randn(self.input_size) * 10

        raise ValueError("Unknown initialization type")

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        lr=0.1,
        batch_size=32,
        verbose=True
    ):

        self.train_losses = []
        self.val_losses = []

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)

                error = y_pred - y_batch

                dw = np.dot(X_batch.T, error) / len(X_batch)
                db = np.mean(error)

                self.w -= lr * dw
                self.b -= lr * db

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)

            train_loss = self.compute_loss(y_train, train_pred)
            val_loss = self.compute_loss(y_val, val_pred)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

        return self.train_losses, self.val_losses

    def predict(self, X):
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)