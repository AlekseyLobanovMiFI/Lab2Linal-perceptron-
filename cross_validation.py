import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from perceptron import Perceptron
from metrics import accuracy


def run_cross_validation(X, y):
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    batch_sizes = [16, 32, 64]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n5-кратная кросс-валидация")

    best_score = -1
    best_params = None

    for lr in learning_rates:
        for batch_size in batch_sizes:
            scores = []

            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                model = Perceptron(input_size=2)

                model.fit(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    epochs=100,
                    lr=lr,
                    batch_size=batch_size,
                    verbose=False
                )

                y_pred = model.predict(X_val)
                score = accuracy(y_val, y_pred)

                scores.append(score)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            print(
                f"lr={lr:<5} batch={batch_size:<3} | "
                f"mean={mean_score:.4f}, std={std_score:.4f}"
            )

            if mean_score > best_score:
                best_score = mean_score
                best_params = (lr, batch_size)

    print(
        f"\nЛучшие параметры: lr={best_params[0]}, "
        f"batch_size={best_params[1]}, accuracy={best_score:.4f}"
    )

    return best_params