import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )

    return split_and_scale(X, y)


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X, y


def generate_custom_data(data_type="linear", n_samples=500, noise=0.05):
    np.random.seed(42)

    if data_type == "linear":
        class_0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
        class_1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])

        X = np.vstack([class_0, class_1])
        y = np.hstack([
            np.zeros(n_samples // 2),
            np.ones(n_samples // 2)
        ])

    elif data_type == "xor":
        X = np.random.uniform(-2, 2, size=(n_samples, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    elif data_type == "circle":
        X = np.random.uniform(-2, 2, size=(n_samples, 2))
        radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        y = (radius > 1).astype(int)

    else:
        raise ValueError("Unknown data type")

    # шум: случайно меняем часть меток
    n_noise = int(noise * n_samples)
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    y[noise_indices] = 1 - y[noise_indices]

    return X, y.astype(int)