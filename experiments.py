from perceptron import Perceptron
from metrics import accuracy


def run_learning_rate_experiment(X_train, y_train, X_test, y_test):
    learning_rates = [0.001, 0.01, 0.5, 1.0]

    print("\nЭксперимент: влияние learning rate")

    results = []

    for lr in learning_rates:
        model = Perceptron(input_size=2)

        model.fit(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=100,
            lr=lr,
            batch_size=32,
            verbose=False
        )

        test_pred = model.predict(X_test)
        test_acc = accuracy(y_test, test_pred)

        results.append((lr, test_acc))

        print(f"lr = {lr:<6} | test accuracy = {test_acc:.4f}")

    return results


def run_batch_size_experiment(X_train, y_train, X_test, y_test):
    batch_sizes = [1, 16, 64, 256]

    print("\nЭксперимент: влияние batch size")

    results = []

    for batch_size in batch_sizes:
        model = Perceptron(input_size=2)

        model.fit(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=100,
            lr=0.1,
            batch_size=batch_size,
            verbose=False
        )

        test_pred = model.predict(X_test)
        test_acc = accuracy(y_test, test_pred)

        results.append((batch_size, test_acc))

        print(f"batch_size = {batch_size:<4} | test accuracy = {test_acc:.4f}")

    return results


def run_initialization_experiment(X_train, y_train, X_test, y_test):
    init_types = ["zeros", "small_random", "large_random"]

    print("\nЭксперимент: влияние инициализации весов")

    results = []

    for init_type in init_types:
        model = Perceptron(input_size=2, init_type=init_type)

        model.fit(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=100,
            lr=0.1,
            batch_size=32,
            verbose=False
        )

        test_pred = model.predict(X_test)
        test_acc = accuracy(y_test, test_pred)

        results.append((init_type, test_acc))

        print(f"init = {init_type:<13} | test accuracy = {test_acc:.4f}")

    return results


def run_all_experiments(X_train, y_train, X_test, y_test):
    run_learning_rate_experiment(X_train, y_train, X_test, y_test)
    run_batch_size_experiment(X_train, y_train, X_test, y_test)
    run_initialization_experiment(X_train, y_train, X_test, y_test)