from data_utils import (
    prepare_data,
    generate_custom_data,
    split_and_scale
)

from perceptron import Perceptron
from metrics import accuracy, print_metrics, plot_roc_curve
from utils import plot_losses, plot_decision_boundary
from experiments import run_all_experiments
from cross_validation import run_cross_validation


def choose_dataset():
    print("""
Выберите набор данных:
1 - классический make_classification
2 - линейно разделимый кастомный
3 - XOR
4 - окружность
""")

    choice = input("Ваш выбор: ")

    if choice == "1":
        return prepare_data()

    if choice == "2":
        X, y = generate_custom_data(data_type="linear")
        return split_and_scale(X, y)

    if choice == "3":
        X, y = generate_custom_data(data_type="xor")
        return split_and_scale(X, y)

    if choice == "4":
        X, y = generate_custom_data(data_type="circle")
        return split_and_scale(X, y)

    print("Некорректный выбор. Используется классический датасет.")
    return prepare_data()


def main():
    X_train, X_test, y_train, y_test, X, y = choose_dataset()

    model = Perceptron(input_size=2)

    print("\nНачинаем обучение базовой модели...")

    model.fit(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        lr=0.1,
        batch_size=32
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_prob = model.forward(X_test)

    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    plot_losses(model.train_losses, model.val_losses)
    plot_decision_boundary(model, X_train, y_train)

    print_metrics(y_test, test_pred, test_prob)
    plot_roc_curve(y_test, test_prob)

    run_all_experiments(X_train, y_train, X_test, y_test)

    run_cross_validation(X, y)


if __name__ == "__main__":
    main()