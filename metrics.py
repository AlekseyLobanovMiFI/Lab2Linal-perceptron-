import numpy as np
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, tn, fp, fn


def precision_score(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fp + 1e-8)


def recall_score(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fn + 1e-8)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return 2 * precision * recall / (precision + recall + 1e-8)


def roc_curve_manual(y_true, y_prob):
    thresholds = np.linspace(0, 1, 100)

    tpr_values = []
    fpr_values = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return np.array(fpr_values), np.array(tpr_values)


def roc_auc_score_manual(y_true, y_prob):
    fpr, tpr = roc_curve_manual(y_true, y_prob)

    sorted_indices = np.argsort(fpr)

    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    return np.trapezoid(tpr, fpr)


def print_metrics(y_true, y_pred, y_prob):
    print("\nМетрики качества:")
    print(f"Accuracy:  {accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score_manual(y_true, y_prob):.4f}")


def plot_roc_curve(y_true, y_prob):
    fpr, tpr = roc_curve_manual(y_true, y_prob)
    auc = roc_auc_score_manual(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC curve, AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend()
    plt.grid()
    plt.show()