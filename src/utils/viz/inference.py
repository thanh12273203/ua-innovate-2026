from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class LifecycleLabels:
    class_order: Tuple[int, int, int] = (-1, 0, 1)
    class_names: Tuple[str, str, str] = ("passed EoL", "EoL > 365", "EoL <= 365")


def plot_confusion_and_roc(
    model,
    X_test,
    y_test: np.ndarray,
    labels: Optional[LifecycleLabels] = None,
    normalize_confusion: bool = False,
    roc_multi_class: str = 'ovr',
    savefig: Optional[str] = None,
) -> None:
    """
    Plot a 3x3 confusion matrix (left) and a macro-average ROC curve (right)
    for a fitted sklearn multiclass classifier.

    Parameters
    ----------
    model : object
        Fitted sklearn classifier implementing predict() and predict_proba().
        Examples: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier.
    X_test : array-like
        Test features.
    y_test : numpy.ndarray
        True labels with values in {-1, 0, 1}.
    labels : LifecycleLabels, optional
        Class order and display names. Defaults to:
        {-1: "passed EoL", 0: "EoL > 365", 1: "EoL <= 365"}.
    normalize_confusion : bool
        If True, normalize confusion matrix rows to sum to 1.
    roc_multi_class : {"ovr", "ovo"}
        Multi-class ROC AUC scheme for roc_auc_score.
    savefig : str, optional
        If provided, save the figure to this path instead of showing.
    """
    if labels is None:
        labels = LifecycleLabels()

    class_order = list(labels.class_order)
    class_names = list(labels.class_names)

    y_test = np.asarray(y_test)
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must implement predict_proba() to plot ROC curves.")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    if hasattr(model, "classes_"):
        model_classes = np.asarray(model.classes_)
        missing_classes = [cls for cls in class_order if cls not in model_classes]
        if missing_classes:
            raise ValueError(
                f"Model probabilities are missing expected classes: {missing_classes}. "
                f"Model classes: {model_classes.tolist()}"
            )
        class_indices = [int(np.where(model_classes == cls)[0][0]) for cls in class_order]
        y_pred_proba = y_pred_proba[:, class_indices]

    # Confusion matrix (in the specified order)
    confusion = confusion_matrix(y_test, y_pred, labels=class_order)
    if normalize_confusion:
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums != 0)

    # ROC (binarize labels to shape (n_samples, n_classes) following class_order)
    y_test_binarized = label_binarize(y_test, classes=class_order)

    # Macro AUC
    macro_auc = roc_auc_score(
        y_test_binarized,
        y_pred_proba,
        average="macro",
        multi_class=roc_multi_class,
    )

    # Per-class ROC curves
    false_positive_rate_by_class: dict[int, np.ndarray] = {}
    true_positive_rate_by_class: dict[int, np.ndarray] = {}

    for class_index in range(len(class_order)):
        y_true_binary = y_test_binarized[:, class_index]
        y_score = y_pred_proba[:, class_index]
        false_positive_rate_by_class[class_index], true_positive_rate_by_class[class_index], _ = roc_curve(
            y_true_binary, y_score
        )

    all_false_positive_rates = np.unique(
        np.concatenate([false_positive_rate_by_class[i] for i in range(len(class_order))])
    )
    mean_true_positive_rate = np.zeros_like(all_false_positive_rates)
    for class_index in range(len(class_order)):
        mean_true_positive_rate += np.interp(
            all_false_positive_rates,
            false_positive_rate_by_class[class_index],
            true_positive_rate_by_class[class_index],
        )

    mean_true_positive_rate /= len(class_order)

    # Plot: 1 row x 2 columns (CM left, ROC right)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Confusion matrix plot ---
    axis_cm = axes[0]
    image = axis_cm.imshow(confusion, interpolation='nearest', cmap='coolwarm')
    axis_cm.set_title("Confusion Matrix" + (" (row-normalized)" if normalize_confusion else ""))
    axis_cm.set_xlabel("Predicted")
    axis_cm.set_ylabel("Actual")
    axis_cm.set_xticks(np.arange(len(class_names)))
    axis_cm.set_yticks(np.arange(len(class_names)))
    axis_cm.set_xticklabels(class_names, rotation=30, ha='right')
    axis_cm.set_yticklabels(class_names)

    # Annotate cells
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            value = confusion[row_index, col_index]
            text_value = f"{value:.2f}" if normalize_confusion else f"{int(value)}"
            axis_cm.text(col_index, row_index, text_value, ha='center', va='center')

    figure.colorbar(image, ax=axis_cm, fraction=0.046, pad=0.04)

    # --- ROC plot ---
    axis_roc = axes[1]
    axis_roc.plot(all_false_positive_rates, mean_true_positive_rate, label=f"Macro AUC ({roc_multi_class}) = {macro_auc:.3f}")
    axis_roc.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    axis_roc.set_xlim(0.0, 1.0)
    axis_roc.set_ylim(0.0, 1.05)
    axis_roc.set_xlabel("False Positive Rate")
    axis_roc.set_ylabel("True Positive Rate")
    axis_roc.set_title("Macro-Average ROC Curve")
    axis_roc.legend(loc='lower right')

    figure.tight_layout()

    if savefig is not None:
        figure.savefig(savefig, dpi=300, bbox_inches='tight')
        plt.close(figure)
    else:
        plt.show()
