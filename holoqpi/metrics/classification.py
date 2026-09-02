"""Drug-condition classification accuracy."""

from __future__ import annotations

import numpy as np


class ClassificationMetrics:
    """Accumulates predictions for accuracy, per-class F1 and a confusion matrix."""

    def __init__(self, class_names: list[str]):
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.reset()

    def reset(self) -> None:
        self._confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predicted: np.ndarray, actual: np.ndarray) -> None:
        for p, a in zip(np.atleast_1d(predicted).ravel(), np.atleast_1d(actual).ravel()):
            self._confusion[int(a), int(p)] += 1

    def compute(self) -> dict:
        total = int(self._confusion.sum())
        if total == 0:
            return {}

        correct = int(np.trace(self._confusion))
        eps = 1e-8

        true_positive = np.diag(self._confusion).astype(float)
        predicted_total = self._confusion.sum(axis=0).astype(float)
        actual_total = self._confusion.sum(axis=1).astype(float)

        precision = true_positive / (predicted_total + eps)
        recall = true_positive / (actual_total + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        present = actual_total > 0
        results = {
            "cls_accuracy": correct / total,
            "cls_macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "cls_balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
            "cls_n_images": total,
        }
        for index, name in enumerate(self.class_names):
            if actual_total[index] > 0:
                results[f"cls_f1_{name}"] = float(f1[index])
        return results

    @property
    def confusion_matrix(self) -> dict:
        return {
            "labels": self.class_names,
            "matrix": self._confusion.tolist(),
        }
