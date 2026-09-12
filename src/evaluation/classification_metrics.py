from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_proba_alfabetizado,
) -> dict[str, Any]:
    """
    Calcula métricas para a classificação de alfabetização.

    Convenção original do target:
        0 = não alfabetizado
        1 = alfabetizado

    Apesar de a classe 1 permanecer como classe probabilística padrão
    do modelo, também calculamos métricas específicas para a classe 0,
    que representa o risco educacional de maior interesse para o projeto.
    """

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    y_risco = (np.asarray(y_true) == 0).astype(int)
    proba_risco = 1.0 - np.asarray(y_proba_alfabetizado)

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc_alfabetizado": float(
            roc_auc_score(
                y_true,
                y_proba_alfabetizado,
            )
        ),
        "pr_auc_nao_alfabetizado": float(
            average_precision_score(
                y_risco,
                proba_risco,
            )
        ),
        "classe_0_nao_alfabetizado": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "classe_1_alfabetizado": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "confusion_matrix_labels": [
            0,
            1,
        ],
        "confusion_matrix": cm.tolist(),
    }