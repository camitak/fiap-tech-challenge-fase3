from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.evaluation.classification_metrics import (
    calculate_classification_metrics,
)
from src.preprocessing.data_loader import (
    load_modeling_data,
)
from src.preprocessing.preprocessor import (
    BASELINE_FEATURES,
    build_preprocessor,
)


RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "logistic_baseline_2023.json"
)


def main() -> None:
    print(
        "Carregando exclusivamente a base de desenvolvimento de 2023..."
    )

    df = load_modeling_data(2023)

    X = df[BASELINE_FEATURES]
    y = df["alfabetizado"].astype(int)
    groups = df["id_municipio"].astype(str)

    print(f"Registros: {len(df):,}")
    print(
        f"Municípios: {groups.nunique():,}"
    )

    # --------------------------------------------------------------
    # Split territorial
    # --------------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    groups_train = groups.iloc[train_idx]
    groups_val = groups.iloc[val_idx]

    train_municipios = set(
        groups_train.unique()
    )
    val_municipios = set(
        groups_val.unique()
    )

    overlap = (
        train_municipios
        & val_municipios
    )

    assert not overlap, (
        "Há municípios presentes simultaneamente "
        "em treino e validação."
    )

    print("\n--- Split de desenvolvimento ---")

    print(
        f"Treino: {len(X_train):,} registros"
    )
    print(
        f"Validação: {len(X_val):,} registros"
    )

    print(
        "Municípios treino:",
        len(train_municipios),
    )

    print(
        "Municípios validação:",
        len(val_municipios),
    )

    print(
        "Sobreposição de municípios:",
        len(overlap),
    )

    train_rate = y_train.mean()
    val_rate = y_val.mean()

    majority_class = int(
        y_train.mode().iloc[0]
    )

    majority_accuracy = float(
        (y_val == majority_class).mean()
    )

    print(
        f"Taxa alfabetizados treino: "
        f"{train_rate:.4%}"
    )

    print(
        f"Taxa alfabetizados validação: "
        f"{val_rate:.4%}"
    )

    # --------------------------------------------------------------
    # Pipeline baseline
    # --------------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    include_uf=False
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=300,
                ),
            ),
        ]
    )

    print(
        "\nTreinando regressão logística baseline..."
    )

    start = perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        perf_counter() - start
    )

    print(
        f"Treinamento concluído em "
        f"{training_seconds:.2f} segundos."
    )

    # --------------------------------------------------------------
    # Validação
    # --------------------------------------------------------------

    print(
        "\nAvaliando municípios de validação..."
    )

    y_pred = model.predict(X_val)

    predicted_non_literate_rate = float(
        (y_pred == 0).mean()
    )

    actual_non_literate_rate = float(
        (y_val == 0).mean()
    )

    y_proba = model.predict_proba(
        X_val
    )[:, 1]

    metrics = (
        calculate_classification_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_proba_alfabetizado=y_proba,
        )
    )

    feature_names = (
        model.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    classifier = (
        model.named_steps["classifier"]
    )

    result = {
        "modelo": (
            "LogisticRegression baseline"
        ),
        "ano_desenvolvimento": 2023,
        "ano_teste_final": 2024,
        "random_state": RANDOM_STATE,
        "validation_size_groups": (
            VALIDATION_SIZE
        ),
        "features_baseline": (
            BASELINE_FEATURES
        ),
        "features_transformadas": (
            feature_names.tolist()
        ),
        "split": {
            "registros_treino": int(
                len(X_train)
            ),
            "registros_validacao": int(
                len(X_val)
            ),
            "municipios_treino": int(
                len(train_municipios)
            ),
            "municipios_validacao": int(
                len(val_municipios)
            ),
            "municipios_overlap": int(
                len(overlap)
            ),
            "taxa_alfabetizados_treino": (
                float(train_rate)
            ),
            "taxa_alfabetizados_validacao": (
                float(val_rate)
            ),
        },
        "treinamento_segundos": float(
            training_seconds
        ),
        "convergencia_iteracoes": (
            classifier.n_iter_.tolist()
        ),
        "metricas_validacao": metrics,
        "baseline_majoritario": {
            "classe_majoritaria_treino": majority_class,
            "accuracy_validacao": majority_accuracy,
        },
        "distribuicao_predicoes": {
            "taxa_real_nao_alfabetizado": (
                actual_non_literate_rate
            ),
            "taxa_predita_nao_alfabetizado": (
                predicted_non_literate_rate
            ),
        },
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Accuracy baseline majoritário: "
        f"{majority_accuracy:.4f}"
    )

    print(
        f"Taxa real não alfabetizado: "
        f"{actual_non_literate_rate:.4%}"
    )

    print(
        f"Taxa predita não alfabetizado: "
        f"{predicted_non_literate_rate:.4%}"
    )

    print("\n--- Métricas ---")

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc_alfabetizado']:.4f}"
    )

    risk = (
        metrics[
            "classe_0_nao_alfabetizado"
        ]
    )

    print(
        f"Precision - não alfabetizado: "
        f"{risk['precision']:.4f}"
    )

    print(
        f"Recall - não alfabetizado: "
        f"{risk['recall']:.4f}"
    )

    print(
        f"F1 - não alfabetizado: "
        f"{risk['f1']:.4f}"
    )

    print(
        f"PR-AUC - não alfabetizado: "
        f"{metrics['pr_auc_nao_alfabetizado']:.4f}"
    )

    print("\nMatriz de confusão [0, 1]:")

    for row in metrics["confusion_matrix"]:
        print(row)

    print(
        "\nResultado salvo em:",
        REPORT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()