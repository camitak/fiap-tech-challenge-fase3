from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

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

REPORT_JSON = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "model_comparison_2023.json"
)

REPORT_CSV = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "model_comparison_2023.csv"
)


def create_development_split(df):
    """
    Divide 2023 em treino e validação por município.

    O mesmo município nunca aparece simultaneamente
    nos dois conjuntos.
    """

    X = df[BASELINE_FEATURES]
    y = df["alfabetizado"].astype(int)
    groups = df["id_municipio"].astype(str)

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
        "Há municípios presentes em treino e validação."
    )

    split_info = {
        "registros_treino": int(len(X_train)),
        "registros_validacao": int(len(X_val)),
        "municipios_treino": int(
            len(train_municipios)
        ),
        "municipios_validacao": int(
            len(val_municipios)
        ),
        "municipios_overlap": int(
            len(overlap)
        ),
        "taxa_alfabetizados_treino": float(
            y_train.mean()
        ),
        "taxa_alfabetizados_validacao": float(
            y_val.mean()
        ),
    }

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        split_info,
    )


def build_candidates():
    """
    Modelos candidatos antes de otimização.

    Os hiperparâmetros das árvores são deliberadamente
    conservadores para evitar árvores excessivamente
    profundas nesta primeira comparação.
    """

    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=100,
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=100,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(
    name,
    classifier,
    X_train,
    X_val,
    y_train,
    y_val,
):
    print(
        f"\n{'=' * 60}\n"
        f"Modelo: {name}\n"
        f"{'=' * 60}"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    include_uf=False
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    start = perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        perf_counter() - start
    )

    y_pred = pipeline.predict(X_val)

    y_proba = pipeline.predict_proba(
        X_val
    )[:, 1]

    metrics = (
        calculate_classification_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_proba_alfabetizado=y_proba,
        )
    )

    risk = (
        metrics[
            "classe_0_nao_alfabetizado"
        ]
    )

    predicted_non_literate_rate = float(
        (y_pred == 0).mean()
    )

    print(
        f"Tempo de treino: "
        f"{training_seconds:.2f}s"
    )

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

    print(
        f"PR-AUC não alfabetizado: "
        f"{metrics['pr_auc_nao_alfabetizado']:.4f}"
    )

    print(
        f"Precision não alfabetizado: "
        f"{risk['precision']:.4f}"
    )

    print(
        f"Recall não alfabetizado: "
        f"{risk['recall']:.4f}"
    )

    print(
        f"F1 não alfabetizado: "
        f"{risk['f1']:.4f}"
    )

    print(
        f"Taxa predita não alfabetizado: "
        f"{predicted_non_literate_rate:.4%}"
    )

    return {
        "modelo": name,
        "treinamento_segundos": float(
            training_seconds
        ),
        "taxa_predita_nao_alfabetizado": (
            predicted_non_literate_rate
        ),
        "metricas": metrics,
    }


def main():
    print(
        "Carregando exclusivamente dados de 2023..."
    )

    df = load_modeling_data(2023)

    print(
        f"Registros: {len(df):,}"
    )

    print(
        "Municípios:",
        f"{df['id_municipio'].nunique():,}",
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        split_info,
    ) = create_development_split(df)

    print("\n--- Split territorial ---")

    for key, value in split_info.items():
        print(
            f"{key}: {value}"
        )

    majority_class = int(
        y_train.mode().iloc[0]
    )

    majority_accuracy = float(
        (y_val == majority_class).mean()
    )

    print(
        "\nAccuracy baseline majoritário:",
        f"{majority_accuracy:.4f}",
    )

    results = []

    for name, classifier in (
        build_candidates().items()
    ):
        result = evaluate_model(
            name=name,
            classifier=classifier,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
        )

        results.append(result)

    # ----------------------------------------------------------
    # Resumo comparativo
    # ----------------------------------------------------------

    rows = []

    for result in results:
        metrics = result["metricas"]

        risk = metrics[
            "classe_0_nao_alfabetizado"
        ]

        rows.append(
            {
                "modelo": result["modelo"],
                "accuracy": metrics[
                    "accuracy"
                ],
                "balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "roc_auc": metrics[
                    "roc_auc_alfabetizado"
                ],
                "pr_auc_nao_alfabetizado": metrics[
                    "pr_auc_nao_alfabetizado"
                ],
                "precision_nao_alfabetizado": risk[
                    "precision"
                ],
                "recall_nao_alfabetizado": risk[
                    "recall"
                ],
                "f1_nao_alfabetizado": risk[
                    "f1"
                ],
                "taxa_predita_nao_alfabetizado": (
                    result[
                        "taxa_predita_nao_alfabetizado"
                    ]
                ),
                "treinamento_segundos": result[
                    "treinamento_segundos"
                ],
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        by="balanced_accuracy",
        ascending=False,
    )

    print(
        "\n"
        + "=" * 60
        + "\nRESUMO COMPARATIVO\n"
        + "=" * 60
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    REPORT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_report = {
        "ano_desenvolvimento": 2023,
        "ano_teste_final": 2024,
        "random_state": RANDOM_STATE,
        "validation_size_groups": (
            VALIDATION_SIZE
        ),
        "baseline_majoritario": {
            "classe": majority_class,
            "accuracy_validacao": (
                majority_accuracy
            ),
        },
        "split": split_info,
        "features": BASELINE_FEATURES,
        "resultados": results,
    }

    with REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            full_report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    comparison.to_csv(
        REPORT_CSV,
        index=False,
    )

    print(
        "\nResultados salvos em:"
    )

    print(
        "-",
        REPORT_JSON.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "-",
        REPORT_CSV.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()