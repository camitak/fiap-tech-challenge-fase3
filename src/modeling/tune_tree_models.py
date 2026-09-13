from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.data_loader import (
    load_modeling_data,
)
from src.preprocessing.preprocessor import (
    BASELINE_FEATURES,
    build_preprocessor,
)


RANDOM_STATE = 42
N_SPLITS = 5
N_ITER = 8

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_JSON = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "hyperparameter_search_2023.json"
)

REPORT_CSV = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "hyperparameter_search_2023.csv"
)


def pr_auc_risk_scorer(
    estimator,
    X,
    y,
) -> float:
    """
    PR-AUC considerando como classe positiva de negócio:

        1 = não alfabetizado

    O target original permanece:
        0 = não alfabetizado
        1 = alfabetizado

    Portanto, a probabilidade de risco é:

        1 - P(alfabetizado)
    """

    y_array = np.asarray(y)

    y_risk = (
        y_array == 0
    ).astype(int)

    proba_alfabetizado = (
        estimator.predict_proba(X)[:, 1]
    )

    proba_risk = (
        1.0 - proba_alfabetizado
    )

    return float(
        average_precision_score(
            y_risk,
            proba_risk,
        )
    )


SCORING = {
    "pr_auc_risk": pr_auc_risk_scorer,

    "roc_auc": "roc_auc",

    "balanced_accuracy": (
        "balanced_accuracy"
    ),

    "recall_risk": make_scorer(
        recall_score,
        pos_label=0,
        zero_division=0,
    ),

    "f1_risk": make_scorer(
        f1_score,
        pos_label=0,
        zero_division=0,
    ),
}


def build_searches():
    """
    Espaços de busca deliberadamente moderados.

    O objetivo não é testar centenas de combinações,
    mas explorar os hiperparâmetros mais relevantes
    com custo computacional controlado.
    """

    decision_tree = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    include_uf=False
                ),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    decision_tree_space = {
        "classifier__criterion": [
            "gini",
            "entropy",
        ],

        "classifier__max_depth": [
            6,
            8,
            10,
            12,
            15,
            20,
        ],

        "classifier__min_samples_leaf": [
            50,
            100,
            250,
            500,
            1000,
        ],

        "classifier__min_samples_split": [
            100,
            250,
            500,
            1000,
            2000,
        ],

        "classifier__class_weight": [
            None,
            "balanced",
        ],

        "classifier__max_features": [
            None,
            "sqrt",
        ],
    }

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    include_uf=False
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest_space = {
        "classifier__n_estimators": [
            100,
            150,
            200,
        ],

        "classifier__max_depth": [
            8,
            12,
            16,
            20,
        ],

        "classifier__min_samples_leaf": [
            50,
            100,
            250,
            500,
        ],

        "classifier__min_samples_split": [
            100,
            250,
            500,
            1000,
        ],

        "classifier__max_features": [
            "sqrt",
            0.7,
        ],

        "classifier__class_weight": [
            None,
            "balanced_subsample",
        ],
    }

    return {
        "decision_tree": (
            decision_tree,
            decision_tree_space,
        ),

        "random_forest": (
            random_forest,
            random_forest_space,
        ),
    }


def extract_best_result(
    model_name,
    search,
    elapsed_seconds,
):
    best_index = (
        search.best_index_
    )

    cv_results = (
        search.cv_results_
    )

    train_pr_auc = float(
        cv_results[
            "mean_train_pr_auc_risk"
        ][best_index]
    )

    validation_pr_auc = float(
        cv_results[
            "mean_test_pr_auc_risk"
        ][best_index]
    )

    return {
        "modelo": model_name,

        "melhores_parametros": (
            search.best_params_
        ),

        "pr_auc_risk_train": (
            train_pr_auc
        ),

        "pr_auc_risk_validation": (
            validation_pr_auc
        ),

        "gap_pr_auc_train_validation": float(
            train_pr_auc
            - validation_pr_auc
        ),

        "roc_auc_validation": float(
            cv_results[
                "mean_test_roc_auc"
            ][best_index]
        ),

        "balanced_accuracy_validation": float(
            cv_results[
                "mean_test_balanced_accuracy"
            ][best_index]
        ),

        "recall_risk_validation": float(
            cv_results[
                "mean_test_recall_risk"
            ][best_index]
        ),

        "f1_risk_validation": float(
            cv_results[
                "mean_test_f1_risk"
            ][best_index]
        ),

        "pr_auc_risk_std": float(
            cv_results[
                "std_test_pr_auc_risk"
            ][best_index]
        ),

        "tempo_busca_segundos": float(
            elapsed_seconds
        ),
    }


def extract_candidates(
    model_name,
    search,
):
    results = (
        search.cv_results_
    )

    rows = []

    for index, params in enumerate(
        results["params"]
    ):
        rows.append(
            {
                "modelo": model_name,

                "rank_pr_auc_risk": int(
                    results[
                        "rank_test_pr_auc_risk"
                    ][index]
                ),

                "pr_auc_risk_train": float(
                    results[
                        "mean_train_pr_auc_risk"
                    ][index]
                ),

                "pr_auc_risk_validation": float(
                    results[
                        "mean_test_pr_auc_risk"
                    ][index]
                ),

                "pr_auc_risk_std": float(
                    results[
                        "std_test_pr_auc_risk"
                    ][index]
                ),

                "roc_auc_validation": float(
                    results[
                        "mean_test_roc_auc"
                    ][index]
                ),

                "balanced_accuracy_validation": float(
                    results[
                        "mean_test_balanced_accuracy"
                    ][index]
                ),

                "recall_risk_validation": float(
                    results[
                        "mean_test_recall_risk"
                    ][index]
                ),

                "f1_risk_validation": float(
                    results[
                        "mean_test_f1_risk"
                    ][index]
                ),

                "fit_time_mean": float(
                    results[
                        "mean_fit_time"
                    ][index]
                ),

                "parametros": json.dumps(
                    params,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    return rows


def main():
    print(
        "Carregando exclusivamente dados de 2023..."
    )

    df = load_modeling_data(2023)

    X = df[BASELINE_FEATURES]

    y = (
        df["alfabetizado"]
        .astype(int)
    )

    groups = (
        df["id_municipio"]
        .astype(str)
    )

    print(
        f"Registros: {len(df):,}"
    )

    print(
        f"Municípios: "
        f"{groups.nunique():,}"
    )

    print(
        f"Features: "
        f"{len(BASELINE_FEATURES)}"
    )

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    searches = (
        build_searches()
    )

    best_results = []
    candidate_rows = []

    for (
        model_name,
        (
            pipeline,
            param_space,
        ),
    ) in searches.items():

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"OTIMIZANDO: {model_name}"
        )

        print(
            "=" * 60
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,

            n_iter=N_ITER,

            scoring=SCORING,

            refit="pr_auc_risk",

            cv=cv,

            random_state=RANDOM_STATE,

            # Mantemos a busca sequencial para evitar
            # múltiplas cópias do dataset em memória.
            # A Random Forest já paraleliza suas árvores.
            n_jobs=1,

            verbose=2,

            return_train_score=True,

            error_score="raise",
        )

        start = perf_counter()

        search.fit(
            X,
            y,
            groups=groups,
        )

        elapsed = (
            perf_counter()
            - start
        )

        best_result = (
            extract_best_result(
                model_name=model_name,
                search=search,
                elapsed_seconds=elapsed,
            )
        )

        best_results.append(
            best_result
        )

        candidate_rows.extend(
            extract_candidates(
                model_name=model_name,
                search=search,
            )
        )

        print(
            "\nMelhores parâmetros:"
        )

        for key, value in (
            search.best_params_.items()
        ):
            print(
                f"  {key}: {value}"
            )

        print(
            "\nMelhores métricas médias:"
        )

        print(
            "PR-AUC risco:",
            f"{best_result['pr_auc_risk_validation']:.4f}",
        )

        print(
            "ROC-AUC:",
            f"{best_result['roc_auc_validation']:.4f}",
        )

        print(
            "Balanced Accuracy:",
            f"{best_result['balanced_accuracy_validation']:.4f}",
        )

        print(
            "Recall risco:",
            f"{best_result['recall_risk_validation']:.4f}",
        )

        print(
            "F1 risco:",
            f"{best_result['f1_risk_validation']:.4f}",
        )

        print(
            "PR-AUC treino:",
            f"{best_result['pr_auc_risk_train']:.4f}",
        )

        print(
            "Gap treino-validação:",
            f"{best_result['gap_pr_auc_train_validation']:.4f}",
        )

    # ---------------------------------------------------------
    # Relatórios
    # ---------------------------------------------------------

    candidates_df = pd.DataFrame(
        candidate_rows
    )

    candidates_df = (
        candidates_df
        .sort_values(
            [
                "modelo",
                "rank_pr_auc_risk",
            ]
        )
    )

    REPORT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "ano_desenvolvimento": 2023,

        "ano_teste_final": 2024,

        "criterio_otimizacao": (
            "PR-AUC da classe não alfabetizado"
        ),

        "refit_metric": (
            "pr_auc_risk"
        ),

        "group_variable": (
            "id_municipio"
        ),

        "cross_validation": {
            "metodo": "GroupKFold",
            "n_splits": N_SPLITS,
        },

        "random_state": (
            RANDOM_STATE
        ),

        "n_iter_por_modelo": (
            N_ITER
        ),

        "features": (
            BASELINE_FEATURES
        ),

        "melhores_resultados": (
            best_results
        ),
    }

    with REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    candidates_df.to_csv(
        REPORT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Resumo final
    # ---------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "modelo": result[
                    "modelo"
                ],

                "pr_auc_risk": result[
                    "pr_auc_risk_validation"
                ],

                "roc_auc": result[
                    "roc_auc_validation"
                ],

                "balanced_accuracy": result[
                    "balanced_accuracy_validation"
                ],

                "recall_risk": result[
                    "recall_risk_validation"
                ],

                "f1_risk": result[
                    "f1_risk_validation"
                ],

                "gap_train_validation": result[
                    "gap_pr_auc_train_validation"
                ],
            }

            for result
            in best_results
        ]
    )

    summary = (
        summary
        .sort_values(
            "pr_auc_risk",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 60
        + "\nRESUMO OTIMIZACAO\n"
        + "=" * 60
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
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