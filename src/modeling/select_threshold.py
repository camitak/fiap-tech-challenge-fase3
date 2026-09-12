from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
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

THRESHOLDS = np.round(
    np.arange(
        0.10,
        0.901,
        0.01,
    ),
    2,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_JSON = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "threshold_selection_2023.json"
)

REPORT_CSV = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "threshold_selection_2023.csv"
)


def build_candidates():
    """
    Configurações vencedoras da etapa de tuning.
    """

    return {
        "decision_tree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=500,
            min_samples_split=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            max_features="sqrt",
            min_samples_leaf=50,
            min_samples_split=100,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_threshold_metrics(
    y_true,
    proba_risk,
    threshold,
):
    """
    Considera risco como classe positiva de negócio:

        risco = não alfabetizado = target original 0

    Retorna métricas calculadas para um threshold específico.
    """

    actual_risk = (
        np.asarray(y_true) == 0
    )

    predicted_risk = (
        np.asarray(proba_risk)
        >= threshold
    )

    tp = int(
        np.sum(
            predicted_risk
            & actual_risk
        )
    )

    fp = int(
        np.sum(
            predicted_risk
            & ~actual_risk
        )
    )

    fn = int(
        np.sum(
            ~predicted_risk
            & actual_risk
        )
    )

    tn = int(
        np.sum(
            ~predicted_risk
            & ~actual_risk
        )
    )

    precision_risk = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall_risk = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    recall_literate = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1_risk = (
        2
        * precision_risk
        * recall_risk
        / (
            precision_risk
            + recall_risk
        )
        if (
            precision_risk
            + recall_risk
        ) > 0
        else 0.0
    )

    balanced_accuracy = (
        recall_risk
        + recall_literate
    ) / 2

    accuracy = (
        (tp + tn)
        / len(actual_risk)
    )

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision_risk": float(
            precision_risk
        ),
        "recall_risk": float(
            recall_risk
        ),
        "recall_alfabetizado": float(
            recall_literate
        ),
        "f1_risk": float(
            f1_risk
        ),
        "taxa_predita_risco": float(
            predicted_risk.mean()
        ),
        "tp_risk": tp,
        "fp_risk": fp,
        "fn_risk": fn,
        "tn_risk": tn,
    }


def generate_oof_probabilities(
    classifier,
    X,
    y,
    groups,
):
    """
    Gera probabilidades out-of-fold usando GroupKFold.

    Cada aluno recebe uma probabilidade produzida por
    um modelo que não foi treinado com o município
    daquele aluno.
    """

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    oof_proba_alfabetizado = np.full(
        shape=len(X),
        fill_value=np.nan,
        dtype=float,
    )

    fold_times = []

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        train_groups = set(
            groups.iloc[
                train_idx
            ].unique()
        )

        val_groups = set(
            groups.iloc[
                val_idx
            ].unique()
        )

        overlap = (
            train_groups
            & val_groups
        )

        assert not overlap, (
            "Há municípios presentes simultaneamente "
            "em treino e validação."
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
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        elapsed = (
            perf_counter()
            - start
        )

        fold_times.append(
            elapsed
        )

        proba = (
            pipeline.predict_proba(
                X.iloc[val_idx]
            )[:, 1]
        )

        oof_proba_alfabetizado[
            val_idx
        ] = proba

        print(
            f"Fold {fold}: "
            f"{len(val_idx):,} registros | "
            f"{len(val_groups):,} municípios | "
            f"{elapsed:.2f}s"
        )

    assert not np.isnan(
        oof_proba_alfabetizado
    ).any()

    return (
        oof_proba_alfabetizado,
        fold_times,
    )


def select_threshold_balanced_accuracy(
    threshold_df,
):
    """
    Critério oficial:

    1. maior Balanced Accuracy;
    2. maior Recall da classe de risco;
    3. maior F1 da classe de risco.
    """

    return (
        threshold_df
        .sort_values(
            by=[
                "balanced_accuracy",
                "recall_risk",
                "f1_risk",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )


def select_threshold_max_f1(
    threshold_df,
):
    """
    Diagnóstico alternativo.

    Mantemos o threshold que maximiza F1 apenas para
    demonstrar por que esse critério pode produzir
    uma solução degenerada neste problema.
    """

    return (
        threshold_df
        .sort_values(
            by=[
                "f1_risk",
                "balanced_accuracy",
                "recall_risk",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )


def main():
    print(
        "Carregando exclusivamente dados de 2023..."
    )

    df = load_modeling_data(
        2023
    )

    X = df[
        BASELINE_FEATURES
    ]

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

    all_threshold_rows = []
    model_summaries = []

    for (
        model_name,
        classifier,
    ) in build_candidates().items():

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"MODELO: {model_name}"
        )

        print(
            "=" * 60
        )

        (
            oof_proba_alfabetizado,
            fold_times,
        ) = generate_oof_probabilities(
            classifier=classifier,
            X=X,
            y=y,
            groups=groups,
        )

        proba_risk = (
            1.0
            - oof_proba_alfabetizado
        )

        y_risk = (
            y.to_numpy() == 0
        ).astype(int)

        pr_auc_risk = float(
            average_precision_score(
                y_risk,
                proba_risk,
            )
        )

        roc_auc = float(
            roc_auc_score(
                y,
                oof_proba_alfabetizado,
            )
        )

        threshold_results = []

        for threshold in THRESHOLDS:
            metrics = (
                calculate_threshold_metrics(
                    y_true=y,
                    proba_risk=proba_risk,
                    threshold=threshold,
                )
            )

            metrics[
                "modelo"
            ] = model_name

            threshold_results.append(
                metrics
            )

            all_threshold_rows.append(
                metrics
            )

        threshold_df = pd.DataFrame(
            threshold_results
        )

        best_row = (
            select_threshold_balanced_accuracy(
                threshold_df
            )
        )

        best_f1_row = (
            select_threshold_max_f1(
                threshold_df
            )
        )

        default_row = (
            threshold_df[
                threshold_df[
                    "threshold"
                ] == 0.50
            ]
            .iloc[0]
        )

        summary = {
            "modelo": model_name,

            "pr_auc_risk_oof": (
                pr_auc_risk
            ),

            "roc_auc_oof": (
                roc_auc
            ),

            "threshold_selecionado": float(
                best_row[
                    "threshold"
                ]
            ),

            "criterio_threshold": (
                "maximizacao da Balanced Accuracy; "
                "desempate por Recall e F1 "
                "da classe nao alfabetizado"
            ),

            "threshold_05": {
                "accuracy": float(
                    default_row[
                        "accuracy"
                    ]
                ),
                "balanced_accuracy": float(
                    default_row[
                        "balanced_accuracy"
                    ]
                ),
                "precision_risk": float(
                    default_row[
                        "precision_risk"
                    ]
                ),
                "recall_risk": float(
                    default_row[
                        "recall_risk"
                    ]
                ),
                "recall_alfabetizado": float(
                    default_row[
                        "recall_alfabetizado"
                    ]
                ),
                "f1_risk": float(
                    default_row[
                        "f1_risk"
                    ]
                ),
                "taxa_predita_risco": float(
                    default_row[
                        "taxa_predita_risco"
                    ]
                ),
            },

            "threshold_otimizado": {
                "threshold": float(
                    best_row[
                        "threshold"
                    ]
                ),
                "accuracy": float(
                    best_row[
                        "accuracy"
                    ]
                ),
                "balanced_accuracy": float(
                    best_row[
                        "balanced_accuracy"
                    ]
                ),
                "precision_risk": float(
                    best_row[
                        "precision_risk"
                    ]
                ),
                "recall_risk": float(
                    best_row[
                        "recall_risk"
                    ]
                ),
                "recall_alfabetizado": float(
                    best_row[
                        "recall_alfabetizado"
                    ]
                ),
                "f1_risk": float(
                    best_row[
                        "f1_risk"
                    ]
                ),
                "taxa_predita_risco": float(
                    best_row[
                        "taxa_predita_risco"
                    ]
                ),
            },

            "diagnostico_f1_maximo": {
                "threshold": float(
                    best_f1_row[
                        "threshold"
                    ]
                ),
                "accuracy": float(
                    best_f1_row[
                        "accuracy"
                    ]
                ),
                "balanced_accuracy": float(
                    best_f1_row[
                        "balanced_accuracy"
                    ]
                ),
                "precision_risk": float(
                    best_f1_row[
                        "precision_risk"
                    ]
                ),
                "recall_risk": float(
                    best_f1_row[
                        "recall_risk"
                    ]
                ),
                "recall_alfabetizado": float(
                    best_f1_row[
                        "recall_alfabetizado"
                    ]
                ),
                "f1_risk": float(
                    best_f1_row[
                        "f1_risk"
                    ]
                ),
                "taxa_predita_risco": float(
                    best_f1_row[
                        "taxa_predita_risco"
                    ]
                ),
            },

            "tempo_medio_fold_segundos": float(
                np.mean(
                    fold_times
                )
            ),
        }

        model_summaries.append(
            summary
        )

        print(
            "\nPR-AUC OOF:",
            f"{pr_auc_risk:.4f}",
        )

        print(
            "ROC-AUC OOF:",
            f"{roc_auc:.4f}",
        )

        print(
            "\nThreshold padrão 0.50:"
        )

        print(
            "  Balanced Accuracy:",
            f"{default_row['balanced_accuracy']:.4f}",
        )

        print(
            "  Precision risco:",
            f"{default_row['precision_risk']:.4f}",
        )

        print(
            "  Recall risco:",
            f"{default_row['recall_risk']:.4f}",
        )

        print(
            "  Recall alfabetizado:",
            f"{default_row['recall_alfabetizado']:.4f}",
        )

        print(
            "  F1 risco:",
            f"{default_row['f1_risk']:.4f}",
        )

        print(
            "  Taxa predita risco:",
            f"{default_row['taxa_predita_risco']:.4%}",
        )

        print(
            "\nThreshold selecionado "
            "(máxima Balanced Accuracy):",
            f"{best_row['threshold']:.2f}",
        )

        print(
            "  Balanced Accuracy:",
            f"{best_row['balanced_accuracy']:.4f}",
        )

        print(
            "  Precision risco:",
            f"{best_row['precision_risk']:.4f}",
        )

        print(
            "  Recall risco:",
            f"{best_row['recall_risk']:.4f}",
        )

        print(
            "  Recall alfabetizado:",
            f"{best_row['recall_alfabetizado']:.4f}",
        )

        print(
            "  F1 risco:",
            f"{best_row['f1_risk']:.4f}",
        )

        print(
            "  Taxa predita risco:",
            f"{best_row['taxa_predita_risco']:.4%}",
        )

        print(
            "\nDiagnóstico - threshold "
            "que maximiza F1:"
        )

        print(
            "  Threshold:",
            f"{best_f1_row['threshold']:.2f}",
        )

        print(
            "  Balanced Accuracy:",
            f"{best_f1_row['balanced_accuracy']:.4f}",
        )

        print(
            "  Precision risco:",
            f"{best_f1_row['precision_risk']:.4f}",
        )

        print(
            "  Recall risco:",
            f"{best_f1_row['recall_risk']:.4f}",
        )

        print(
            "  Recall alfabetizado:",
            f"{best_f1_row['recall_alfabetizado']:.4f}",
        )

        print(
            "  F1 risco:",
            f"{best_f1_row['f1_risk']:.4f}",
        )

        print(
            "  Taxa predita risco:",
            f"{best_f1_row['taxa_predita_risco']:.4%}",
        )

    # ---------------------------------------------------------
    # Persistência
    # ---------------------------------------------------------

    threshold_df_all = pd.DataFrame(
        all_threshold_rows
    )

    REPORT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_df_all.to_csv(
        REPORT_CSV,
        index=False,
    )

    report = {
        "ano_desenvolvimento": 2023,

        "ano_teste_final": 2024,

        "metodo_probabilidade": (
            "out-of-fold com GroupKFold"
        ),

        "group_variable": (
            "id_municipio"
        ),

        "n_splits": (
            N_SPLITS
        ),

        "criterio_threshold": (
            "maximizacao da Balanced Accuracy; "
            "desempate por Recall e F1 "
            "da classe nao alfabetizado"
        ),

        "observacao_f1": (
            "O threshold que maximiza F1 tambem foi "
            "registrado como diagnostico, mas nao foi "
            "utilizado como criterio oficial porque pode "
            "produzir classificacao excessiva da populacao "
            "como risco."
        ),

        "threshold_grid": {
            "inicio": float(
                THRESHOLDS.min()
            ),
            "fim": float(
                THRESHOLDS.max()
            ),
            "passo": 0.01,
        },

        "modelos": (
            model_summaries
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

    # ---------------------------------------------------------
    # Comparação final desta etapa
    # ---------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "modelo": item[
                    "modelo"
                ],

                "pr_auc_oof": item[
                    "pr_auc_risk_oof"
                ],

                "roc_auc_oof": item[
                    "roc_auc_oof"
                ],

                "threshold": item[
                    "threshold_selecionado"
                ],

                "balanced_accuracy": item[
                    "threshold_otimizado"
                ][
                    "balanced_accuracy"
                ],

                "precision_risk": item[
                    "threshold_otimizado"
                ][
                    "precision_risk"
                ],

                "recall_risk": item[
                    "threshold_otimizado"
                ][
                    "recall_risk"
                ],

                "recall_alfabetizado": item[
                    "threshold_otimizado"
                ][
                    "recall_alfabetizado"
                ],

                "f1_risk": item[
                    "threshold_otimizado"
                ][
                    "f1_risk"
                ],

                "taxa_predita_risco": item[
                    "threshold_otimizado"
                ][
                    "taxa_predita_risco"
                ],
            }

            for item
            in model_summaries
        ]
    )

    print(
        "\n"
        + "=" * 60
        + "\nRESUMO THRESHOLD OOF\n"
        + "=" * 60
    )

    print(
        summary_df.to_string(
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