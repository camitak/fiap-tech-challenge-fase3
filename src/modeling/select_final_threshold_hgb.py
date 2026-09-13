from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocessing.preprocessor import (
    build_preprocessor,
    get_model_features,
)


# ============================================================
# Tech Challenge Fase 3
#
# Seleção do threshold operacional final
#
# Modelo já congelado:
# HistGradientBoostingClassifier
#
# Feature set já congelado:
# ENRICHED + UF
#
# Regras:
# - somente 2023;
# - 2024 permanece completamente fechado;
# - GroupKFold por município;
# - probabilidades OOF;
# - risco = 1 - P(alfabetizado);
# - hiperparâmetros NÃO são alterados;
# - features NÃO são alteradas;
# - threshold escolhido por Balanced Accuracy;
# - desempate:
#       1. maior recall de risco
#       2. maior F1 de risco
#
# Depois deste script ficam congelados:
# - features
# - algoritmo
# - hiperparâmetros
# - threshold
#
# Somente depois poderá ser aberto 2024.
# ============================================================


RANDOM_STATE = 42
N_SPLITS = 5


# ------------------------------------------------------------
# Grid de threshold de risco.
#
# 0.10 até 0.90, passo de 0.01.
#
# Mantemos uma grade discreta e interpretável para evitar
# escolher um threshold excessivamente específico para 2023.
# ------------------------------------------------------------

THRESHOLDS = np.round(
    np.arange(
        0.10,
        0.901,
        0.01,
    ),
    2,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "base_modelagem"
    / "v2"
    / "ano=2023"
)


REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)


REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Saídas
# ============================================================


FOLDS_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_final_oof_2023_folds.csv"
)


THRESHOLD_GRID_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_final_threshold_grid_2023.csv"
)


FINAL_THRESHOLD_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_final_threshold_2023.json"
)


# ============================================================
# Features
# ============================================================


FEATURES = get_model_features(
    feature_set="enriched",
    include_uf=True,
)


CONTROL_COLUMNS = [
    "aluno_key",
    "id_municipio",
    "alfabetizado",
]


READ_COLUMNS = list(
    dict.fromkeys(
        CONTROL_COLUMNS
        + FEATURES
    )
)


# ============================================================
# Configuração FINALISTA congelada do HGB
#
# Estes valores NÃO devem ser alterados nesta etapa.
# ============================================================


FINAL_HGB_PARAMS = {
    "learning_rate": (
        0.06197763962309192
    ),

    "max_iter": 230,

    "max_leaf_nodes": 15,

    "max_depth": None,

    "min_samples_leaf": 70,

    "l2_regularization": (
        0.716404042819101
    ),

    "max_bins": 255,

    "early_stopping": False,

    "random_state": (
        RANDOM_STATE
    ),
}


# ============================================================
# Dados
# ============================================================


def load_2023() -> pd.DataFrame:

    files = sorted(
        DATA_PATH.glob(
            "*.parquet"
        )
    )

    if not files:

        raise FileNotFoundError(
            "Nenhum arquivo Parquet "
            f"encontrado em {DATA_PATH}"
        )


    frames = [
        pd.read_parquet(
            file,
            columns=READ_COLUMNS,
        )

        for file in files
    ]


    df = pd.concat(
        frames,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Reconciliação da base de desenvolvimento
    # --------------------------------------------------------

    assert (
        len(df)
        == 1_502_809
    )

    assert (
        df[
            "aluno_key"
        ]
        .nunique()
        == 1_502_809
    )

    assert int(
        (
            df[
                "alfabetizado"
            ]
            == 0
        ).sum()
    ) == 625_382

    assert int(
        (
            df[
                "alfabetizado"
            ]
            == 1
        ).sum()
    ) == 877_427

    assert (
        df[
            "id_municipio"
        ]
        .isna()
        .sum()
        == 0
    )


    return df


# ============================================================
# Sparse -> dense
# ============================================================


def to_dense(
    matrix,
):

    if sparse.issparse(
        matrix
    ):

        return matrix.toarray()


    return np.asarray(
        matrix
    )


# ============================================================
# Pipeline finalista
# ============================================================


def build_final_pipeline() -> Pipeline:

    preprocessor = (
        build_preprocessor(
            feature_set="enriched",
            include_uf=True,
        )
    )


    dense_transformer = (
        FunctionTransformer(
            to_dense,
            validate=False,
        )
    )


    model = (
        HistGradientBoostingClassifier(
            **FINAL_HGB_PARAMS
        )
    )


    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "to_dense",
                dense_transformer,
            ),

            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# Probabilidade alfabetizado=1
# ============================================================


def predict_literate_probability(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:

    probabilities = (
        pipeline.predict_proba(
            X
        )
    )


    classes = (
        pipeline
        .named_steps[
            "model"
        ]
        .classes_
    )


    positions = np.where(
        classes == 1
    )[0]


    if len(
        positions
    ) != 1:

        raise RuntimeError(
            "Classe alfabetizado=1 "
            "não localizada."
        )


    return probabilities[
        :,
        positions[0],
    ]


# ============================================================
# Métricas independentes de threshold
# ============================================================


def ranking_metrics(
    y_true: np.ndarray,
    p_literate: np.ndarray,
) -> dict[str, float]:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    p_literate = np.asarray(
        p_literate,
        dtype=float,
    )


    y_risk = (
        1
        - y_true
    )

    p_risk = (
        1.0
        - p_literate
    )


    return {
        "roc_auc_literate": (
            roc_auc_score(
                y_true,
                p_literate,
            )
        ),

        "pr_auc_risk": (
            average_precision_score(
                y_risk,
                p_risk,
            )
        ),
    }


# ============================================================
# Métricas para um threshold de RISCO
#
# Se P(risco) >= threshold:
#     previsão = risco / não alfabetizado
#
# y_risk:
#     1 = não alfabetizado
#     0 = alfabetizado
# ============================================================


def calculate_threshold_metrics(
    y_true_literate: np.ndarray,
    p_risk: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:

    y_true_literate = np.asarray(
        y_true_literate,
        dtype=int,
    )

    p_risk = np.asarray(
        p_risk,
        dtype=float,
    )


    y_true_risk = (
        1
        - y_true_literate
    )


    y_pred_risk = (
        p_risk
        >= threshold
    ).astype(int)


    y_pred_literate = (
        1
        - y_pred_risk
    )


    # --------------------------------------------------------
    # Matriz de confusão na perspectiva da classe RISCO:
    #
    # TN = alfabetizado corretamente fora do risco
    # FP = alfabetizado classificado como risco
    # FN = não alfabetizado não identificado
    # TP = não alfabetizado identificado como risco
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true_risk,
            y_pred_risk,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )


    return {
        "threshold_risk": (
            float(
                threshold
            )
        ),

        "accuracy": (
            accuracy_score(
                y_true_literate,
                y_pred_literate,
            )
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true_risk,
                y_pred_risk,
            )
        ),

        "precision_risk": (
            precision_score(
                y_true_risk,
                y_pred_risk,
                zero_division=0,
            )
        ),

        "recall_risk": (
            recall_score(
                y_true_risk,
                y_pred_risk,
                zero_division=0,
            )
        ),

        "f1_risk": (
            f1_score(
                y_true_risk,
                y_pred_risk,
                zero_division=0,
            )
        ),

        "recall_literate": (
            recall_score(
                y_true_literate,
                y_pred_literate,
                pos_label=1,
                zero_division=0,
            )
        ),

        "actual_risk_rate": (
            y_true_risk.mean()
        ),

        "predicted_risk_rate": (
            y_pred_risk.mean()
        ),

        "true_negative": (
            int(tn)
        ),

        "false_positive": (
            int(fp)
        ),

        "false_negative": (
            int(fn)
        ),

        "true_positive": (
            int(tp)
        ),
    }


# ============================================================
# Gerar probabilidades OOF
# ============================================================


def generate_oof(
    df: pd.DataFrame,
    folds,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:

    X = df[
        FEATURES
    ]


    y = (
        df[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    groups = (
        df[
            "id_municipio"
        ]
        .astype(str)
    )


    oof_p_literate = np.full(
        len(df),
        np.nan,
        dtype=np.float32,
    )


    oof_fold = np.zeros(
        len(df),
        dtype=np.int8,
    )


    fold_rows = []


    for fold_number, (
        train_index,
        valid_index,
    ) in enumerate(
        folds,
        start=1,
    ):

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"FOLD {fold_number}/{N_SPLITS}"
        )

        print(
            "=" * 72
        )


        train_groups = set(
            groups.iloc[
                train_index
            ].unique()
        )


        valid_groups = set(
            groups.iloc[
                valid_index
            ].unique()
        )


        overlap = (
            train_groups
            & valid_groups
        )


        assert not overlap


        print(
            f"Treino: "
            f"{len(train_index):,}"
        )

        print(
            f"Validação: "
            f"{len(valid_index):,}"
        )

        print(
            f"Municípios treino: "
            f"{len(train_groups):,}"
        )

        print(
            f"Municípios validação: "
            f"{len(valid_groups):,}"
        )

        print(
            "Overlap municípios: 0"
        )


        pipeline = (
            build_final_pipeline()
        )


        start_time = (
            time.perf_counter()
        )


        pipeline.fit(
            X.iloc[
                train_index
            ],
            y[
                train_index
            ],
        )


        p_literate = (
            predict_literate_probability(
                pipeline,
                X.iloc[
                    valid_index
                ],
            )
        )


        elapsed = (
            time.perf_counter()
            - start_time
        )


        oof_p_literate[
            valid_index
        ] = (
            p_literate.astype(
                np.float32
            )
        )


        oof_fold[
            valid_index
        ] = (
            fold_number
        )


        fold_metrics = (
            ranking_metrics(
                y[
                    valid_index
                ],
                p_literate,
            )
        )


        transformed_features = len(
            pipeline
            .named_steps[
                "preprocessor"
            ]
            .get_feature_names_out()
        )


        assert (
            transformed_features
            == 45
        )


        fold_rows.append(
            {
                "fold": (
                    fold_number
                ),

                "train_rows": (
                    len(
                        train_index
                    )
                ),

                "validation_rows": (
                    len(
                        valid_index
                    )
                ),

                "train_municipalities": (
                    len(
                        train_groups
                    )
                ),

                "validation_municipalities": (
                    len(
                        valid_groups
                    )
                ),

                "municipality_overlap": (
                    len(
                        overlap
                    )
                ),

                "raw_features": (
                    len(
                        FEATURES
                    )
                ),

                "transformed_features": (
                    transformed_features
                ),

                "roc_auc_literate": (
                    fold_metrics[
                        "roc_auc_literate"
                    ]
                ),

                "pr_auc_risk": (
                    fold_metrics[
                        "pr_auc_risk"
                    ]
                ),

                "runtime_seconds": (
                    elapsed
                ),
            }
        )


        print(
            f"PR-AUC risco: "
            f"{fold_metrics['pr_auc_risk']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{fold_metrics['roc_auc_literate']:.4f}"
        )

        print(
            f"Tempo: "
            f"{elapsed:.1f}s"
        )


        del pipeline
        del p_literate

        gc.collect()


    if np.isnan(
        oof_p_literate
    ).any():

        missing = int(
            np.isnan(
                oof_p_literate
            ).sum()
        )

        raise RuntimeError(
            "Predições OOF incompletas. "
            f"Ausentes: {missing:,}"
        )


    if (
        oof_fold
        == 0
    ).any():

        raise RuntimeError(
            "Existem registros sem fold OOF."
        )


    folds_df = pd.DataFrame(
        fold_rows
    )


    return (
        oof_p_literate,
        oof_fold,
        folds_df,
    )


# ============================================================
# Avaliar grid de thresholds
# ============================================================


def evaluate_threshold_grid(
    y_true: np.ndarray,
    p_risk: np.ndarray,
) -> pd.DataFrame:

    rows = []


    print(
        "\nAvaliando thresholds..."
    )


    for threshold in (
        THRESHOLDS
    ):

        metrics = (
            calculate_threshold_metrics(
                y_true_literate=(
                    y_true
                ),
                p_risk=(
                    p_risk
                ),
                threshold=(
                    float(
                        threshold
                    )
                ),
            )
        )


        rows.append(
            metrics
        )


    grid = pd.DataFrame(
        rows
    )


    return grid


# ============================================================
# Seleção oficial do threshold
# ============================================================


def select_best_threshold(
    grid: pd.DataFrame,
) -> pd.Series:

    # --------------------------------------------------------
    # Critério definido previamente:
    #
    # 1. maior Balanced Accuracy
    # 2. maior recall de risco
    # 3. maior F1 de risco
    #
    # Threshold menor/maior NÃO participa como regra de
    # desempate adicional.
    # --------------------------------------------------------

    ranked = (
        grid
        .sort_values(
            [
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
        .reset_index(
            drop=True
        )
    )


    return ranked.iloc[0]


# ============================================================
# Resumo final
# ============================================================


def build_final_summary(
    df: pd.DataFrame,
    oof_p_literate: np.ndarray,
    folds_df: pd.DataFrame,
    best: pd.Series,
) -> dict:

    y = (
        df[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    p_risk = (
        1.0
        - oof_p_literate
    )


    oof_rank = (
        ranking_metrics(
            y,
            oof_p_literate,
        )
    )


    threshold_050 = (
        calculate_threshold_metrics(
            y_true_literate=y,
            p_risk=p_risk,
            threshold=0.50,
        )
    )


    return {
        "model": (
            "HistGradientBoostingClassifier"
        ),

        "feature_set": (
            "enriched_uf"
        ),

        "development_year": 2023,

        "validation_strategy": (
            "5-fold GroupKFold by id_municipio"
        ),

        "n_rows": (
            int(
                len(df)
            )
        ),

        "n_municipalities": (
            int(
                df[
                    "id_municipio"
                ]
                .nunique()
            )
        ),

        "oof_roc_auc_literate": (
            float(
                oof_rank[
                    "roc_auc_literate"
                ]
            )
        ),

        "oof_pr_auc_risk": (
            float(
                oof_rank[
                    "pr_auc_risk"
                ]
            )
        ),

        "fold_mean_pr_auc_risk": (
            float(
                folds_df[
                    "pr_auc_risk"
                ]
                .mean()
            )
        ),

        "fold_std_pr_auc_risk": (
            float(
                folds_df[
                    "pr_auc_risk"
                ]
                .std(
                    ddof=1
                )
            )
        ),

        "fold_mean_roc_auc_literate": (
            float(
                folds_df[
                    "roc_auc_literate"
                ]
                .mean()
            )
        ),

        "fold_std_roc_auc_literate": (
            float(
                folds_df[
                    "roc_auc_literate"
                ]
                .std(
                    ddof=1
                )
            )
        ),

        "selection_rule": (
            "maximize balanced_accuracy; "
            "tie-break by higher recall_risk; "
            "then higher f1_risk"
        ),

        "threshold_grid": {
            "minimum": 0.10,
            "maximum": 0.90,
            "step": 0.01,
        },

        "selected_threshold_risk": (
            float(
                best[
                    "threshold_risk"
                ]
            )
        ),

        "selected_threshold_metrics": {
            "accuracy": (
                float(
                    best[
                        "accuracy"
                    ]
                )
            ),

            "balanced_accuracy": (
                float(
                    best[
                        "balanced_accuracy"
                    ]
                )
            ),

            "precision_risk": (
                float(
                    best[
                        "precision_risk"
                    ]
                )
            ),

            "recall_risk": (
                float(
                    best[
                        "recall_risk"
                    ]
                )
            ),

            "f1_risk": (
                float(
                    best[
                        "f1_risk"
                    ]
                )
            ),

            "recall_literate": (
                float(
                    best[
                        "recall_literate"
                    ]
                )
            ),

            "actual_risk_rate": (
                float(
                    best[
                        "actual_risk_rate"
                    ]
                )
            ),

            "predicted_risk_rate": (
                float(
                    best[
                        "predicted_risk_rate"
                    ]
                )
            ),

            "true_negative": (
                int(
                    best[
                        "true_negative"
                    ]
                )
            ),

            "false_positive": (
                int(
                    best[
                        "false_positive"
                    ]
                )
            ),

            "false_negative": (
                int(
                    best[
                        "false_negative"
                    ]
                )
            ),

            "true_positive": (
                int(
                    best[
                        "true_positive"
                    ]
                )
            ),
        },

        "threshold_050_diagnostic": {
            "balanced_accuracy": (
                float(
                    threshold_050[
                        "balanced_accuracy"
                    ]
                )
            ),

            "precision_risk": (
                float(
                    threshold_050[
                        "precision_risk"
                    ]
                )
            ),

            "recall_risk": (
                float(
                    threshold_050[
                        "recall_risk"
                    ]
                )
            ),

            "f1_risk": (
                float(
                    threshold_050[
                        "f1_risk"
                    ]
                )
            ),

            "predicted_risk_rate": (
                float(
                    threshold_050[
                        "predicted_risk_rate"
                    ]
                )
            ),
        },

        "final_hgb_params": (
            FINAL_HGB_PARAMS
        ),

        "test_year_2024_used": False,
    }


# ============================================================
# Main
# ============================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "SELEÇÃO DO THRESHOLD FINAL — HGB"
    )

    print(
        "=" * 72
    )

    print(
        "Desenvolvimento: somente 2023."
    )

    print(
        "2024 permanece fechado."
    )

    print(
        "Feature set: ENRICHED + UF."
    )

    print(
        "Modelo: HistGradientBoostingClassifier."
    )

    print(
        "Hiperparâmetros: congelados."
    )

    print(
        "Critério: maximizar Balanced Accuracy."
    )

    print(
        "Desempate: recall risco -> F1 risco."
    )


    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    print(
        "\nCarregando base..."
    )


    df = load_2023()


    print(
        f"✓ Linhas: "
        f"{len(df):,}"
    )

    print(
        f"✓ Municípios: "
        f"{df['id_municipio'].nunique():,}"
    )

    print(
        f"✓ Features brutas: "
        f"{len(FEATURES)}"
    )


    # --------------------------------------------------------
    # Folds idênticos aos experimentos anteriores
    # --------------------------------------------------------

    y = (
        df[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    groups = (
        df[
            "id_municipio"
        ]
        .astype(str)
    )


    splitter = GroupKFold(
        n_splits=N_SPLITS
    )


    folds = list(
        splitter.split(
            X=np.zeros(
                len(df)
            ),
            y=y,
            groups=groups,
        )
    )


    assert (
        len(folds)
        == N_SPLITS
    )


    # --------------------------------------------------------
    # OOF
    # --------------------------------------------------------

    total_start = (
        time.perf_counter()
    )


    (
        oof_p_literate,
        oof_fold,
        folds_df,
    ) = generate_oof(
        df=df,
        folds=folds,
    )


    p_risk = (
        1.0
        - oof_p_literate
    )


    # --------------------------------------------------------
    # Auditorias OOF
    # --------------------------------------------------------

    assert np.all(
        (
            oof_p_literate
            >= 0.0
        )
        & (
            oof_p_literate
            <= 1.0
        )
    )


    assert np.all(
        (
            p_risk
            >= 0.0
        )
        & (
            p_risk
            <= 1.0
        )
    )


    assert np.allclose(
        (
            oof_p_literate
            + p_risk
        ),
        1.0,
        atol=1e-6,
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    threshold_grid = (
        evaluate_threshold_grid(
            y_true=y,
            p_risk=p_risk,
        )
    )


    best = (
        select_best_threshold(
            threshold_grid
        )
    )


    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    summary = (
        build_final_summary(
            df=df,
            oof_p_literate=(
                oof_p_literate
            ),
            folds_df=(
                folds_df
            ),
            best=best,
        )
    )


    # --------------------------------------------------------
    # Persistência
    # --------------------------------------------------------

    folds_df.to_csv(
        FOLDS_OUTPUT_PATH,
        index=False,
    )


    threshold_grid.to_csv(
        THRESHOLD_GRID_OUTPUT_PATH,
        index=False,
    )


    with open(
        FINAL_THRESHOLD_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


    elapsed = (
        time.perf_counter()
        - total_start
    )


    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PERFORMANCE OOF — MODELO CONGELADO"
    )

    print(
        "=" * 72
    )


    print(
        f"PR-AUC risco: "
        f"{summary['oof_pr_auc_risk']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{summary['oof_roc_auc_literate']:.4f}"
    )

    print(
        f"PR-AUC média folds: "
        f"{summary['fold_mean_pr_auc_risk']:.4f}"
    )

    print(
        f"PR-AUC std folds: "
        f"{summary['fold_std_pr_auc_risk']:.4f}"
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "THRESHOLD SELECIONADO"
    )

    print(
        "=" * 72
    )


    print(
        f"Threshold de risco: "
        f"{summary['selected_threshold_risk']:.2f}"
    )


    selected = (
        summary[
            "selected_threshold_metrics"
        ]
    )


    print(
        f"Accuracy: "
        f"{selected['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{selected['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision risco: "
        f"{selected['precision_risk']:.4f}"
    )

    print(
        f"Recall risco: "
        f"{selected['recall_risk']:.4f}"
    )

    print(
        f"F1 risco: "
        f"{selected['f1_risk']:.4f}"
    )

    print(
        f"Recall alfabetizado: "
        f"{selected['recall_literate']:.4f}"
    )

    print(
        f"Taxa real de risco: "
        f"{selected['actual_risk_rate']:.4f}"
    )

    print(
        f"Taxa prevista de risco: "
        f"{selected['predicted_risk_rate']:.4f}"
    )


    print(
        "\nMatriz de confusão "
        "(classe positiva = risco):"
    )

    print(
        f"TN: "
        f"{selected['true_negative']:,}"
    )

    print(
        f"FP: "
        f"{selected['false_positive']:,}"
    )

    print(
        f"FN: "
        f"{selected['false_negative']:,}"
    )

    print(
        f"TP: "
        f"{selected['true_positive']:,}"
    )


    # --------------------------------------------------------
    # Comparar contra 0.50
    # --------------------------------------------------------

    diagnostic = (
        summary[
            "threshold_050_diagnostic"
        ]
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "COMPARAÇÃO COM THRESHOLD 0.50"
    )

    print(
        "=" * 72
    )


    print(
        f"Balanced Accuracy @0.50: "
        f"{diagnostic['balanced_accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy final: "
        f"{selected['balanced_accuracy']:.4f}"
    )

    print(
        f"Recall risco @0.50: "
        f"{diagnostic['recall_risk']:.4f}"
    )

    print(
        f"Recall risco final: "
        f"{selected['recall_risk']:.4f}"
    )

    print(
        f"F1 risco @0.50: "
        f"{diagnostic['f1_risk']:.4f}"
    )

    print(
        f"F1 risco final: "
        f"{selected['f1_risk']:.4f}"
    )


    print(
        f"\nTempo total: "
        f"{elapsed / 60:.2f} min"
    )


    print(
        "\nArquivos gerados:"
    )

    print(
        "  ✓ "
        + str(
            FOLDS_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            THRESHOLD_GRID_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            FINAL_THRESHOLD_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "SELEÇÃO DE THRESHOLD CONCLUÍDA"
    )

    print(
        "=" * 72
    )

    print(
        "NÃO abrir 2024 antes da análise "
        "e do congelamento formal deste threshold."
    )


if __name__ == "__main__":
    main()