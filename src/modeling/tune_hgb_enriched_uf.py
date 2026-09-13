from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.stats import loguniform, randint

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, ParameterSampler

from src.preprocessing.preprocessor import (
    build_preprocessor,
    get_model_features,
)


# ============================================================
# Tech Challenge Fase 3
#
# Tuning principal:
# HistGradientBoostingClassifier
#
# Feature set:
# ENRICHED + UF
#
# Estratégia:
# - somente 2023;
# - 2024 permanece fechado;
# - GroupKFold por município;
# - Random Search reproduzível;
# - configuração atual incluída como candidato 0;
# - PR-AUC de risco = métrica principal;
# - ROC-AUC = métrica complementar;
# - threshold 0.50 = apenas diagnóstico;
# - mede também gap train x validation;
# - checkpoints após CADA fit.
#
# Esta etapa escolhe hiperparâmetros.
# Ela ainda NÃO escolhe threshold operacional.
# ============================================================


RANDOM_STATE = 42
N_SPLITS = 5

N_RANDOM_CANDIDATES = 14

# configuração atual + 14 aleatórias
EXPECTED_CANDIDATES = (
    1
    + N_RANDOM_CANDIDATES
)

EXPECTED_FITS = (
    EXPECTED_CANDIDATES
    * N_SPLITS
)

THRESHOLD = 0.50


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
# Saídas finais
# ============================================================


FOLDS_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_tuning_2023_folds.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_tuning_2023_summary.csv"
)


BEST_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_tuning_2023_best.json"
)


# ============================================================
# Checkpoint temporário
# ============================================================


CHECKPOINT_PATH = (
    REPORTS_PATH
    / ".hgb_tuning_checkpoint.csv"
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
# Carregamento
# ============================================================


def load_2023() -> pd.DataFrame:

    files = sorted(
        DATA_PATH.glob(
            "*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Nenhum Parquet encontrado em: "
            f"{DATA_PATH}"
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

    assert len(df) == 1_502_809

    assert (
        df["aluno_key"]
        .nunique()
        == 1_502_809
    )

    assert int(
        (
            df["alfabetizado"]
            == 0
        ).sum()
    ) == 625_382

    assert int(
        (
            df["alfabetizado"]
            == 1
        ).sum()
    ) == 877_427

    assert (
        df["id_municipio"]
        .isna()
        .sum()
        == 0
    )

    return df


# ============================================================
# Conversão sparse -> dense
# ============================================================


def to_dense(
    matrix,
) -> np.ndarray:

    if sparse.issparse(
        matrix
    ):
        return matrix.toarray()

    return np.asarray(
        matrix
    )


# ============================================================
# Espaço de busca
#
# Random Search:
# - learning_rate e regularização:
#   distribuições logarítmicas;
# - complexidade:
#   profundidade, folhas e min_samples_leaf;
# - número de iterações:
#   quantidade de boosting iterations.
#
# Mantemos early_stopping=False porque a validação oficial
# deste experimento é externa e agrupada por município.
# ============================================================


def build_search_space():

    return {
        "learning_rate": (
            loguniform(
                0.03,
                0.20,
            )
        ),

        "max_iter": (
            randint(
                100,
                301,
            )
        ),

        "max_leaf_nodes": [
            15,
            31,
            63,
        ],

        "max_depth": [
            None,
            6,
            10,
        ],

        "min_samples_leaf": (
            randint(
                20,
                201,
            )
        ),

        "l2_regularization": (
            loguniform(
                0.01,
                10.0,
            )
        ),

        "max_bins": [
            127,
            255,
        ],
    }


# ============================================================
# Configuração atual
#
# Incluída explicitamente para garantir que o Random Search
# seja comparado contra o HGB que já conhecemos.
# ============================================================


def get_current_configuration():

    return {
        "learning_rate": 0.08,
        "max_iter": 150,
        "max_leaf_nodes": 31,
        "max_depth": None,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "max_bins": 255,
    }


# ============================================================
# Normalizar tipos do scipy/numpy para tipos Python
# ============================================================


def normalize_value(
    value,
):

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    return value


def normalize_params(
    params: dict,
) -> dict:

    return {
        key: normalize_value(
            value
        )
        for key, value
        in params.items()
    }


# ============================================================
# Gerar candidatos
# ============================================================


def build_candidates() -> list[dict]:

    candidates = [
        get_current_configuration()
    ]


    sampled = list(
        ParameterSampler(
            param_distributions=(
                build_search_space()
            ),
            n_iter=(
                N_RANDOM_CANDIDATES
            ),
            random_state=(
                RANDOM_STATE
            ),
        )
    )


    for params in sampled:

        candidates.append(
            normalize_params(
                params
            )
        )


    assert (
        len(candidates)
        == EXPECTED_CANDIDATES
    )


    return candidates


# ============================================================
# Criar modelo
# ============================================================


def build_model(
    params: dict,
):

    return (
        HistGradientBoostingClassifier(
            **params,
            early_stopping=False,
            random_state=(
                RANDOM_STATE
            ),
        )
    )


# ============================================================
# Métricas de ranking
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
        "pr_auc_risk": (
            average_precision_score(
                y_risk,
                p_risk,
            )
        ),

        "roc_auc_literate": (
            roc_auc_score(
                y_true,
                p_literate,
            )
        ),
    }


# ============================================================
# Métricas diagnósticas @0.50
# ============================================================


def threshold_metrics(
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


    y_pred_literate = (
        p_literate
        >= THRESHOLD
    ).astype(int)


    y_true_risk = (
        1
        - y_true
    )

    y_pred_risk = (
        1
        - y_pred_literate
    )


    return {
        "accuracy": (
            accuracy_score(
                y_true,
                y_pred_literate,
            )
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred_literate,
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
                y_true,
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
    }


# ============================================================
# Checkpoint
# ============================================================


def load_checkpoint() -> list[dict]:

    if not CHECKPOINT_PATH.exists():

        return []


    checkpoint = pd.read_csv(
        CHECKPOINT_PATH
    )


    return checkpoint.to_dict(
        orient="records"
    )


def save_checkpoint(
    rows: list[dict],
) -> None:

    pd.DataFrame(
        rows
    ).to_csv(
        CHECKPOINT_PATH,
        index=False,
    )


# ============================================================
# Identificador candidato-fold
# ============================================================


def completed_keys(
    rows: list[dict],
) -> set[tuple[int, int]]:

    return {
        (
            int(
                row[
                    "candidate_id"
                ]
            ),
            int(
                row[
                    "fold"
                ]
            ),
        )
        for row in rows
    }


# ============================================================
# Predição classe alfabetizado
# ============================================================


def predict_literate_probability(
    model,
    X: np.ndarray,
) -> np.ndarray:

    probabilities = (
        model.predict_proba(
            X
        )
    )

    positions = np.where(
        model.classes_
        == 1
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
# Execução do Random Search
#
# O preprocessing é ajustado uma vez por fold.
# Todos os candidatos daquele fold usam exatamente a mesma
# representação transformada.
#
# Isso preserva a ausência de leakage e evita repetir o
# preprocessing 15 vezes no mesmo fold.
# ============================================================


def run_search(
    df: pd.DataFrame,
    folds,
    candidates: list[dict],
) -> pd.DataFrame:

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


    rows = (
        load_checkpoint()
    )


    done = completed_keys(
        rows
    )


    if rows:

        print(
            "\n✓ Checkpoint encontrado: "
            f"{len(rows)}/{EXPECTED_FITS} "
            "fits já concluídos."
        )

    else:

        print(
            "\n✓ Nenhum checkpoint anterior."
        )


    # --------------------------------------------------------
    # Loop por fold
    # --------------------------------------------------------

    for fold_number, (
        train_index,
        valid_index,
    ) in enumerate(
        folds,
        start=1,
    ):

        pending_candidates = [
            candidate_id
            for candidate_id
            in range(
                len(candidates)
            )
            if (
                candidate_id,
                fold_number,
            )
            not in done
        ]


        if not pending_candidates:

            print(
                "\n"
                + "=" * 72
            )

            print(
                f"FOLD {fold_number}/{N_SPLITS} "
                "já concluído."
            )

            print(
                "=" * 72
            )

            continue


        print(
            "\n"
            + "=" * 72
        )

        print(
            f"PREPARANDO FOLD "
            f"{fold_number}/{N_SPLITS}"
        )

        print(
            "=" * 72
        )


        # ----------------------------------------------------
        # Auditoria de grupos
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Dados brutos
        # ----------------------------------------------------

        X_train_raw = (
            X.iloc[
                train_index
            ]
        )

        X_valid_raw = (
            X.iloc[
                valid_index
            ]
        )

        y_train = y[
            train_index
        ]

        y_valid = y[
            valid_index
        ]


        # ----------------------------------------------------
        # Fit do preprocessing APENAS no treino deste fold
        # ----------------------------------------------------

        print(
            "\nAjustando preprocessing "
            "do fold..."
        )


        preprocessor = (
            build_preprocessor(
                feature_set=(
                    "enriched"
                ),
                include_uf=True,
            )
        )


        X_train = (
            preprocessor
            .fit_transform(
                X_train_raw,
                y_train,
            )
        )

        X_valid = (
            preprocessor
            .transform(
                X_valid_raw
            )
        )


        transformed_features = len(
            preprocessor
            .get_feature_names_out()
        )


        assert (
            transformed_features
            == 45
        )


        # ----------------------------------------------------
        # HGB exige matriz densa
        # ----------------------------------------------------

        X_train = to_dense(
            X_train
        )

        X_valid = to_dense(
            X_valid
        )


        print(
            f"✓ Features transformadas: "
            f"{transformed_features}"
        )

        print(
            f"✓ Shape treino: "
            f"{X_train.shape}"
        )

        print(
            f"✓ Shape validação: "
            f"{X_valid.shape}"
        )


        # ----------------------------------------------------
        # Candidatos
        # ----------------------------------------------------

        for candidate_id in (
            pending_candidates
        ):

            params = (
                candidates[
                    candidate_id
                ]
            )


            print(
                "\n"
                + "-" * 72
            )

            print(
                f"Candidate "
                f"{candidate_id + 1}/"
                f"{len(candidates)} "
                f"| Fold "
                f"{fold_number}/{N_SPLITS}"
            )

            print(
                "-" * 72
            )


            print(
                json.dumps(
                    params,
                    indent=2,
                    ensure_ascii=False,
                )
            )


            model = build_model(
                params
            )


            start_time = (
                time.perf_counter()
            )


            # ------------------------------------------------
            # Treino
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train,
            )


            # ------------------------------------------------
            # Probabilidades
            # ------------------------------------------------

            train_p_literate = (
                predict_literate_probability(
                    model,
                    X_train,
                )
            )


            valid_p_literate = (
                predict_literate_probability(
                    model,
                    X_valid,
                )
            )


            # ------------------------------------------------
            # Métricas
            # ------------------------------------------------

            train_rank = (
                ranking_metrics(
                    y_train,
                    train_p_literate,
                )
            )


            valid_rank = (
                ranking_metrics(
                    y_valid,
                    valid_p_literate,
                )
            )


            valid_threshold = (
                threshold_metrics(
                    y_valid,
                    valid_p_literate,
                )
            )


            elapsed = (
                time.perf_counter()
                - start_time
            )


            pr_gap = (
                train_rank[
                    "pr_auc_risk"
                ]
                - valid_rank[
                    "pr_auc_risk"
                ]
            )


            roc_gap = (
                train_rank[
                    "roc_auc_literate"
                ]
                - valid_rank[
                    "roc_auc_literate"
                ]
            )


            row = {
                "candidate_id": (
                    candidate_id
                ),

                "fold": (
                    fold_number
                ),

                "feature_set": (
                    "enriched_uf"
                ),

                "model": (
                    "hist_gradient_boosting"
                ),

                "raw_features": (
                    len(FEATURES)
                ),

                "transformed_features": (
                    transformed_features
                ),

                "train_rows": (
                    len(train_index)
                ),

                "validation_rows": (
                    len(valid_index)
                ),

                "train_municipalities": (
                    len(train_groups)
                ),

                "validation_municipalities": (
                    len(valid_groups)
                ),

                "municipality_overlap": (
                    len(overlap)
                ),

                "runtime_seconds": (
                    elapsed
                ),

                "train_pr_auc_risk": (
                    train_rank[
                        "pr_auc_risk"
                    ]
                ),

                "validation_pr_auc_risk": (
                    valid_rank[
                        "pr_auc_risk"
                    ]
                ),

                "pr_auc_gap_train_validation": (
                    pr_gap
                ),

                "train_roc_auc_literate": (
                    train_rank[
                        "roc_auc_literate"
                    ]
                ),

                "validation_roc_auc_literate": (
                    valid_rank[
                        "roc_auc_literate"
                    ]
                ),

                "roc_auc_gap_train_validation": (
                    roc_gap
                ),

                "validation_accuracy": (
                    valid_threshold[
                        "accuracy"
                    ]
                ),

                "validation_balanced_accuracy": (
                    valid_threshold[
                        "balanced_accuracy"
                    ]
                ),

                "validation_precision_risk": (
                    valid_threshold[
                        "precision_risk"
                    ]
                ),

                "validation_recall_risk": (
                    valid_threshold[
                        "recall_risk"
                    ]
                ),

                "validation_f1_risk": (
                    valid_threshold[
                        "f1_risk"
                    ]
                ),

                "validation_recall_literate": (
                    valid_threshold[
                        "recall_literate"
                    ]
                ),

                "validation_actual_risk_rate": (
                    valid_threshold[
                        "actual_risk_rate"
                    ]
                ),

                "validation_predicted_risk_rate": (
                    valid_threshold[
                        "predicted_risk_rate"
                    ]
                ),

                "learning_rate": (
                    params[
                        "learning_rate"
                    ]
                ),

                "max_iter": (
                    params[
                        "max_iter"
                    ]
                ),

                "max_leaf_nodes": (
                    params[
                        "max_leaf_nodes"
                    ]
                ),

                "max_depth": (
                    params[
                        "max_depth"
                    ]
                ),

                "min_samples_leaf": (
                    params[
                        "min_samples_leaf"
                    ]
                ),

                "l2_regularization": (
                    params[
                        "l2_regularization"
                    ]
                ),

                "max_bins": (
                    params[
                        "max_bins"
                    ]
                ),
            }


            rows.append(
                row
            )


            save_checkpoint(
                rows
            )


            done.add(
                (
                    candidate_id,
                    fold_number,
                )
            )


            print(
                f"Train PR-AUC risco: "
                f"{train_rank['pr_auc_risk']:.4f}"
            )

            print(
                f"Validation PR-AUC risco: "
                f"{valid_rank['pr_auc_risk']:.4f}"
            )

            print(
                f"Gap PR-AUC: "
                f"{pr_gap:.4f}"
            )

            print(
                f"Validation ROC-AUC: "
                f"{valid_rank['roc_auc_literate']:.4f}"
            )

            print(
                f"Balanced Accuracy @0.50: "
                f"{valid_threshold['balanced_accuracy']:.4f}"
            )

            print(
                f"Recall risco @0.50: "
                f"{valid_threshold['recall_risk']:.4f}"
            )

            print(
                f"F1 risco @0.50: "
                f"{valid_threshold['f1_risk']:.4f}"
            )

            print(
                f"Tempo: "
                f"{elapsed:.1f}s"
            )

            print(
                "✓ Checkpoint salvo."
            )


            del model
            del train_p_literate
            del valid_p_literate

            gc.collect()


        # ----------------------------------------------------
        # Liberar fold antes de montar o próximo
        # ----------------------------------------------------

        del preprocessor

        del X_train
        del X_valid

        del X_train_raw
        del X_valid_raw

        del y_train
        del y_valid

        gc.collect()


    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "candidate_id",
                "fold",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    expected = (
        EXPECTED_CANDIDATES
        * N_SPLITS
    )


    if len(
        result
    ) != expected:

        raise RuntimeError(
            "Quantidade inesperada de fits. "
            f"Esperado: {expected}. "
            f"Encontrado: {len(result)}."
        )


    return result


# ============================================================
# Agregação por candidato
# ============================================================


def summarize_candidates(
    folds_df: pd.DataFrame,
) -> pd.DataFrame:

    parameter_columns = [
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
        "l2_regularization",
        "max_bins",
    ]


    rows = []


    for candidate_id, group in (
        folds_df.groupby(
            "candidate_id",
            sort=True,
        )
    ):

        first = (
            group.iloc[0]
        )


        row = {
            "candidate_id": (
                int(
                    candidate_id
                )
            ),

            "is_current_configuration": (
                int(
                    candidate_id
                )
                == 0
            ),

            "mean_train_pr_auc_risk": (
                group[
                    "train_pr_auc_risk"
                ]
                .mean()
            ),

            "mean_validation_pr_auc_risk": (
                group[
                    "validation_pr_auc_risk"
                ]
                .mean()
            ),

            "std_validation_pr_auc_risk": (
                group[
                    "validation_pr_auc_risk"
                ]
                .std(
                    ddof=1
                )
            ),

            "mean_pr_auc_gap_train_validation": (
                group[
                    "pr_auc_gap_train_validation"
                ]
                .mean()
            ),

            "mean_train_roc_auc_literate": (
                group[
                    "train_roc_auc_literate"
                ]
                .mean()
            ),

            "mean_validation_roc_auc_literate": (
                group[
                    "validation_roc_auc_literate"
                ]
                .mean()
            ),

            "std_validation_roc_auc_literate": (
                group[
                    "validation_roc_auc_literate"
                ]
                .std(
                    ddof=1
                )
            ),

            "mean_roc_auc_gap_train_validation": (
                group[
                    "roc_auc_gap_train_validation"
                ]
                .mean()
            ),

            "mean_validation_accuracy": (
                group[
                    "validation_accuracy"
                ]
                .mean()
            ),

            "mean_validation_balanced_accuracy": (
                group[
                    "validation_balanced_accuracy"
                ]
                .mean()
            ),

            "mean_validation_precision_risk": (
                group[
                    "validation_precision_risk"
                ]
                .mean()
            ),

            "mean_validation_recall_risk": (
                group[
                    "validation_recall_risk"
                ]
                .mean()
            ),

            "mean_validation_f1_risk": (
                group[
                    "validation_f1_risk"
                ]
                .mean()
            ),

            "mean_validation_recall_literate": (
                group[
                    "validation_recall_literate"
                ]
                .mean()
            ),

            "mean_runtime_seconds": (
                group[
                    "runtime_seconds"
                ]
                .mean()
            ),

            "total_runtime_seconds": (
                group[
                    "runtime_seconds"
                ]
                .sum()
            ),
        }


        for column in (
            parameter_columns
        ):

            value = (
                first[
                    column
                ]
            )

            if pd.isna(
                value
            ):
                value = None

            row[
                column
            ] = value


        rows.append(
            row
        )


    summary = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Ranking oficial:
    #
    # 1. maior PR-AUC risco de validação;
    # 2. maior ROC-AUC;
    # 3. menor gap train-validation PR-AUC.
    # --------------------------------------------------------

    summary = (
        summary
        .sort_values(
            [
                "mean_validation_pr_auc_risk",
                "mean_validation_roc_auc_literate",
                "mean_pr_auc_gap_train_validation",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )


    return summary


# ============================================================
# Melhor configuração
# ============================================================


def save_best_candidate(
    summary_df: pd.DataFrame,
) -> dict:

    best = (
        summary_df.iloc[0]
    )


    def clean(
        value,
    ):

        if pd.isna(
            value
        ):
            return None

        if isinstance(
            value,
            np.integer,
        ):
            return int(value)

        if isinstance(
            value,
            np.floating,
        ):
            return float(value)

        return value


    params = {
        "learning_rate": (
            clean(
                best[
                    "learning_rate"
                ]
            )
        ),

        "max_iter": (
            int(
                best[
                    "max_iter"
                ]
            )
        ),

        "max_leaf_nodes": (
            int(
                best[
                    "max_leaf_nodes"
                ]
            )
        ),

        "max_depth": (
            clean(
                best[
                    "max_depth"
                ]
            )
        ),

        "min_samples_leaf": (
            int(
                best[
                    "min_samples_leaf"
                ]
            )
        ),

        "l2_regularization": (
            clean(
                best[
                    "l2_regularization"
                ]
            )
        ),

        "max_bins": (
            int(
                best[
                    "max_bins"
                ]
            )
        ),
    }


    result = {
        "model": (
            "HistGradientBoostingClassifier"
        ),

        "feature_set": (
            "enriched_uf"
        ),

        "development_year": 2023,

        "selection_metric": (
            "mean_validation_pr_auc_risk"
        ),

        "candidate_id": (
            int(
                best[
                    "candidate_id"
                ]
            )
        ),

        "mean_validation_pr_auc_risk": (
            float(
                best[
                    "mean_validation_pr_auc_risk"
                ]
            )
        ),

        "std_validation_pr_auc_risk": (
            float(
                best[
                    "std_validation_pr_auc_risk"
                ]
            )
        ),

        "mean_validation_roc_auc_literate": (
            float(
                best[
                    "mean_validation_roc_auc_literate"
                ]
            )
        ),

        "mean_pr_auc_gap_train_validation": (
            float(
                best[
                    "mean_pr_auc_gap_train_validation"
                ]
            )
        ),

        "params": params,
    }


    with open(
        BEST_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )


    return result


# ============================================================
# Main
# ============================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "TUNING HISTGRADIENTBOOSTING — 2023"
    )

    print(
        "=" * 72
    )

    print(
        "Feature set: ENRICHED + UF"
    )

    print(
        "2024 permanece fechado."
    )

    print(
        f"Candidatos: "
        f"{EXPECTED_CANDIDATES}"
    )

    print(
        f"GroupKFold: "
        f"{N_SPLITS}"
    )

    print(
        f"Fits esperados: "
        f"{EXPECTED_FITS}"
    )

    print(
        "Métrica principal: PR-AUC de risco."
    )

    print(
        "Threshold 0.50 é apenas diagnóstico."
    )


    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    print(
        "\nCarregando base 2023..."
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
    # Candidatos
    # --------------------------------------------------------

    candidates = (
        build_candidates()
    )


    print(
        "\nConfiguração atual "
        "incluída como candidato 1."
    )


    print(
        "\nEspaço aleatório preparado."
    )


    # --------------------------------------------------------
    # Folds
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


    print(
        f"✓ {len(folds)} folds "
        "por município preparados."
    )


    # --------------------------------------------------------
    # Tuning
    # --------------------------------------------------------

    start_total = (
        time.perf_counter()
    )


    folds_df = run_search(
        df=df,
        folds=folds,
        candidates=candidates,
    )


    total_elapsed = (
        time.perf_counter()
        - start_total
    )


    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    summary_df = (
        summarize_candidates(
            folds_df
        )
    )


    best = (
        save_best_candidate(
            summary_df
        )
    )


    # --------------------------------------------------------
    # Persistência final
    # --------------------------------------------------------

    folds_df.to_csv(
        FOLDS_OUTPUT_PATH,
        index=False,
    )


    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )


    # --------------------------------------------------------
    # Mostrar top 5
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP 5 — HGB"
    )

    print(
        "=" * 72
    )


    display_columns = [
        "rank",
        "candidate_id",
        "is_current_configuration",
        "mean_validation_pr_auc_risk",
        "std_validation_pr_auc_risk",
        "mean_validation_roc_auc_literate",
        "mean_pr_auc_gap_train_validation",
        "mean_validation_balanced_accuracy",
        "mean_validation_recall_risk",
        "mean_validation_f1_risk",
        "total_runtime_seconds",
    ]


    print(
        summary_df[
            display_columns
        ]
        .head(5)
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "MELHOR CONFIGURAÇÃO"
    )

    print(
        "=" * 72
    )


    print(
        json.dumps(
            best,
            indent=2,
            ensure_ascii=False,
        )
    )


    # --------------------------------------------------------
    # Comparação candidato atual x melhor
    # --------------------------------------------------------

    current_row = (
        summary_df[
            summary_df[
                "candidate_id"
            ]
            == 0
        ]
        .iloc[0]
    )


    best_row = (
        summary_df.iloc[0]
    )


    delta_pr = (
        best_row[
            "mean_validation_pr_auc_risk"
        ]
        - current_row[
            "mean_validation_pr_auc_risk"
        ]
    )


    delta_roc = (
        best_row[
            "mean_validation_roc_auc_literate"
        ]
        - current_row[
            "mean_validation_roc_auc_literate"
        ]
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "GANHO DO TUNING"
    )

    print(
        "=" * 72
    )


    print(
        f"PR-AUC atual: "
        f"{current_row['mean_validation_pr_auc_risk']:.4f}"
    )

    print(
        f"PR-AUC melhor: "
        f"{best_row['mean_validation_pr_auc_risk']:.4f}"
    )

    print(
        f"Delta PR-AUC: "
        f"{delta_pr:+.4f}"
    )


    print(
        f"\nROC-AUC atual: "
        f"{current_row['mean_validation_roc_auc_literate']:.4f}"
    )

    print(
        f"ROC-AUC melhor: "
        f"{best_row['mean_validation_roc_auc_literate']:.4f}"
    )

    print(
        f"Delta ROC-AUC: "
        f"{delta_roc:+.4f}"
    )


    print(
        f"\nTempo total desta execução: "
        f"{total_elapsed / 60:.1f} min"
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
            SUMMARY_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            BEST_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )


    # --------------------------------------------------------
    # Só remover checkpoint após gravação final completa
    # --------------------------------------------------------

    if CHECKPOINT_PATH.exists():

        CHECKPOINT_PATH.unlink()


    print(
        "  ✓ Checkpoint temporário removido."
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "TUNING HGB CONCLUÍDO"
    )

    print(
        "=" * 72
    )

    print(
        "Não escolher threshold ainda."
    )

    print(
        "Não utilizar 2024 ainda."
    )

    print(
        "O resultado deve ser analisado antes "
        "do tuning do Random Forest."
    )


if __name__ == "__main__":
    main()