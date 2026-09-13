from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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
# RandomForestClassifier
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
# - threshold 0.50 = somente diagnóstico;
# - checkpoint depois de cada fit;
# - preprocessing ajustado apenas no treino de cada fold;
# - preprocessing reutilizado entre candidatos do mesmo fold.
#
# Esta etapa escolhe hiperparâmetros.
# Ela NÃO escolhe o threshold operacional.
# ============================================================


RANDOM_STATE = 42
N_SPLITS = 5

N_RANDOM_CANDIDATES = 10

EXPECTED_CANDIDATES = (
    1
    + N_RANDOM_CANDIDATES
)

EXPECTED_FITS = (
    EXPECTED_CANDIDATES
    * N_SPLITS
)

THRESHOLD = 0.50

# ------------------------------------------------------------
# O Random Forest é caro para gerar predict_proba sobre
# 1,2 milhão de linhas após cada fit.
#
# Para diagnóstico de overfitting, usamos uma amostra fixa do
# conjunto de treino de cada fold.
#
# A seleção oficial do modelo NÃO depende dessa amostra.
# ------------------------------------------------------------

TRAIN_EVAL_SAMPLE_SIZE = 250_000


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
    / "rf_tuning_2023_folds.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "rf_tuning_2023_summary.csv"
)


BEST_OUTPUT_PATH = (
    REPORTS_PATH
    / "rf_tuning_2023_best.json"
)


# ============================================================
# Checkpoint temporário
# ============================================================


CHECKPOINT_PATH = (
    REPORTS_PATH
    / ".rf_tuning_checkpoint.csv"
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
# Espaço de busca
#
# Espaço propositalmente amplo, mas limitado para não tornar
# a execução inviável localmente.
# ============================================================


def build_search_space():

    return {
        "n_estimators": [
            150,
            200,
            250,
            300,
        ],

        "max_depth": [
            8,
            12,
            16,
            None,
        ],

        "min_samples_split": [
            50,
            100,
            200,
            400,
        ],

        "min_samples_leaf": [
            20,
            50,
            100,
            150,
            200,
        ],

        "max_features": [
            "sqrt",
            "log2",
            0.5,
        ],

        "class_weight": [
            None,
            "balanced",
            "balanced_subsample",
        ],

        "criterion": [
            "gini",
            "entropy",
        ],
    }


# ============================================================
# Configuração atual do RF
#
# É inserida explicitamente para termos comparação direta.
# ============================================================


def get_current_configuration():

    return {
        "n_estimators": 150,
        "max_depth": 12,
        "min_samples_split": 100,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
        "class_weight": None,
        "criterion": "gini",
    }


# ============================================================
# Normalização
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
# Candidatos
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

        normalized = (
            normalize_params(
                params
            )
        )

        # Evitar duplicar explicitamente a configuração atual.
        if (
            normalized
            == get_current_configuration()
        ):
            continue

        candidates.append(
            normalized
        )


    # Caso o sampler tenha sorteado exatamente a configuração
    # atual, completar deterministicamente com novas amostras.
    extra_seed = (
        RANDOM_STATE
        + 1
    )

    while len(
        candidates
    ) < EXPECTED_CANDIDATES:

        extra = next(
            iter(
                ParameterSampler(
                    param_distributions=(
                        build_search_space()
                    ),
                    n_iter=1,
                    random_state=(
                        extra_seed
                    ),
                )
            )
        )

        extra_seed += 1

        extra = normalize_params(
            extra
        )

        if extra not in candidates:

            candidates.append(
                extra
            )


    candidates = candidates[
        :EXPECTED_CANDIDATES
    ]


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
) -> RandomForestClassifier:

    return RandomForestClassifier(
        **params,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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
# Probabilidade alfabetizado=1
# ============================================================


def predict_literate_probability(
    model,
    X,
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
# Amostra fixa de treino para diagnóstico de overfitting
# ============================================================


def get_train_evaluation_indices(
    n_rows: int,
    fold_number: int,
) -> np.ndarray:

    sample_size = min(
        TRAIN_EVAL_SAMPLE_SIZE,
        n_rows,
    )


    rng = np.random.default_rng(
        RANDOM_STATE
        + fold_number
    )


    indices = rng.choice(
        n_rows,
        size=sample_size,
        replace=False,
    )


    return np.sort(
        indices
    )


# ============================================================
# Execução
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


    rows = load_checkpoint()

    done = completed_keys(
        rows
    )


    if rows:

        print(
            "\n✓ Checkpoint encontrado: "
            f"{len(rows)}/{EXPECTED_FITS} "
            "fits concluídos."
        )

    else:

        print(
            "\n✓ Nenhum checkpoint anterior."
        )


    # ========================================================
    # Loop pelos folds
    # ========================================================

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
        # Auditoria de municípios
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
        # Preprocessing
        #
        # Fit SOMENTE no treino do fold.
        # Reutilizado para todos os candidatos do fold.
        # ----------------------------------------------------

        print(
            "\nAjustando preprocessing "
            "do fold..."
        )


        preprocessor = (
            build_preprocessor(
                feature_set="enriched",
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

        print(
            "✓ Matriz sparse preservada "
            "para Random Forest."
        )


        # ----------------------------------------------------
        # Amostra de treino fixa neste fold
        # ----------------------------------------------------

        train_eval_positions = (
            get_train_evaluation_indices(
                n_rows=(
                    len(train_index)
                ),
                fold_number=(
                    fold_number
                ),
            )
        )


        X_train_eval = (
            X_train[
                train_eval_positions
            ]
        )

        y_train_eval = (
            y_train[
                train_eval_positions
            ]
        )


        print(
            "✓ Amostra de treino para "
            "diagnóstico: "
            f"{len(train_eval_positions):,}"
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
            # Fit
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train,
            )


            # ------------------------------------------------
            # Probabilidade de validação
            # ------------------------------------------------

            valid_p_literate = (
                predict_literate_probability(
                    model,
                    X_valid,
                )
            )


            # ------------------------------------------------
            # Probabilidade em amostra de treino
            # ------------------------------------------------

            train_p_literate = (
                predict_literate_probability(
                    model,
                    X_train_eval,
                )
            )


            # ------------------------------------------------
            # Métricas
            # ------------------------------------------------

            train_rank = (
                ranking_metrics(
                    y_train_eval,
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


            estimated_pr_gap = (
                train_rank[
                    "pr_auc_risk"
                ]
                - valid_rank[
                    "pr_auc_risk"
                ]
            )


            estimated_roc_gap = (
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
                    "random_forest"
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

                "train_evaluation_rows": (
                    len(
                        train_eval_positions
                    )
                ),

                "runtime_seconds": (
                    elapsed
                ),

                # --------------------------------------------
                # Train = amostra diagnóstica
                # --------------------------------------------

                "train_sample_pr_auc_risk": (
                    train_rank[
                        "pr_auc_risk"
                    ]
                ),

                "validation_pr_auc_risk": (
                    valid_rank[
                        "pr_auc_risk"
                    ]
                ),

                "estimated_pr_auc_gap_train_validation": (
                    estimated_pr_gap
                ),

                "train_sample_roc_auc_literate": (
                    train_rank[
                        "roc_auc_literate"
                    ]
                ),

                "validation_roc_auc_literate": (
                    valid_rank[
                        "roc_auc_literate"
                    ]
                ),

                "estimated_roc_auc_gap_train_validation": (
                    estimated_roc_gap
                ),

                # --------------------------------------------
                # Threshold 0.50 — diagnóstico apenas
                # --------------------------------------------

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

                # --------------------------------------------
                # Hiperparâmetros
                # --------------------------------------------

                "n_estimators": (
                    params[
                        "n_estimators"
                    ]
                ),

                "max_depth": (
                    params[
                        "max_depth"
                    ]
                ),

                "min_samples_split": (
                    params[
                        "min_samples_split"
                    ]
                ),

                "min_samples_leaf": (
                    params[
                        "min_samples_leaf"
                    ]
                ),

                "max_features": (
                    params[
                        "max_features"
                    ]
                ),

                "class_weight": (
                    params[
                        "class_weight"
                    ]
                ),

                "criterion": (
                    params[
                        "criterion"
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
                "Train sample PR-AUC risco: "
                f"{train_rank['pr_auc_risk']:.4f}"
            )

            print(
                "Validation PR-AUC risco: "
                f"{valid_rank['pr_auc_risk']:.4f}"
            )

            print(
                "Gap PR-AUC estimado: "
                f"{estimated_pr_gap:.4f}"
            )

            print(
                "Validation ROC-AUC: "
                f"{valid_rank['roc_auc_literate']:.4f}"
            )

            print(
                "Balanced Accuracy @0.50: "
                f"{valid_threshold['balanced_accuracy']:.4f}"
            )

            print(
                "Recall risco @0.50: "
                f"{valid_threshold['recall_risk']:.4f}"
            )

            print(
                "F1 risco @0.50: "
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
            del valid_p_literate
            del train_p_literate

            gc.collect()


        # ----------------------------------------------------
        # Liberar memória ao terminar o fold
        # ----------------------------------------------------

        del preprocessor

        del X_train
        del X_valid

        del X_train_eval
        del y_train_eval

        del X_train_raw
        del X_valid_raw

        del y_train
        del y_valid

        del train_eval_positions

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


    if len(
        result
    ) != EXPECTED_FITS:

        raise RuntimeError(
            "Quantidade inesperada de fits. "
            f"Esperado: {EXPECTED_FITS}. "
            f"Encontrado: {len(result)}."
        )


    return result


# ============================================================
# Resumo dos candidatos
# ============================================================


def summarize_candidates(
    folds_df: pd.DataFrame,
) -> pd.DataFrame:

    parameter_columns = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "criterion",
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

            "mean_train_sample_pr_auc_risk": (
                group[
                    "train_sample_pr_auc_risk"
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

            "mean_estimated_pr_auc_gap": (
                group[
                    "estimated_pr_auc_gap_train_validation"
                ]
                .mean()
            ),

            "mean_train_sample_roc_auc_literate": (
                group[
                    "train_sample_roc_auc_literate"
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

            "mean_estimated_roc_auc_gap": (
                group[
                    "estimated_roc_auc_gap_train_validation"
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

            value = first[
                column
            ]

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
    # Ranking oficial
    #
    # 1. maior PR-AUC de validação
    # 2. maior ROC-AUC de validação
    # 3. menor variabilidade de PR-AUC entre folds
    #
    # O gap estimado de treino NÃO participa do ranking,
    # pois foi calculado em uma amostra de treino.
    # --------------------------------------------------------

    summary = (
        summary
        .sort_values(
            [
                "mean_validation_pr_auc_risk",
                "mean_validation_roc_auc_literate",
                "std_validation_pr_auc_risk",
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
# Melhor candidato
# ============================================================


def save_best_candidate(
    summary_df: pd.DataFrame,
    candidates: list[dict],
) -> dict:

    best = (
        summary_df.iloc[0]
    )


    candidate_id = int(
        best[
            "candidate_id"
        ]
    )


    # Utilizamos o dicionário original para preservar tipos:
    # None, float, strings etc.
    params = candidates[
        candidate_id
    ]


    result = {
        "model": (
            "RandomForestClassifier"
        ),

        "feature_set": (
            "enriched_uf"
        ),

        "development_year": 2023,

        "selection_metric": (
            "mean_validation_pr_auc_risk"
        ),

        "candidate_id": (
            candidate_id
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

        "mean_estimated_pr_auc_gap": (
            float(
                best[
                    "mean_estimated_pr_auc_gap"
                ]
            )
        ),

        "train_gap_note": (
            "Train metrics were calculated on a fixed "
            "250000-row training sample per fold and are "
            "diagnostic only."
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
        "TUNING RANDOM FOREST — 2023"
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
        "Métrica principal: "
        "PR-AUC de risco."
    )

    print(
        "Threshold 0.50 é "
        "somente diagnóstico."
    )

    print(
        "Métricas de treino usam amostra "
        "fixa e não participam do ranking."
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
        "\n✓ Configuração atual incluída "
        "como candidato 1."
    )

    print(
        f"✓ {N_RANDOM_CANDIDATES} "
        "configurações aleatórias preparadas."
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


    best = save_best_candidate(
        summary_df,
        candidates,
    )


    # --------------------------------------------------------
    # Persistência
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
    # Top 5
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP 5 — RANDOM FOREST"
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
        "mean_estimated_pr_auc_gap",
        "mean_validation_balanced_accuracy",
        "mean_validation_precision_risk",
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


    # --------------------------------------------------------
    # Melhor
    # --------------------------------------------------------

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
    # Ganho sobre configuração atual
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
        "PR-AUC atual: "
        f"{current_row['mean_validation_pr_auc_risk']:.4f}"
    )

    print(
        "PR-AUC melhor: "
        f"{best_row['mean_validation_pr_auc_risk']:.4f}"
    )

    print(
        f"Delta PR-AUC: "
        f"{delta_pr:+.4f}"
    )


    print(
        "\nROC-AUC atual: "
        f"{current_row['mean_validation_roc_auc_literate']:.4f}"
    )

    print(
        "ROC-AUC melhor: "
        f"{best_row['mean_validation_roc_auc_literate']:.4f}"
    )

    print(
        f"Delta ROC-AUC: "
        f"{delta_roc:+.4f}"
    )


    print(
        "\nTempo total desta execução: "
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
    # Checkpoint só some depois do sucesso completo
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
        "TUNING RANDOM FOREST CONCLUÍDO"
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
        "Próxima etapa: comparar RF otimizado "
        "com HGB otimizado."
    )


if __name__ == "__main__":
    main()