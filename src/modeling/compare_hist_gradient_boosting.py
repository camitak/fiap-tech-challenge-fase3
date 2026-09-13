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
# Comparação de família de modelo:
#
# HistGradientBoostingClassifier
# vs.
# Decision Tree / Random Forest já avaliados
#
# Regras metodológicas:
# - somente 2023;
# - 2024 permanece fechado;
# - ENRICHED + UF já definido como feature set principal;
# - mesmos 5 folds por município;
# - nenhum município aparece simultaneamente em treino/validação;
# - threshold 0.50 somente diagnóstico;
# - PR-AUC risco e ROC-AUC são as métricas principais;
# - RF e DT não são retreinados: resultados anteriores são reutilizados;
# - esta etapa NÃO é tuning final.
# ============================================================


RANDOM_STATE = 42
N_SPLITS = 5
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
# Resultado anterior de DT / RF com ENRICHED + UF
# ============================================================


REFERENCE_SUMMARY_PATH = (
    REPORTS_PATH
    / "uf_comparison_2023_summary.csv"
)


# ============================================================
# Saídas
# ============================================================


FOLDS_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_comparison_2023_folds.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_comparison_2023_summary.csv"
)


MODELS_OUTPUT_PATH = (
    REPORTS_PATH
    / "model_family_comparison_2023.csv"
)


# ============================================================
# Checkpoints
# ============================================================


CHECKPOINT_FOLDS_PATH = (
    REPORTS_PATH
    / ".hgb_checkpoint_folds.csv"
)


CHECKPOINT_OOF_PATH = (
    REPORTS_PATH
    / ".hgb_checkpoint_oof.npy"
)


CHECKPOINT_STATE_PATH = (
    REPORTS_PATH
    / ".hgb_checkpoint_state.json"
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
# Dados
# ============================================================


def load_2023() -> pd.DataFrame:

    files = sorted(
        DATA_PATH.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            "Nenhum arquivo Parquet encontrado em: "
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
    # Reconciliação conhecida da base de desenvolvimento
    # --------------------------------------------------------

    assert len(df) == 1_502_809

    assert (
        df["aluno_key"]
        .nunique()
        == 1_502_809
    )

    assert int(
        (df["alfabetizado"] == 0)
        .sum()
    ) == 625_382

    assert int(
        (df["alfabetizado"] == 1)
        .sum()
    ) == 877_427

    assert (
        df["id_municipio"]
        .isna()
        .sum()
        == 0
    )

    return df


# ============================================================
# Sparse -> dense
#
# HGB não aceita a matriz sparse produzida pelo OHE.
# Mantemos o MESMO preprocessing e apenas convertemos sua
# saída para matriz densa antes do estimador.
# ============================================================


def to_dense(matrix):

    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)


# ============================================================
# Modelo
#
# Configuração inicial conservadora.
# Ainda NÃO é tuning.
#
# early_stopping=False evita criar internamente uma nova
# validação aleatória por aluno, mantendo nossa estratégia
# externa por município como referência de avaliação.
# ============================================================


def build_hgb() -> HistGradientBoostingClassifier:

    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


# ============================================================
# Métricas
# ============================================================


def calculate_metrics(
    y_true: np.ndarray,
    p_literate: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float]:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    p_literate = np.asarray(
        p_literate,
        dtype=float,
    )

    p_risk = (
        1.0
        - p_literate
    )

    y_pred_literate = (
        p_literate
        >= threshold
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
        "accuracy": accuracy_score(
            y_true,
            y_pred_literate,
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred_literate,
            )
        ),

        "roc_auc_literate": (
            roc_auc_score(
                y_true,
                p_literate,
            )
        ),

        "pr_auc_risk": (
            average_precision_score(
                y_true_risk,
                p_risk,
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
# Probabilidade da classe alfabetizado = 1
# ============================================================


def predict_literate_probability(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:

    probabilities = (
        pipeline.predict_proba(X)
    )

    classes = (
        pipeline
        .named_steps["model"]
        .classes_
    )

    positions = np.where(
        classes == 1
    )[0]

    if len(positions) != 1:
        raise RuntimeError(
            "Não foi possível localizar "
            "a classe alfabetizado=1."
        )

    return probabilities[
        :,
        positions[0],
    ]


# ============================================================
# Checkpoint
# ============================================================


def load_checkpoint_rows() -> list[dict]:

    if not CHECKPOINT_FOLDS_PATH.exists():
        return []

    df = pd.read_csv(
        CHECKPOINT_FOLDS_PATH
    )

    return df.to_dict(
        orient="records"
    )


def save_checkpoint_rows(
    rows: list[dict],
) -> None:

    pd.DataFrame(
        rows
    ).to_csv(
        CHECKPOINT_FOLDS_PATH,
        index=False,
    )


def load_or_create_oof(
    n_rows: int,
) -> np.ndarray:

    if CHECKPOINT_OOF_PATH.exists():

        oof = np.load(
            CHECKPOINT_OOF_PATH
        )

        if len(oof) != n_rows:
            raise RuntimeError(
                "Checkpoint OOF incompatível "
                "com a base atual."
            )

        return oof

    return np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )


def save_checkpoint_state(
    fold_number: int,
) -> None:

    state = {
        "model": (
            "hist_gradient_boosting"
        ),

        "last_completed_fold": (
            fold_number
        ),

        "timestamp": (
            pd.Timestamp.now()
            .isoformat()
        ),
    }

    with open(
        CHECKPOINT_STATE_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=2,
        )


def cleanup_checkpoints() -> None:

    for path in [
        CHECKPOINT_FOLDS_PATH,
        CHECKPOINT_OOF_PATH,
        CHECKPOINT_STATE_PATH,
    ]:

        if path.exists():
            path.unlink()


# ============================================================
# Referência dos modelos anteriores
# ============================================================


def load_reference_models() -> pd.DataFrame:

    if not REFERENCE_SUMMARY_PATH.exists():

        raise FileNotFoundError(
            "Resultado anterior não encontrado: "
            f"{REFERENCE_SUMMARY_PATH}"
        )

    reference = pd.read_csv(
        REFERENCE_SUMMARY_PATH
    )

    reference = reference[
        reference["feature_set"]
        == "enriched_uf"
    ].copy()

    expected_models = {
        "decision_tree",
        "random_forest",
    }

    actual_models = set(
        reference["model"]
    )

    if not expected_models.issubset(
        actual_models
    ):

        raise RuntimeError(
            "O arquivo de referência não contém "
            "Decision Tree e Random Forest."
        )

    return reference


# ============================================================
# Treinamento
# ============================================================


def run_hgb(
    df: pd.DataFrame,
    folds,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    X = df[
        FEATURES
    ]

    y = (
        df["alfabetizado"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        df["id_municipio"]
        .astype(str)
    )


    checkpoint_rows = (
        load_checkpoint_rows()
    )


    completed_folds = {
        int(row["fold"])
        for row in checkpoint_rows
    }


    oof_p_literate = (
        load_or_create_oof(
            len(df)
        )
    )


    transformed_features = None

    if checkpoint_rows:

        transformed_features = int(
            checkpoint_rows[0][
                "transformed_features"
            ]
        )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "HIST_GRADIENT_BOOSTING | ENRICHED + UF"
    )

    print(
        "=" * 72
    )

    print(
        f"Features brutas: "
        f"{len(FEATURES)}"
    )


    for fold_number, (
        train_index,
        valid_index,
    ) in enumerate(
        folds,
        start=1,
    ):

        if fold_number in completed_folds:

            print(
                f"\nFold {fold_number}/{N_SPLITS} "
                "já concluído — checkpoint encontrado."
            )

            continue


        print(
            f"\nIniciando Fold "
            f"{fold_number}/{N_SPLITS}..."
        )


        start_time = (
            time.perf_counter()
        )


        # ----------------------------------------------------
        # Auditoria dos grupos
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


        # ----------------------------------------------------
        # Dados
        # ----------------------------------------------------

        X_train = X.iloc[
            train_index
        ]

        X_valid = X.iloc[
            valid_index
        ]

        y_train = y[
            train_index
        ]

        y_valid = y[
            valid_index
        ]


        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        feature_set="enriched",
                        include_uf=True,
                    ),
                ),

                (
                    "to_dense",
                    FunctionTransformer(
                        to_dense,
                        validate=False,
                    ),
                ),

                (
                    "model",
                    build_hgb(),
                ),
            ]
        )


        pipeline.fit(
            X_train,
            y_train,
        )


        if transformed_features is None:

            transformed_features = len(
                pipeline
                .named_steps[
                    "preprocessor"
                ]
                .get_feature_names_out()
            )


        # ----------------------------------------------------
        # Predição
        # ----------------------------------------------------

        p_literate = (
            predict_literate_probability(
                pipeline,
                X_valid,
            )
        )


        oof_p_literate[
            valid_index
        ] = (
            p_literate.astype(
                np.float32
            )
        )


        metrics = calculate_metrics(
            y_valid,
            p_literate,
            threshold=THRESHOLD,
        )


        elapsed = (
            time.perf_counter()
            - start_time
        )


        row = {
            "feature_set": (
                "enriched_uf"
            ),

            "model": (
                "hist_gradient_boosting"
            ),

            "fold": (
                fold_number
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

            **metrics,
        }


        checkpoint_rows.append(
            row
        )


        # ----------------------------------------------------
        # Persistir logo após cada fold
        # ----------------------------------------------------

        save_checkpoint_rows(
            checkpoint_rows
        )

        np.save(
            CHECKPOINT_OOF_PATH,
            oof_p_literate,
        )

        save_checkpoint_state(
            fold_number
        )


        print(
            f"  Treino: "
            f"{len(train_index):,}"
        )

        print(
            f"  Validação: "
            f"{len(valid_index):,}"
        )

        print(
            f"  Municípios treino: "
            f"{len(train_groups):,}"
        )

        print(
            f"  Municípios validação: "
            f"{len(valid_groups):,}"
        )

        print(
            f"  Features transformadas: "
            f"{transformed_features}"
        )

        print(
            f"  PR-AUC risco: "
            f"{metrics['pr_auc_risk']:.4f}"
        )

        print(
            f"  ROC-AUC: "
            f"{metrics['roc_auc_literate']:.4f}"
        )

        print(
            f"  Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Precision risco @0.50: "
            f"{metrics['precision_risk']:.4f}"
        )

        print(
            f"  Recall risco @0.50: "
            f"{metrics['recall_risk']:.4f}"
        )

        print(
            f"  F1 risco @0.50: "
            f"{metrics['f1_risk']:.4f}"
        )

        print(
            f"  Tempo: "
            f"{elapsed:.1f}s"
        )

        print(
            "  ✓ Checkpoint salvo."
        )


        del pipeline
        del X_train
        del X_valid
        del p_literate

        gc.collect()


    # ========================================================
    # OOF global
    # ========================================================

    if np.isnan(
        oof_p_literate
    ).any():

        missing = int(
            np.isnan(
                oof_p_literate
            ).sum()
        )

        raise RuntimeError(
            "OOF incompleto. "
            f"Predições ausentes: {missing:,}"
        )


    oof_metrics = calculate_metrics(
        y,
        oof_p_literate,
        threshold=THRESHOLD,
    )


    folds_df = (
        pd.DataFrame(
            checkpoint_rows
        )
        .sort_values(
            "fold"
        )
        .reset_index(
            drop=True
        )
    )


    summary = {
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
    }


    for metric_name, value in (
        oof_metrics.items()
    ):

        summary[
            f"oof_{metric_name}"
        ] = value


    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc_literate",
        "pr_auc_risk",
        "precision_risk",
        "recall_risk",
        "f1_risk",
        "recall_literate",
        "actual_risk_rate",
        "predicted_risk_rate",
    ]


    for metric in metric_columns:

        summary[
            f"fold_mean_{metric}"
        ] = (
            folds_df[
                metric
            ]
            .mean()
        )

        summary[
            f"fold_std_{metric}"
        ] = (
            folds_df[
                metric
            ]
            .std(
                ddof=1
            )
        )


    summary[
        "total_runtime_seconds"
    ] = (
        folds_df[
            "runtime_seconds"
        ]
        .sum()
    )


    return (
        folds_df,
        summary,
    )


# ============================================================
# Comparação entre famílias
# ============================================================


def build_model_family_comparison(
    reference: pd.DataFrame,
    hgb_summary: dict,
) -> pd.DataFrame:

    columns = [
        "feature_set",
        "model",
        "raw_features",
        "transformed_features",
        "oof_accuracy",
        "oof_balanced_accuracy",
        "oof_roc_auc_literate",
        "oof_pr_auc_risk",
        "oof_precision_risk",
        "oof_recall_risk",
        "oof_f1_risk",
        "oof_recall_literate",
        "oof_actual_risk_rate",
        "oof_predicted_risk_rate",
        "fold_mean_roc_auc_literate",
        "fold_std_roc_auc_literate",
        "fold_mean_pr_auc_risk",
        "fold_std_pr_auc_risk",
        "total_runtime_seconds",
    ]


    reference = (
        reference[
            columns
        ]
        .copy()
    )


    hgb_df = pd.DataFrame(
        [hgb_summary]
    )[
        columns
    ]


    comparison = pd.concat(
        [
            reference,
            hgb_df,
        ],
        ignore_index=True,
    )


    comparison = (
        comparison
        .sort_values(
            [
                "oof_pr_auc_risk",
                "oof_roc_auc_literate",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    comparison[
        "rank_pr_auc_risk"
    ] = (
        comparison[
            "oof_pr_auc_risk"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )


    return comparison


# ============================================================
# Main
# ============================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "COMPARAÇÃO DE FAMÍLIAS DE MODELO — 2023"
    )

    print(
        "=" * 72
    )

    print(
        "Feature set: ENRICHED + UF"
    )

    print(
        "Somente desenvolvimento 2023."
    )

    print(
        "2024 permanece fechado."
    )

    print(
        "HistGradientBoosting ainda sem tuning final."
    )


    # --------------------------------------------------------
    # Referências anteriores
    # --------------------------------------------------------

    reference = (
        load_reference_models()
    )

    print(
        "\n✓ Resultados anteriores de "
        "Decision Tree e Random Forest localizados."
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
    # Mesmos folds GroupKFold
    # --------------------------------------------------------

    y = (
        df["alfabetizado"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        df["id_municipio"]
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


    print(
        f"✓ GroupKFold: "
        f"{len(folds)} folds"
    )


    # --------------------------------------------------------
    # HGB
    # --------------------------------------------------------

    folds_df, hgb_summary = (
        run_hgb(
            df=df,
            folds=folds,
        )
    )


    # --------------------------------------------------------
    # Persistência
    # --------------------------------------------------------

    hgb_summary_df = pd.DataFrame(
        [hgb_summary]
    )


    comparison_df = (
        build_model_family_comparison(
            reference,
            hgb_summary,
        )
    )


    folds_df.to_csv(
        FOLDS_OUTPUT_PATH,
        index=False,
    )


    hgb_summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )


    comparison_df.to_csv(
        MODELS_OUTPUT_PATH,
        index=False,
    )


    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "HGB — OOF GLOBAL"
    )

    print(
        "=" * 72
    )


    print(
        f"PR-AUC risco: "
        f"{hgb_summary['oof_pr_auc_risk']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{hgb_summary['oof_roc_auc_literate']:.4f}"
    )

    print(
        f"Balanced Accuracy @0.50: "
        f"{hgb_summary['oof_balanced_accuracy']:.4f}"
    )

    print(
        f"Precision risco @0.50: "
        f"{hgb_summary['oof_precision_risk']:.4f}"
    )

    print(
        f"Recall risco @0.50: "
        f"{hgb_summary['oof_recall_risk']:.4f}"
    )

    print(
        f"F1 risco @0.50: "
        f"{hgb_summary['oof_f1_risk']:.4f}"
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "COMPARAÇÃO DE FAMÍLIAS"
    )

    print(
        "=" * 72
    )


    display_columns = [
        "model",
        "oof_pr_auc_risk",
        "oof_roc_auc_literate",
        "oof_balanced_accuracy",
        "oof_precision_risk",
        "oof_recall_risk",
        "oof_f1_risk",
        "rank_pr_auc_risk",
    ]


    print(
        comparison_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nArquivos gerados:"
    )

    print(
        "  ✓ "
        + str(
            FOLDS_OUTPUT_PATH.relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            SUMMARY_OUTPUT_PATH.relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            MODELS_OUTPUT_PATH.relative_to(
                PROJECT_ROOT
            )
        )
    )


    cleanup_checkpoints()

    print(
        "  ✓ Checkpoints temporários removidos."
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "COMPARAÇÃO CONCLUÍDA"
    )

    print(
        "=" * 72
    )

    print(
        "Nenhuma decisão de modelo campeão "
        "deve ser tomada antes da análise "
        "dos resultados."
    )

    print(
        "2024 continua reservado para "
        "avaliação temporal final."
    )


if __name__ == "__main__":
    main()