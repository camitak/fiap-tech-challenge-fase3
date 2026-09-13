from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.preprocessor import (
    build_preprocessor,
    get_model_features,
)


# ============================================================
# Tech Challenge Fase 3
# Experimento controlado:
# ENRICHED x ENRICHED + UF
#
# Objetivo:
# avaliar se sigla_uf adiciona capacidade preditiva ao
# conjunto enriquecido já aprovado.
#
# Regras:
# - somente 2023;
# - 2024 permanece fechado;
# - mesmos 5 folds por município;
# - mesmos modelos;
# - mesmos hiperparâmetros;
# - threshold 0.50 apenas diagnóstico;
# - referência ENRICHED carregada do experimento anterior;
# - somente ENRICHED + UF é retreinado.
#
# O script possui checkpoint por fold.
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
# Artefatos do experimento anterior
# ============================================================


PREVIOUS_SUMMARY_PATH = (
    REPORTS_PATH
    / "feature_set_comparison_2023_summary.csv"
)


# ============================================================
# Saídas deste experimento
# ============================================================


FOLDS_OUTPUT_PATH = (
    REPORTS_PATH
    / "uf_comparison_2023_folds.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "uf_comparison_2023_summary.csv"
)


DELTAS_OUTPUT_PATH = (
    REPORTS_PATH
    / "uf_comparison_2023_deltas.csv"
)


# ============================================================
# Checkpoints temporários
# ============================================================


CHECKPOINT_FOLDS_PATH = (
    REPORTS_PATH
    / ".uf_comparison_checkpoint_folds.csv"
)


CHECKPOINT_STATE_PATH = (
    REPORTS_PATH
    / ".uf_comparison_checkpoint_state.json"
)


def get_oof_checkpoint_path(
    model_name: str,
) -> Path:
    return (
        REPORTS_PATH
        / f".uf_comparison_oof_{model_name}.npy"
    )


# ============================================================
# Features
# ============================================================


ENRICHED_FEATURES = (
    get_model_features(
        feature_set="enriched",
        include_uf=False,
    )
)


ENRICHED_UF_FEATURES = (
    get_model_features(
        feature_set="enriched",
        include_uf=True,
    )
)


CONTROL_COLUMNS = [
    "aluno_key",
    "id_municipio",
    "alfabetizado",
]


READ_COLUMNS = list(
    dict.fromkeys(
        CONTROL_COLUMNS
        + ENRICHED_UF_FEATURES
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
# Modelos
#
# Mantemos os mesmos hiperparâmetros da comparação anterior.
# ============================================================


def build_models():

    return {
        "decision_tree": (
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=10,
                min_samples_split=2000,
                min_samples_leaf=500,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),

        "random_forest": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=100,
                min_samples_leaf=50,
                max_features="sqrt",
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


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
# Probabilidade da classe alfabetizado
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


def load_checkpoint_folds() -> pd.DataFrame:

    if not CHECKPOINT_FOLDS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        CHECKPOINT_FOLDS_PATH
    )


def save_checkpoint_folds(
    rows: list[dict],
) -> None:

    pd.DataFrame(
        rows
    ).to_csv(
        CHECKPOINT_FOLDS_PATH,
        index=False,
    )


def save_checkpoint_state(
    model_name: str,
    fold_number: int,
) -> None:

    state = {
        "model": model_name,
        "last_completed_fold": fold_number,
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


def load_or_create_oof(
    model_name: str,
    n_rows: int,
) -> np.ndarray:

    path = get_oof_checkpoint_path(
        model_name
    )

    if path.exists():

        oof = np.load(
            path
        )

        if len(oof) != n_rows:
            raise RuntimeError(
                "Checkpoint OOF incompatível "
                f"para {model_name}."
            )

        return oof

    return np.full(
        n_rows,
        np.nan,
        dtype=np.float32,
    )


def save_oof_checkpoint(
    model_name: str,
    oof: np.ndarray,
) -> None:

    np.save(
        get_oof_checkpoint_path(
            model_name
        ),
        oof,
    )


def cleanup_checkpoints() -> None:

    paths = [
        CHECKPOINT_FOLDS_PATH,
        CHECKPOINT_STATE_PATH,
    ]

    for model_name in [
        "decision_tree",
        "random_forest",
    ]:
        paths.append(
            get_oof_checkpoint_path(
                model_name
            )
        )

    for path in paths:

        if path.exists():
            path.unlink()


# ============================================================
# Validação do experimento anterior
# ============================================================


def load_previous_enriched_summary() -> pd.DataFrame:

    if not PREVIOUS_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Resumo do experimento anterior "
            "não encontrado em: "
            f"{PREVIOUS_SUMMARY_PATH}"
        )

    previous = pd.read_csv(
        PREVIOUS_SUMMARY_PATH
    )

    enriched = previous[
        previous["feature_set"]
        == "enriched"
    ].copy()

    required_models = {
        "decision_tree",
        "random_forest",
    }

    actual_models = set(
        enriched["model"]
    )

    if not required_models.issubset(
        actual_models
    ):
        raise RuntimeError(
            "O resumo anterior não contém "
            "os modelos enriched esperados."
        )

    return enriched


# ============================================================
# Execução ENRICHED + UF
# ============================================================


def run_model(
    *,
    df: pd.DataFrame,
    folds,
    model_name: str,
    estimator,
    checkpoint_rows: list[dict],
) -> tuple[
    list[dict],
    dict,
]:

    features = ENRICHED_UF_FEATURES

    X = df[
        features
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

    oof_p_literate = (
        load_or_create_oof(
            model_name,
            len(df),
        )
    )

    existing_for_model = {
        int(row["fold"])
        for row in checkpoint_rows
        if row["model"] == model_name
    }

    model_rows = [
        row
        for row in checkpoint_rows
        if row["model"] == model_name
    ]

    transformed_features = None

    if model_rows:
        transformed_features = int(
            model_rows[0][
                "transformed_features"
            ]
        )


    print(
        "\n"
        + "=" * 72
    )

    print(
        f"{model_name.upper()} | "
        "ENRICHED + UF"
    )

    print(
        "=" * 72
    )

    print(
        f"Features brutas: "
        f"{len(features)}"
    )


    for fold_number, (
        train_index,
        valid_index,
    ) in enumerate(
        folds,
        start=1,
    ):

        if fold_number in existing_for_model:

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
                    "model",
                    clone(
                        estimator
                    ),
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
        # Predições
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

            "model": model_name,

            "fold": fold_number,

            "raw_features": (
                len(features)
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

        model_rows.append(
            row
        )


        # ----------------------------------------------------
        # Salvar imediatamente após o fold
        # ----------------------------------------------------

        save_oof_checkpoint(
            model_name,
            oof_p_literate,
        )

        save_checkpoint_folds(
            checkpoint_rows
        )

        save_checkpoint_state(
            model_name,
            fold_number,
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
            "OOF incompleto após os folds. "
            f"Registros sem predição: {missing:,}"
        )


    oof_metrics = (
        calculate_metrics(
            y,
            oof_p_literate,
            threshold=THRESHOLD,
        )
    )


    model_rows_df = pd.DataFrame(
        model_rows
    ).sort_values(
        "fold"
    )


    summary = {
        "feature_set": (
            "enriched_uf"
        ),

        "model": model_name,

        "raw_features": (
            len(features)
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
            model_rows_df[
                metric
            ]
            .mean()
        )

        summary[
            f"fold_std_{metric}"
        ] = (
            model_rows_df[
                metric
            ]
            .std(
                ddof=1
            )
        )


    summary[
        "total_runtime_seconds"
    ] = (
        model_rows_df[
            "runtime_seconds"
        ]
        .sum()
    )


    print(
        "\nOOF GLOBAL"
    )

    print(
        f"  PR-AUC risco: "
        f"{oof_metrics['pr_auc_risk']:.4f}"
    )

    print(
        f"  ROC-AUC: "
        f"{oof_metrics['roc_auc_literate']:.4f}"
    )

    print(
        f"  Balanced Accuracy @0.50: "
        f"{oof_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Recall risco @0.50: "
        f"{oof_metrics['recall_risk']:.4f}"
    )

    print(
        f"  F1 risco @0.50: "
        f"{oof_metrics['f1_risk']:.4f}"
    )


    return (
        model_rows,
        summary,
    )


# ============================================================
# Construção da comparação
# ============================================================


def build_comparison_summary(
    previous_enriched: pd.DataFrame,
    uf_summary_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    metric_columns = [
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
    ]


    for model_name in [
        "decision_tree",
        "random_forest",
    ]:

        previous_row = (
            previous_enriched[
                previous_enriched[
                    "model"
                ]
                == model_name
            ]
            .iloc[0]
        )

        uf_row = (
            uf_summary_df[
                uf_summary_df[
                    "model"
                ]
                == model_name
            ]
            .iloc[0]
        )


        base_row = {
            "model": model_name,

            "enriched_raw_features": (
                int(
                    previous_row[
                        "raw_features"
                    ]
                )
            ),

            "enriched_uf_raw_features": (
                int(
                    uf_row[
                        "raw_features"
                    ]
                )
            ),

            "enriched_transformed_features": (
                int(
                    previous_row[
                        "transformed_features"
                    ]
                )
            ),

            "enriched_uf_transformed_features": (
                int(
                    uf_row[
                        "transformed_features"
                    ]
                )
            ),
        }


        for metric in metric_columns:

            enriched_value = float(
                previous_row[
                    metric
                ]
            )

            uf_value = float(
                uf_row[
                    metric
                ]
            )

            base_row[
                f"enriched_{metric}"
            ] = enriched_value

            base_row[
                f"enriched_uf_{metric}"
            ] = uf_value

            base_row[
                f"delta_{metric}"
            ] = (
                uf_value
                - enriched_value
            )


        rows.append(
            base_row
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "COMPARAÇÃO ENRICHED x ENRICHED + UF — 2023"
    )

    print(
        "=" * 72
    )

    print(
        "Somente desenvolvimento 2023."
    )

    print(
        "2024 permanece fechado."
    )

    print(
        "O ENRICHED anterior será reutilizado "
        "a partir dos CSVs já produzidos."
    )


    # --------------------------------------------------------
    # Referência anterior
    # --------------------------------------------------------

    previous_enriched = (
        load_previous_enriched_summary()
    )

    print(
        "\n✓ Resultado ENRICHED anterior localizado."
    )


    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    print(
        "\nCarregando base v2..."
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
        f"✓ ENRICHED: "
        f"{len(ENRICHED_FEATURES)} "
        "features brutas"
    )

    print(
        f"✓ ENRICHED + UF: "
        f"{len(ENRICHED_UF_FEATURES)} "
        "features brutas"
    )

    print(
        "✓ UFs presentes em 2023: "
        f"{df['sigla_uf'].nunique()}"
    )


    # --------------------------------------------------------
    # Mesmos folds do experimento anterior
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
    # Recuperar eventual checkpoint
    # --------------------------------------------------------

    checkpoint_df = (
        load_checkpoint_folds()
    )

    if checkpoint_df.empty:

        checkpoint_rows = []

        print(
            "✓ Nenhum checkpoint anterior encontrado."
        )

    else:

        checkpoint_rows = (
            checkpoint_df
            .to_dict(
                orient="records"
            )
        )

        print(
            "✓ Checkpoint encontrado: "
            f"{len(checkpoint_rows)} "
            "fold(s) já concluído(s)."
        )


    # --------------------------------------------------------
    # Executar apenas ENRICHED + UF
    # --------------------------------------------------------

    models = build_models()

    uf_summaries = []


    for model_name, estimator in (
        models.items()
    ):

        _, summary = run_model(
            df=df,
            folds=folds,
            model_name=model_name,
            estimator=estimator,
            checkpoint_rows=checkpoint_rows,
        )

        uf_summaries.append(
            summary
        )


    # ========================================================
    # Persistência final
    # ========================================================

    uf_folds_df = pd.DataFrame(
        checkpoint_rows
    ).sort_values(
        [
            "model",
            "fold",
        ]
    )


    uf_summary_df = pd.DataFrame(
        uf_summaries
    )


    comparison_df = (
        build_comparison_summary(
            previous_enriched,
            uf_summary_df,
        )
    )


    uf_folds_df.to_csv(
        FOLDS_OUTPUT_PATH,
        index=False,
    )


    uf_summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )


    comparison_df.to_csv(
        DELTAS_OUTPUT_PATH,
        index=False,
    )


    # --------------------------------------------------------
    # Mostrar resultado
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RESUMO ENRICHED + UF"
    )

    print(
        "=" * 72
    )


    display_summary_columns = [
        "model",
        "raw_features",
        "transformed_features",
        "oof_pr_auc_risk",
        "oof_roc_auc_literate",
        "oof_balanced_accuracy",
        "oof_precision_risk",
        "oof_recall_risk",
        "oof_f1_risk",
    ]


    print(
        uf_summary_df[
            display_summary_columns
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "DELTA (ENRICHED + UF) - ENRICHED"
    )

    print(
        "=" * 72
    )


    delta_columns = [
        "model",
        "delta_oof_pr_auc_risk",
        "delta_oof_roc_auc_literate",
        "delta_oof_balanced_accuracy",
        "delta_oof_precision_risk",
        "delta_oof_recall_risk",
        "delta_oof_f1_risk",
    ]


    print(
        comparison_df[
            delta_columns
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nArquivos gerados:"
    )

    print(
        f"  ✓ "
        f"{FOLDS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"  ✓ "
        f"{SUMMARY_OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"  ✓ "
        f"{DELTAS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


    # --------------------------------------------------------
    # Checkpoints temporários só são removidos depois que
    # todos os resultados finais foram gravados.
    # --------------------------------------------------------

    cleanup_checkpoints()

    print(
        "  ✓ Checkpoints temporários removidos."
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "COMPARAÇÃO DE UF CONCLUÍDA"
    )

    print(
        "=" * 72
    )

    print(
        "2024 permanece reservado para "
        "avaliação temporal final."
    )


if __name__ == "__main__":
    main()