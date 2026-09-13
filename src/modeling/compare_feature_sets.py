from __future__ import annotations

import gc
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
# Comparação controlada de conjuntos de features
#
# Objetivo:
# medir se o enriquecimento educacional do Censo Escolar
# adiciona capacidade preditiva ao baseline.
#
# Desenvolvimento:
# exclusivamente 2023.
#
# NÃO utilizar:
# - 2024;
# - sigla_uf;
# - otimização de threshold;
# - novo tuning nesta etapa.
#
# Estratégia:
# - mesmos 5 folds por município;
# - mesmos modelos;
# - mesmos hiperparâmetros;
# - única mudança = conjunto de features.
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
# Features
# ============================================================


BASELINE_FEATURES = get_model_features(
    feature_set="baseline",
    include_uf=False,
)


ENRICHED_FEATURES = get_model_features(
    feature_set="enriched",
    include_uf=False,
)


CONTROL_COLUMNS = [
    "aluno_key",
    "id_municipio",
    "alfabetizado",
]


READ_COLUMNS = list(
    dict.fromkeys(
        CONTROL_COLUMNS
        + ENRICHED_FEATURES
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
    # Reconciliação
    # --------------------------------------------------------

    assert len(df) == 1_502_809, (
        "Total de registros diferente do esperado."
    )

    assert (
        df["aluno_key"].nunique()
        == 1_502_809
    ), (
        "aluno_key não é único."
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
# Usamos os melhores hiperparâmetros já encontrados na etapa
# anterior com a base baseline.
#
# Eles NÃO serão retunados aqui.
#
# O objetivo é isolar a contribuição das novas features.
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
#
# Target:
#   1 = alfabetizado
#   0 = não alfabetizado
#
# Risco:
#   risco = 1 - P(alfabetizado)
#
# PR-AUC é calculada especificamente para a classe de risco
# (não alfabetizado).
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

    if len(
        positions
    ) != 1:
        raise RuntimeError(
            "Não foi possível localizar "
            "a classe alfabetizado=1."
        )

    return probabilities[
        :,
        positions[0],
    ]


# ============================================================
# Execução de uma configuração
# ============================================================


def run_configuration(
    *,
    df: pd.DataFrame,
    folds,
    feature_set: str,
    model_name: str,
    estimator,
) -> tuple[
    list[dict],
    dict,
]:

    features = get_model_features(
        feature_set=feature_set,
        include_uf=False,
    )

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

    oof_p_literate = np.full(
        len(df),
        np.nan,
        dtype=np.float32,
    )

    fold_results = []

    transformed_features = None


    print(
        "\n"
        + "=" * 72
    )

    print(
        f"{model_name.upper()} | "
        f"{feature_set.upper()}"
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

        start_time = time.perf_counter()


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

        assert not overlap, (
            f"Fold {fold_number}: "
            "há municípios presentes em treino "
            "e validação."
        )


        # ----------------------------------------------------
        # Dados do fold
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
        # Pipeline completa
        #
        # O preprocessing é treinado somente no treino do fold.
        # ----------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        feature_set=feature_set,
                        include_uf=False,
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


        # ----------------------------------------------------
        # Dimensionalidade após preprocessing
        # ----------------------------------------------------

        if (
            transformed_features
            is None
        ):
            transformed_features = len(
                pipeline
                .named_steps[
                    "preprocessor"
                ]
                .get_feature_names_out()
            )


        # ----------------------------------------------------
        # Probabilidades
        # ----------------------------------------------------

        p_literate = (
            predict_literate_probability(
                pipeline,
                X_valid,
            )
        )


        oof_p_literate[
            valid_index
        ] = p_literate.astype(
            np.float32
        )


        # ----------------------------------------------------
        # Métricas do fold
        # ----------------------------------------------------

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
            "feature_set": feature_set,
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


        fold_results.append(
            row
        )


        print(
            f"\nFold {fold_number}/{N_SPLITS}"
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


        # ----------------------------------------------------
        # Liberação de memória
        # ----------------------------------------------------

        del pipeline
        del X_train
        del X_valid

        gc.collect()


    # ========================================================
    # Métricas OOF
    # ========================================================

    assert not np.isnan(
        oof_p_literate
    ).any(), (
        "Existem registros sem predição OOF."
    )


    oof_metrics = calculate_metrics(
        y,
        oof_p_literate,
        threshold=THRESHOLD,
    )


    fold_df = pd.DataFrame(
        fold_results
    )


    summary = {
        "feature_set": feature_set,
        "model": model_name,

        "raw_features": (
            len(features)
        ),

        "transformed_features": (
            transformed_features
        ),
    }


    # --------------------------------------------------------
    # Métricas globais OOF
    # --------------------------------------------------------

    for metric_name, value in (
        oof_metrics.items()
    ):
        summary[
            f"oof_{metric_name}"
        ] = value


    # --------------------------------------------------------
    # Média e desvio entre folds
    # --------------------------------------------------------

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
            fold_df[
                metric
            ]
            .mean()
        )

        summary[
            f"fold_std_{metric}"
        ] = (
            fold_df[
                metric
            ]
            .std(
                ddof=1
            )
        )


    summary[
        "total_runtime_seconds"
    ] = (
        fold_df[
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
        fold_results,
        summary,
    )


# ============================================================
# Comparação enriquecido - baseline
# ============================================================


def build_delta_table(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "oof_pr_auc_risk",
        "oof_roc_auc_literate",
        "oof_balanced_accuracy",
        "oof_precision_risk",
        "oof_recall_risk",
        "oof_f1_risk",
    ]

    rows = []


    for model_name in sorted(
        summary_df[
            "model"
        ].unique()
    ):

        model_df = (
            summary_df[
                summary_df[
                    "model"
                ]
                == model_name
            ]
            .set_index(
                "feature_set"
            )
        )


        if not {
            "baseline",
            "enriched",
        }.issubset(
            model_df.index
        ):
            continue


        row = {
            "model": model_name,
        }


        for metric in metrics:

            baseline_value = (
                model_df.loc[
                    "baseline",
                    metric,
                ]
            )

            enriched_value = (
                model_df.loc[
                    "enriched",
                    metric,
                ]
            )

            row[
                f"baseline_{metric}"
            ] = baseline_value

            row[
                f"enriched_{metric}"
            ] = enriched_value

            row[
                f"delta_{metric}"
            ] = (
                enriched_value
                - baseline_value
            )


        rows.append(
            row
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
        "COMPARAÇÃO DE FEATURE SETS — 2023"
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
        f"✓ Baseline: "
        f"{len(BASELINE_FEATURES)} features"
    )

    print(
        f"✓ Enriched: "
        f"{len(ENRICHED_FEATURES)} features"
    )


    # --------------------------------------------------------
    # Target / grupos
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


    # --------------------------------------------------------
    # Folds
    #
    # Criados uma única vez.
    #
    # Assim baseline e enriched recebem exatamente os mesmos
    # municípios em cada fold.
    # --------------------------------------------------------

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
        f"✓ Folds GroupKFold: "
        f"{len(folds)}"
    )


    # --------------------------------------------------------
    # Configurações
    # --------------------------------------------------------

    models = build_models()

    feature_sets = [
        "baseline",
        "enriched",
    ]


    all_fold_results = []
    all_summaries = []


    for model_name, estimator in (
        models.items()
    ):

        for feature_set in (
            feature_sets
        ):

            folds_result, summary = (
                run_configuration(
                    df=df,
                    folds=folds,
                    feature_set=feature_set,
                    model_name=model_name,
                    estimator=estimator,
                )
            )

            all_fold_results.extend(
                folds_result
            )

            all_summaries.append(
                summary
            )


    # ========================================================
    # Resultados
    # ========================================================

    folds_df = pd.DataFrame(
        all_fold_results
    )

    summary_df = pd.DataFrame(
        all_summaries
    )

    delta_df = build_delta_table(
        summary_df
    )


    # --------------------------------------------------------
    # Persistência
    # --------------------------------------------------------

    folds_path = (
        REPORTS_PATH
        / "feature_set_comparison_2023_folds.csv"
    )

    summary_path = (
        REPORTS_PATH
        / "feature_set_comparison_2023_summary.csv"
    )

    delta_path = (
        REPORTS_PATH
        / "feature_set_comparison_2023_deltas.csv"
    )


    folds_df.to_csv(
        folds_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    delta_df.to_csv(
        delta_path,
        index=False,
    )


    # --------------------------------------------------------
    # Resumo no terminal
    # --------------------------------------------------------

    display_columns = [
        "model",
        "feature_set",
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
        "\n"
        + "=" * 72
    )

    print(
        "RESUMO OOF"
    )

    print(
        "=" * 72
    )

    print(
        summary_df[
            display_columns
        ]
        .sort_values(
            [
                "model",
                "feature_set",
            ]
        )
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "DELTA ENRICHED - BASELINE"
    )

    print(
        "=" * 72
    )


    delta_display = [
        "model",
        "delta_oof_pr_auc_risk",
        "delta_oof_roc_auc_literate",
        "delta_oof_balanced_accuracy",
        "delta_oof_precision_risk",
        "delta_oof_recall_risk",
        "delta_oof_f1_risk",
    ]


    print(
        delta_df[
            delta_display
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nArquivos gerados:"
    )

    print(
        f"  ✓ {folds_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"  ✓ {summary_path.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"  ✓ {delta_path.relative_to(PROJECT_ROOT)}"
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
        "Não utilizar 2024 antes da definição "
        "final de features, modelo, hiperparâmetros "
        "e threshold."
    )


if __name__ == "__main__":
    main()