from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocessing.preprocessor import (
    build_preprocessor,
    get_model_features,
)


# ============================================================
# Tech Challenge Fase 3
#
# INTERPRETABILIDADE — PERMUTATION IMPORTANCE
#
# Objetivo:
# medir quanto a performance do pipeline final piora quando
# cada feature bruta é embaralhada.
#
# Regras:
# - modelo já congelado;
# - features já congeladas;
# - hiperparâmetros já congelados;
# - threshold já congelado;
# - 2024 utilizado apenas para interpretação pós-teste;
# - nenhuma decisão de modelagem será alterada;
# - importância NÃO deve ser interpretada como causalidade.
#
# Métrica principal:
# queda no PR-AUC da classe de risco.
#
# Métricas complementares:
# - queda no ROC-AUC;
# - queda na Balanced Accuracy @ threshold 0.41.
# ============================================================


RANDOM_STATE = 42

SAMPLE_SIZE = 250_000

N_REPEATS = 5


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "base_modelagem"
    / "v2"
    / "ano=2023"
)


TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "base_modelagem"
    / "v2"
    / "ano=2024"
)


REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)


IMAGES_PATH = (
    PROJECT_ROOT
    / "images"
)


REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


IMAGES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Artefato de configuração congelada
# ============================================================


FINAL_CONFIG_PATH = (
    REPORTS_PATH
    / "hgb_final_threshold_2023.json"
)


# ============================================================
# Saídas
# ============================================================


REPEATS_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_permutation_importance_2024_repeats.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_permutation_importance_2024_summary.csv"
)


METADATA_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_permutation_importance_2024.json"
)


IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "hgb_permutation_importance_2024.png"
)


# ============================================================
# Features congeladas
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
# Configuração
# ============================================================


def load_final_config() -> dict:

    if not FINAL_CONFIG_PATH.exists():

        raise FileNotFoundError(
            "Configuração final não encontrada: "
            f"{FINAL_CONFIG_PATH}"
        )


    with open(
        FINAL_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )


    assert (
        config[
            "model"
        ]
        == "HistGradientBoostingClassifier"
    )

    assert (
        config[
            "feature_set"
        ]
        == "enriched_uf"
    )

    assert np.isclose(
        config[
            "selected_threshold_risk"
        ],
        0.41,
    )


    return config


# ============================================================
# Dados
# ============================================================


def load_parquet_dataset(
    path: Path,
    expected_rows: int,
) -> pd.DataFrame:

    files = sorted(
        path.glob(
            "*.parquet"
        )
    )


    if not files:

        raise FileNotFoundError(
            f"Nenhum Parquet encontrado em {path}"
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


    assert (
        len(df)
        == expected_rows
    )


    assert (
        df[
            "aluno_key"
        ]
        .nunique()
        == expected_rows
    )


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
# Pipeline congelado
# ============================================================


def build_final_pipeline(
    config: dict,
) -> Pipeline:

    params = dict(
        config[
            "final_hgb_params"
        ]
    )


    preprocessor = (
        build_preprocessor(
            feature_set="enriched",
            include_uf=True,
        )
    )


    model = (
        HistGradientBoostingClassifier(
            **params
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
                FunctionTransformer(
                    to_dense,
                    validate=False,
                ),
            ),

            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# Probabilidade alfabetizado
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
# Métricas
# ============================================================


def calculate_metrics(
    y_literate: np.ndarray,
    p_literate: np.ndarray,
    threshold_risk: float,
) -> dict[str, float]:

    y_literate = np.asarray(
        y_literate,
        dtype=int,
    )


    p_literate = np.asarray(
        p_literate,
        dtype=float,
    )


    y_risk = (
        1
        - y_literate
    )


    p_risk = (
        1.0
        - p_literate
    )


    y_pred_risk = (
        p_risk
        >= threshold_risk
    ).astype(int)


    return {
        "pr_auc_risk": (
            float(
                average_precision_score(
                    y_risk,
                    p_risk,
                )
            )
        ),

        "roc_auc_literate": (
            float(
                roc_auc_score(
                    y_literate,
                    p_literate,
                )
            )
        ),

        "balanced_accuracy": (
            float(
                balanced_accuracy_score(
                    y_risk,
                    y_pred_risk,
                )
            )
        ),
    }


# ============================================================
# Amostra estratificada de 2024
#
# Estratificação:
# alfabetizado + UF
#
# Isso preserva aproximadamente:
# - prevalência da classe;
# - composição territorial.
# ============================================================


def build_interpretability_sample(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if SAMPLE_SIZE >= len(
        df
    ):

        return (
            df
            .copy()
            .reset_index(
                drop=True
            )
        )


    assert (
        df[
            "sigla_uf"
        ]
        .isna()
        .sum()
        == 0
    )


    strata = (
        df[
            "alfabetizado"
        ]
        .astype(str)
        + "__"
        + df[
            "sigla_uf"
        ]
        .astype(str)
    )


    splitter = (
        StratifiedShuffleSplit(
            n_splits=1,
            train_size=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )
    )


    sample_index, _ = next(
        splitter.split(
            X=np.zeros(
                len(df)
            ),
            y=strata,
        )
    )


    sample = (
        df
        .iloc[
            sample_index
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    assert (
        len(sample)
        == SAMPLE_SIZE
    )


    return sample


# ============================================================
# Permutation Importance
# ============================================================


def calculate_permutation_importance(
    pipeline: Pipeline,
    sample: pd.DataFrame,
    threshold_risk: float,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    X = (
        sample[
            FEATURES
        ]
        .copy()
    )


    y = (
        sample[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    # --------------------------------------------------------
    # Performance sem permutação
    # --------------------------------------------------------

    print(
        "\nCalculando performance baseline..."
    )


    baseline_probability = (
        predict_literate_probability(
            pipeline,
            X,
        )
    )


    baseline = (
        calculate_metrics(
            y_literate=y,
            p_literate=baseline_probability,
            threshold_risk=threshold_risk,
        )
    )


    print(
        "✓ PR-AUC risco baseline: "
        f"{baseline['pr_auc_risk']:.4f}"
    )

    print(
        "✓ ROC-AUC baseline: "
        f"{baseline['roc_auc_literate']:.4f}"
    )

    print(
        "✓ Balanced Accuracy baseline: "
        f"{baseline['balanced_accuracy']:.4f}"
    )


    # --------------------------------------------------------
    # Uma cópia é reutilizada.
    #
    # Antes de passar para a próxima feature, a coluna original
    # é restaurada.
    # --------------------------------------------------------

    X_permuted = (
        X.copy()
    )


    rows = []


    total_experiments = (
        len(FEATURES)
        * N_REPEATS
    )


    experiment = 0


    # ========================================================
    # Features
    # ========================================================

    for feature_index, feature in enumerate(
        FEATURES,
        start=1,
    ):

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"FEATURE "
            f"{feature_index}/{len(FEATURES)} "
            f"— {feature}"
        )

        print(
            "=" * 72
        )


        original_values = (
            X[
                feature
            ]
            .to_numpy(
                copy=True
            )
        )


        for repeat in range(
            1,
            N_REPEATS + 1,
        ):

            experiment += 1


            seed = (
                RANDOM_STATE
                + feature_index * 100
                + repeat
            )


            rng = (
                np.random.default_rng(
                    seed
                )
            )


            shuffled_values = (
                rng.permutation(
                    original_values
                )
            )


            X_permuted[
                feature
            ] = (
                shuffled_values
            )


            start_time = (
                time.perf_counter()
            )


            permuted_probability = (
                predict_literate_probability(
                    pipeline,
                    X_permuted,
                )
            )


            metrics = (
                calculate_metrics(
                    y_literate=y,
                    p_literate=(
                        permuted_probability
                    ),
                    threshold_risk=(
                        threshold_risk
                    ),
                )
            )


            elapsed = (
                time.perf_counter()
                - start_time
            )


            # ------------------------------------------------
            # Importância = performance original
            #             - performance após permutação
            #
            # Valores positivos:
            # embaralhar a feature piorou o modelo.
            #
            # Quanto maior, maior a dependência do modelo.
            # ------------------------------------------------

            pr_drop = (
                baseline[
                    "pr_auc_risk"
                ]
                - metrics[
                    "pr_auc_risk"
                ]
            )


            roc_drop = (
                baseline[
                    "roc_auc_literate"
                ]
                - metrics[
                    "roc_auc_literate"
                ]
            )


            ba_drop = (
                baseline[
                    "balanced_accuracy"
                ]
                - metrics[
                    "balanced_accuracy"
                ]
            )


            rows.append(
                {
                    "feature": (
                        feature
                    ),

                    "repeat": (
                        repeat
                    ),

                    "seed": (
                        seed
                    ),

                    "baseline_pr_auc_risk": (
                        baseline[
                            "pr_auc_risk"
                        ]
                    ),

                    "permuted_pr_auc_risk": (
                        metrics[
                            "pr_auc_risk"
                        ]
                    ),

                    "importance_pr_auc_drop": (
                        pr_drop
                    ),

                    "baseline_roc_auc_literate": (
                        baseline[
                            "roc_auc_literate"
                        ]
                    ),

                    "permuted_roc_auc_literate": (
                        metrics[
                            "roc_auc_literate"
                        ]
                    ),

                    "importance_roc_auc_drop": (
                        roc_drop
                    ),

                    "baseline_balanced_accuracy": (
                        baseline[
                            "balanced_accuracy"
                        ]
                    ),

                    "permuted_balanced_accuracy": (
                        metrics[
                            "balanced_accuracy"
                        ]
                    ),

                    "importance_balanced_accuracy_drop": (
                        ba_drop
                    ),

                    "runtime_seconds": (
                        elapsed
                    ),
                }
            )


            print(
                f"[{experiment:02d}/"
                f"{total_experiments}] "
                f"repeat={repeat} "
                f"| ΔPR={pr_drop:+.5f} "
                f"| ΔROC={roc_drop:+.5f} "
                f"| ΔBA={ba_drop:+.5f} "
                f"| {elapsed:.1f}s"
            )


        # ----------------------------------------------------
        # Restaurar feature antes de passar para próxima
        # ----------------------------------------------------

        X_permuted[
            feature
        ] = (
            original_values
        )


    repeats_df = (
        pd.DataFrame(
            rows
        )
    )


    return (
        repeats_df,
        baseline,
    )


# ============================================================
# Resumo
# ============================================================


def summarize_importance(
    repeats_df: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        repeats_df
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            importance_pr_auc_mean=(
                "importance_pr_auc_drop",
                "mean",
            ),

            importance_pr_auc_std=(
                "importance_pr_auc_drop",
                "std",
            ),

            importance_pr_auc_min=(
                "importance_pr_auc_drop",
                "min",
            ),

            importance_pr_auc_max=(
                "importance_pr_auc_drop",
                "max",
            ),

            importance_roc_auc_mean=(
                "importance_roc_auc_drop",
                "mean",
            ),

            importance_roc_auc_std=(
                "importance_roc_auc_drop",
                "std",
            ),

            importance_balanced_accuracy_mean=(
                "importance_balanced_accuracy_drop",
                "mean",
            ),

            importance_balanced_accuracy_std=(
                "importance_balanced_accuracy_drop",
                "std",
            ),

            mean_runtime_seconds=(
                "runtime_seconds",
                "mean",
            ),
        )
    )


    summary = (
        summary
        .sort_values(
            [
                "importance_pr_auc_mean",
                "importance_roc_auc_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    summary.insert(
        0,
        "rank_pr_auc",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )


    return summary


# ============================================================
# Visualização
# ============================================================


def create_importance_plot(
    summary: pd.DataFrame,
) -> None:

    plot_data = (
        summary
        .sort_values(
            "importance_pr_auc_mean",
            ascending=True,
        )
        .copy()
    )


    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
    )


    ax.barh(
        plot_data[
            "feature"
        ],
        plot_data[
            "importance_pr_auc_mean"
        ],
        xerr=(
            plot_data[
                "importance_pr_auc_std"
            ]
            .fillna(0.0)
        ),
        capsize=3,
    )


    ax.axvline(
        0.0,
        linewidth=1,
    )


    ax.set_title(
        "Permutation Importance — HGB final"
        "\nQueda média no PR-AUC de risco — amostra 2024"
    )


    ax.set_xlabel(
        "Queda no PR-AUC após permutação"
    )


    ax.set_ylabel(
        "Feature original"
    )


    fig.tight_layout()


    fig.savefig(
        IMAGE_OUTPUT_PATH,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "INTERPRETABILIDADE — "
        "PERMUTATION IMPORTANCE"
    )

    print(
        "=" * 72
    )

    print(
        "O modelo permanece congelado."
    )

    print(
        "Nenhuma decisão de modelagem "
        "será alterada."
    )


    # ========================================================
    # 1. CONFIGURAÇÃO
    # ========================================================

    config = (
        load_final_config()
    )


    threshold_risk = float(
        config[
            "selected_threshold_risk"
        ]
    )


    print(
        "\nConfiguração:"
    )

    print(
        f"  Modelo: "
        f"{config['model']}"
    )

    print(
        f"  Feature set: "
        f"{config['feature_set']}"
    )

    print(
        f"  Threshold risco: "
        f"{threshold_risk:.2f}"
    )

    print(
        f"  Features brutas: "
        f"{len(FEATURES)}"
    )


    # ========================================================
    # 2. TREINO FINAL EM TODO 2023
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TREINANDO PIPELINE FINAL EM 2023"
    )

    print(
        "=" * 72
    )


    train_df = (
        load_parquet_dataset(
            path=TRAIN_PATH,
            expected_rows=1_502_809,
        )
    )


    y_train = (
        train_df[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    pipeline = (
        build_final_pipeline(
            config
        )
    )


    fit_start = (
        time.perf_counter()
    )


    pipeline.fit(
        train_df[
            FEATURES
        ],
        y_train,
    )


    fit_seconds = (
        time.perf_counter()
        - fit_start
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


    print(
        f"✓ Treino concluído: "
        f"{fit_seconds:.1f}s"
    )

    print(
        f"✓ Features transformadas: "
        f"{transformed_features}"
    )


    del train_df
    del y_train

    gc.collect()


    # ========================================================
    # 3. AMOSTRA 2024
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CRIANDO AMOSTRA DE INTERPRETABILIDADE — 2024"
    )

    print(
        "=" * 72
    )


    test_df = (
        load_parquet_dataset(
            path=TEST_PATH,
            expected_rows=1_851_852,
        )
    )


    sample = (
        build_interpretability_sample(
            test_df
        )
    )


    full_risk_rate = (
        1.0
        - test_df[
            "alfabetizado"
        ].mean()
    )


    sample_risk_rate = (
        1.0
        - sample[
            "alfabetizado"
        ].mean()
    )


    print(
        f"✓ Base 2024: "
        f"{len(test_df):,}"
    )

    print(
        f"✓ Amostra: "
        f"{len(sample):,}"
    )

    print(
        f"✓ Taxa risco 2024: "
        f"{full_risk_rate:.4f}"
    )

    print(
        f"✓ Taxa risco amostra: "
        f"{sample_risk_rate:.4f}"
    )

    print(
        f"✓ UFs na amostra: "
        f"{sample['sigla_uf'].nunique()}"
    )


    del test_df

    gc.collect()


    # ========================================================
    # 4. PERMUTATION IMPORTANCE
    # ========================================================

    total_start = (
        time.perf_counter()
    )


    (
        repeats_df,
        baseline,
    ) = (
        calculate_permutation_importance(
            pipeline=pipeline,
            sample=sample,
            threshold_risk=(
                threshold_risk
            ),
        )
    )


    summary_df = (
        summarize_importance(
            repeats_df
        )
    )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    # ========================================================
    # 5. PERSISTÊNCIA
    # ========================================================

    repeats_df.to_csv(
        REPEATS_OUTPUT_PATH,
        index=False,
    )


    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )


    create_importance_plot(
        summary_df
    )


    metadata = {
        "analysis": (
            "permutation_importance"
        ),

        "model": (
            config[
                "model"
            ]
        ),

        "feature_set": (
            config[
                "feature_set"
            ]
        ),

        "development_year": 2023,

        "interpretability_year": 2024,

        "model_changed": False,

        "hyperparameters_changed": False,

        "threshold_changed": False,

        "sample_size": (
            int(
                len(sample)
            )
        ),

        "sample_strategy": (
            "StratifiedShuffleSplit by "
            "alfabetizado + sigla_uf"
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "n_repeats": (
            N_REPEATS
        ),

        "primary_importance_metric": (
            "drop in PR-AUC risk"
        ),

        "secondary_importance_metrics": [
            "drop in ROC-AUC",
            "drop in Balanced Accuracy",
        ],

        "threshold_risk": (
            threshold_risk
        ),

        "baseline_metrics": (
            baseline
        ),

        "sample_actual_risk_rate": (
            float(
                sample_risk_rate
            )
        ),

        "top_features": (
            summary_df[
                [
                    "rank_pr_auc",
                    "feature",
                    "importance_pr_auc_mean",
                    "importance_pr_auc_std",
                    "importance_roc_auc_mean",
                    "importance_balanced_accuracy_mean",
                ]
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        ),

        "interpretation_note": (
            "Permutation importance measures predictive "
            "dependence, not causal effect. Correlated "
            "features may share or mask importance."
        ),

        "runtime_seconds": (
            float(
                total_seconds
            )
        ),
    }


    with open(
        METADATA_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # 6. RESULTADOS
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP FEATURES — "
        "PERMUTATION IMPORTANCE"
    )

    print(
        "=" * 72
    )


    print(
        summary_df[
            [
                "rank_pr_auc",
                "feature",
                "importance_pr_auc_mean",
                "importance_pr_auc_std",
                "importance_roc_auc_mean",
                "importance_balanced_accuracy_mean",
            ]
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
        "INTERPRETAÇÃO"
    )

    print(
        "=" * 72
    )

    print(
        "Valores positivos indicam que embaralhar "
        "a variável piorou a performance."
    )

    print(
        "Quanto maior a queda no PR-AUC, maior "
        "a dependência preditiva do modelo."
    )

    print(
        "Importância preditiva NÃO representa "
        "efeito causal."
    )

    print(
        "Features correlacionadas podem dividir "
        "ou mascarar importância."
    )


    print(
        f"\nTempo total da análise: "
        f"{total_seconds / 60:.2f} min"
    )


    print(
        "\nArquivos gerados:"
    )

    print(
        "  ✓ "
        + str(
            REPEATS_OUTPUT_PATH
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
            METADATA_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            IMAGE_OUTPUT_PATH
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
        "PERMUTATION IMPORTANCE CONCLUÍDA"
    )

    print(
        "=" * 72
    )

    print(
        "Não interpretar os resultados "
        "como causalidade."
    )

    print(
        "Próxima etapa: SHAP Values."
    )


    del pipeline
    del sample
    del repeats_df
    del summary_df

    gc.collect()


if __name__ == "__main__":
    main()