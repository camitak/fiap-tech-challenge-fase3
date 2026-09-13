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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.preprocessing.preprocessor import (
    build_preprocessor,
    get_model_features,
)


# ============================================================
# Tech Challenge Fase 3
#
# AVALIAÇÃO TEMPORAL FINAL — 2024
#
# Regras metodológicas:
#
# - 2023 = desenvolvimento
# - 2024 = teste temporal final
#
# Neste ponto já estão congelados:
# - features
# - algoritmo
# - hiperparâmetros
# - threshold
#
# Nenhuma decisão poderá ser alterada com base nos resultados
# observados em 2024.
# ============================================================


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


REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuração congelada
# ============================================================


FROZEN_CONFIG_PATH = (
    REPORTS_PATH
    / "hgb_final_threshold_2023.json"
)


# ============================================================
# Saídas
# ============================================================


FINAL_TEST_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_final_test_2024.json"
)


SEGMENTS_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_final_test_2024_segments.csv"
)


MUNICIPALITIES_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_final_test_2024_municipalities.csv"
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


# Algumas dessas colunas já pertencem a FEATURES.
# dict.fromkeys evita duplicação.
READ_COLUMNS = list(
    dict.fromkeys(
        CONTROL_COLUMNS
        + FEATURES
    )
)


# ============================================================
# Carregamento da configuração congelada
# ============================================================


def load_frozen_config() -> dict:

    if not FROZEN_CONFIG_PATH.exists():

        raise FileNotFoundError(
            "Configuração final não encontrada: "
            f"{FROZEN_CONFIG_PATH}"
        )


    with open(
        FROZEN_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(
            file
        )


    # --------------------------------------------------------
    # Auditorias metodológicas
    # --------------------------------------------------------

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

    assert (
        config[
            "development_year"
        ]
        == 2023
    )

    assert (
        config[
            "test_year_2024_used"
        ]
        is False
    )

    assert np.isclose(
        config[
            "selected_threshold_risk"
        ],
        0.41,
    )


    return config


# ============================================================
# Carregamento dos Parquets
# ============================================================


def load_data(
    path: Path,
    year: int,
) -> pd.DataFrame:

    files = sorted(
        path.glob(
            "*.parquet"
        )
    )


    if not files:

        raise FileNotFoundError(
            "Nenhum arquivo Parquet encontrado em: "
            f"{path}"
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
    # Reconciliações conhecidas das bases v2
    # --------------------------------------------------------

    if year == 2023:

        expected_rows = (
            1_502_809
        )

        expected_non_literate = (
            625_382
        )

        expected_literate = (
            877_427
        )

        expected_municipalities = (
            4_871
        )

    elif year == 2024:

        expected_rows = (
            1_851_852
        )

        expected_non_literate = (
            744_733
        )

        expected_literate = (
            1_107_119
        )

        expected_municipalities = (
            5_517
        )

    else:

        raise ValueError(
            f"Ano inesperado: {year}"
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

    assert int(
        (
            df[
                "alfabetizado"
            ]
            == 0
        ).sum()
    ) == expected_non_literate

    assert int(
        (
            df[
                "alfabetizado"
            ]
            == 1
        ).sum()
    ) == expected_literate

    assert (
        df[
            "id_municipio"
        ]
        .nunique()
        == expected_municipalities
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
# Pipeline final
#
# Os hiperparâmetros são lidos diretamente do artefato
# congelado de 2023.
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
                HistGradientBoostingClassifier(
                    **params
                ),
            ),
        ]
    )


# ============================================================
# Probabilidade da classe alfabetizado=1
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
    y_true_literate: np.ndarray,
    p_literate: np.ndarray,
) -> dict:

    y_true_literate = np.asarray(
        y_true_literate,
        dtype=int,
    )


    p_literate = np.asarray(
        p_literate,
        dtype=float,
    )


    y_true_risk = (
        1
        - y_true_literate
    )


    p_risk = (
        1.0
        - p_literate
    )


    return {
        "roc_auc_literate": (
            float(
                roc_auc_score(
                    y_true_literate,
                    p_literate,
                )
            )
        ),

        "pr_auc_risk": (
            float(
                average_precision_score(
                    y_true_risk,
                    p_risk,
                )
            )
        ),
    }


# ============================================================
# Métricas no threshold final
# ============================================================


def threshold_metrics(
    y_true_literate: np.ndarray,
    p_risk: np.ndarray,
    threshold: float,
) -> dict:

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
            float(
                accuracy_score(
                    y_true_literate,
                    y_pred_literate,
                )
            )
        ),

        "balanced_accuracy": (
            float(
                balanced_accuracy_score(
                    y_true_risk,
                    y_pred_risk,
                )
            )
        ),

        "precision_risk": (
            float(
                precision_score(
                    y_true_risk,
                    y_pred_risk,
                    zero_division=0,
                )
            )
        ),

        "recall_risk": (
            float(
                recall_score(
                    y_true_risk,
                    y_pred_risk,
                    zero_division=0,
                )
            )
        ),

        "f1_risk": (
            float(
                f1_score(
                    y_true_risk,
                    y_pred_risk,
                    zero_division=0,
                )
            )
        ),

        "recall_literate": (
            float(
                recall_score(
                    y_true_literate,
                    y_pred_literate,
                    pos_label=1,
                    zero_division=0,
                )
            )
        ),

        "actual_risk_rate": (
            float(
                y_true_risk.mean()
            )
        ),

        "predicted_risk_rate": (
            float(
                y_pred_risk.mean()
            )
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
# Métricas seguras para segmentos
#
# ROC-AUC e PR-AUC só são reportadas quando o segmento contém
# as duas classes.
# ============================================================


def segment_metrics(
    frame: pd.DataFrame,
    threshold: float,
) -> dict:

    y = (
        frame[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    p_literate = (
        frame[
            "p_alfabetizado"
        ]
        .to_numpy()
    )


    p_risk = (
        frame[
            "p_risco"
        ]
        .to_numpy()
    )


    result = {
        "n_rows": (
            int(
                len(frame)
            )
        ),

        "n_municipalities": (
            int(
                frame[
                    "id_municipio"
                ]
                .nunique()
            )
        ),

        "n_ufs": (
            int(
                frame[
                    "sigla_uf"
                ]
                .nunique()
            )
        ),
    }


    threshold_result = (
        threshold_metrics(
            y_true_literate=y,
            p_risk=p_risk,
            threshold=threshold,
        )
    )


    result.update(
        threshold_result
    )


    if (
        np.unique(
            y
        ).size
        == 2
    ):

        rank = (
            ranking_metrics(
                y_true_literate=y,
                p_literate=p_literate,
            )
        )


        result.update(
            rank
        )

    else:

        result[
            "roc_auc_literate"
        ] = np.nan

        result[
            "pr_auc_risk"
        ] = np.nan


    return result


# ============================================================
# Resumos por segmento
# ============================================================


def build_segment_report(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    rows = []


    def add_segment(
        segment_type: str,
        segment_value: str,
        frame: pd.DataFrame,
    ) -> None:

        metrics = (
            segment_metrics(
                frame=frame,
                threshold=threshold,
            )
        )


        rows.append(
            {
                "segment_type": (
                    segment_type
                ),

                "segment_value": (
                    segment_value
                ),

                **metrics,
            }
        )


    # --------------------------------------------------------
    # Geral
    # --------------------------------------------------------

    add_segment(
        segment_type="overall",
        segment_value="all_2024",
        frame=predictions,
    )


    # --------------------------------------------------------
    # Município visto / não visto em 2023
    # --------------------------------------------------------

    for value in [
        False,
        True,
    ]:

        subset = predictions[
            predictions[
                "municipio_visto_2023"
            ]
            == value
        ]


        if len(
            subset
        ) == 0:

            continue


        add_segment(
            segment_type=(
                "municipality_seen_status"
            ),

            segment_value=(
                "seen_in_2023"
                if value
                else "new_in_2024"
            ),

            frame=subset,
        )


    # --------------------------------------------------------
    # UF vista / não vista em 2023
    # --------------------------------------------------------

    for value in [
        False,
        True,
    ]:

        subset = predictions[
            predictions[
                "uf_vista_2023"
            ]
            == value
        ]


        if len(
            subset
        ) == 0:

            continue


        add_segment(
            segment_type=(
                "uf_seen_status"
            ),

            segment_value=(
                "seen_in_2023"
                if value
                else "unseen_in_2023"
            ),

            frame=subset,
        )


    # --------------------------------------------------------
    # Redes
    # --------------------------------------------------------

    for network in sorted(
        predictions[
            "rede_nome"
        ]
        .dropna()
        .unique()
    ):

        subset = predictions[
            predictions[
                "rede_nome"
            ]
            == network
        ]


        add_segment(
            segment_type="rede_nome",
            segment_value=str(
                network
            ),
            frame=subset,
        )


    # --------------------------------------------------------
    # UFs que não existiam no desenvolvimento
    # --------------------------------------------------------

    unseen_ufs = sorted(
        predictions.loc[
            ~predictions[
                "uf_vista_2023"
            ],
            "sigla_uf",
        ]
        .dropna()
        .unique()
    )


    for uf in unseen_ufs:

        subset = predictions[
            predictions[
                "sigla_uf"
            ]
            == uf
        ]


        add_segment(
            segment_type="unseen_uf",
            segment_value=str(
                uf
            ),
            frame=subset,
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Resumo municipal
#
# Esse arquivo será útil depois para:
# - ranking de municípios de maior risco;
# - análise territorial;
# - comparação com metas.
#
# Nesta etapa ele é apenas gerado.
# Não fazemos decisão de modelo com ele.
# ============================================================


def build_municipality_report(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    frame = (
        predictions.copy()
    )


    frame[
        "risco_real"
    ] = (
        1
        - frame[
            "alfabetizado"
        ]
    )


    grouped = (
        frame
        .groupby(
            [
                "id_municipio",
                "sigla_uf",
                "regiao",
                "municipio_visto_2023",
                "uf_vista_2023",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_alunos=(
                "aluno_key",
                "size",
            ),

            taxa_alfabetizacao_real=(
                "alfabetizado",
                "mean",
            ),

            taxa_risco_real=(
                "risco_real",
                "mean",
            ),

            probabilidade_media_alfabetizado=(
                "p_alfabetizado",
                "mean",
            ),

            probabilidade_media_risco=(
                "p_risco",
                "mean",
            ),

            taxa_sinalizada_risco=(
                "predito_risco",
                "mean",
            ),
        )
    )


    grouped = (
        grouped
        .sort_values(
            [
                "probabilidade_media_risco",
                "n_alunos",
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


    grouped.insert(
        0,
        "rank_risco_modelo",
        np.arange(
            1,
            len(grouped) + 1,
        ),
    )


    return grouped


# ============================================================
# Main
# ============================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "AVALIAÇÃO TEMPORAL FINAL — 2024"
    )

    print(
        "=" * 72
    )

    print(
        "Nenhuma decisão de modelagem "
        "poderá ser alterada após esta execução."
    )


    # --------------------------------------------------------
    # Configuração congelada
    # --------------------------------------------------------

    config = (
        load_frozen_config()
    )


    threshold = float(
        config[
            "selected_threshold_risk"
        ]
    )


    print(
        "\nConfiguração congelada:"
    )

    print(
        "  Modelo: "
        f"{config['model']}"
    )

    print(
        "  Feature set: "
        f"{config['feature_set']}"
    )

    print(
        "  Threshold risco: "
        f"{threshold:.2f}"
    )


    # ========================================================
    # 1. TREINO FINAL — TODO 2023
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "1. TREINO FINAL COM TODO 2023"
    )

    print(
        "=" * 72
    )


    train_df = (
        load_data(
            path=TRAIN_PATH,
            year=2023,
        )
    )


    print(
        f"✓ Linhas treino: "
        f"{len(train_df):,}"
    )

    print(
        f"✓ Municípios treino: "
        f"{train_df['id_municipio'].nunique():,}"
    )


    seen_municipalities = set(
        train_df[
            "id_municipio"
        ]
        .astype(str)
        .unique()
    )


    seen_ufs = set(
        train_df[
            "sigla_uf"
        ]
        .dropna()
        .astype(str)
        .unique()
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
        f"✓ Features transformadas: "
        f"{transformed_features}"
    )

    print(
        f"✓ Treino final concluído em "
        f"{fit_seconds:.1f}s"
    )


    # --------------------------------------------------------
    # Liberar base de treino antes de abrir teste
    # --------------------------------------------------------

    del train_df
    del y_train

    gc.collect()


    # ========================================================
    # 2. ABRINDO 2024
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "2. ABRINDO 2024 — TESTE TEMPORAL FINAL"
    )

    print(
        "=" * 72
    )


    test_df = (
        load_data(
            path=TEST_PATH,
            year=2024,
        )
    )


    print(
        f"✓ Linhas teste: "
        f"{len(test_df):,}"
    )

    print(
        f"✓ Municípios teste: "
        f"{test_df['id_municipio'].nunique():,}"
    )


    # --------------------------------------------------------
    # Municípios vistos / novos
    # --------------------------------------------------------

    municipality_as_string = (
        test_df[
            "id_municipio"
        ]
        .astype(str)
    )


    test_df[
        "municipio_visto_2023"
    ] = (
        municipality_as_string
        .isin(
            seen_municipalities
        )
    )


    test_df[
        "uf_vista_2023"
    ] = (
        test_df[
            "sigla_uf"
        ]
        .astype(str)
        .isin(
            seen_ufs
        )
    )


    test_municipalities = set(
        municipality_as_string
        .unique()
    )


    known_municipalities = (
        test_municipalities
        & seen_municipalities
    )


    new_municipalities = (
        test_municipalities
        - seen_municipalities
    )


    unseen_ufs = sorted(
        set(
            test_df[
                "sigla_uf"
            ]
            .dropna()
            .astype(str)
            .unique()
        )
        - seen_ufs
    )


    print(
        f"✓ Municípios também vistos em 2023: "
        f"{len(known_municipalities):,}"
    )

    print(
        f"✓ Municípios novos em 2024: "
        f"{len(new_municipalities):,}"
    )

    print(
        "✓ UFs não vistas no desenvolvimento: "
        + (
            ", ".join(
                unseen_ufs
            )
            if unseen_ufs
            else "nenhuma"
        )
    )


    # Reconciliação com a EDA anterior.
    assert (
        len(
            known_municipalities
        )
        == 4_841
    )

    assert (
        len(
            new_municipalities
        )
        == 676
    )


    # ========================================================
    # 3. PREDIÇÃO — UMA ÚNICA VEZ
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "3. GERANDO PREDIÇÕES DE 2024"
    )

    print(
        "=" * 72
    )


    predict_start = (
        time.perf_counter()
    )


    p_literate = (
        predict_literate_probability(
            pipeline,
            test_df[
                FEATURES
            ],
        )
    )


    predict_seconds = (
        time.perf_counter()
        - predict_start
    )


    p_risk = (
        1.0
        - p_literate
    )


    predicted_risk = (
        p_risk
        >= threshold
    ).astype(
        np.int8
    )


    assert np.all(
        (
            p_literate
            >= 0.0
        )
        & (
            p_literate
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


    print(
        f"✓ Predições concluídas em "
        f"{predict_seconds:.1f}s"
    )


    # ========================================================
    # 4. MÉTRICAS GLOBAIS
    # ========================================================

    y_test = (
        test_df[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    rank = (
        ranking_metrics(
            y_true_literate=y_test,
            p_literate=p_literate,
        )
    )


    threshold_result = (
        threshold_metrics(
            y_true_literate=y_test,
            p_risk=p_risk,
            threshold=threshold,
        )
    )


    # --------------------------------------------------------
    # Tabela para análises de segmentos
    # --------------------------------------------------------

    predictions = test_df[
        [
            "aluno_key",
            "id_municipio",
            "alfabetizado",
            "rede_nome",
            "regiao",
            "sigla_uf",
            "municipio_visto_2023",
            "uf_vista_2023",
        ]
    ].copy()


    predictions[
        "p_alfabetizado"
    ] = p_literate


    predictions[
        "p_risco"
    ] = p_risk


    predictions[
        "predito_risco"
    ] = predicted_risk


    # ========================================================
    # 5. SEGMENTOS DE GENERALIZAÇÃO
    # ========================================================

    segments = (
        build_segment_report(
            predictions=predictions,
            threshold=threshold,
        )
    )


    municipalities = (
        build_municipality_report(
            predictions
        )
    )


    # ========================================================
    # 6. COMPARAÇÃO TEMPORAL
    #
    # Isto é apenas relatório.
    # Não existe retuning após essa comparação.
    # ========================================================

    development = {
        "oof_pr_auc_risk": (
            float(
                config[
                    "oof_pr_auc_risk"
                ]
            )
        ),

        "oof_roc_auc_literate": (
            float(
                config[
                    "oof_roc_auc_literate"
                ]
            )
        ),

        "balanced_accuracy": (
            float(
                config[
                    "selected_threshold_metrics"
                ][
                    "balanced_accuracy"
                ]
            )
        ),

        "precision_risk": (
            float(
                config[
                    "selected_threshold_metrics"
                ][
                    "precision_risk"
                ]
            )
        ),

        "recall_risk": (
            float(
                config[
                    "selected_threshold_metrics"
                ][
                    "recall_risk"
                ]
            )
        ),

        "f1_risk": (
            float(
                config[
                    "selected_threshold_metrics"
                ][
                    "f1_risk"
                ]
            )
        ),
    }


    temporal_delta = {
        "pr_auc_risk": (
            rank[
                "pr_auc_risk"
            ]
            - development[
                "oof_pr_auc_risk"
            ]
        ),

        "roc_auc_literate": (
            rank[
                "roc_auc_literate"
            ]
            - development[
                "oof_roc_auc_literate"
            ]
        ),

        "balanced_accuracy": (
            threshold_result[
                "balanced_accuracy"
            ]
            - development[
                "balanced_accuracy"
            ]
        ),

        "precision_risk": (
            threshold_result[
                "precision_risk"
            ]
            - development[
                "precision_risk"
            ]
        ),

        "recall_risk": (
            threshold_result[
                "recall_risk"
            ]
            - development[
                "recall_risk"
            ]
        ),

        "f1_risk": (
            threshold_result[
                "f1_risk"
            ]
            - development[
                "f1_risk"
            ]
        ),
    }


    final_result = {
        "evaluation_type": (
            "final_temporal_test"
        ),

        "train_year": 2023,

        "test_year": 2024,

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

        "raw_features": (
            len(
                FEATURES
            )
        ),

        "transformed_features": (
            transformed_features
        ),

        "threshold_risk": (
            threshold
        ),

        "final_hgb_params": (
            config[
                "final_hgb_params"
            ]
        ),

        "n_train_municipalities": (
            len(
                seen_municipalities
            )
        ),

        "n_test_rows": (
            int(
                len(
                    predictions
                )
            )
        ),

        "n_test_municipalities": (
            int(
                predictions[
                    "id_municipio"
                ]
                .nunique()
            )
        ),

        "municipalities_seen_in_2023": (
            int(
                len(
                    known_municipalities
                )
            )
        ),

        "municipalities_new_in_2024": (
            int(
                len(
                    new_municipalities
                )
            )
        ),

        "ufs_seen_in_2023": (
            sorted(
                seen_ufs
            )
        ),

        "ufs_unseen_in_2023": (
            unseen_ufs
        ),

        "test_ranking_metrics": (
            rank
        ),

        "test_threshold_metrics": (
            threshold_result
        ),

        "development_2023_oof_reference": (
            development
        ),

        "temporal_delta_test_minus_development": (
            temporal_delta
        ),

        "fit_runtime_seconds": (
            float(
                fit_seconds
            )
        ),

        "prediction_runtime_seconds": (
            float(
                predict_seconds
            )
        ),

        "model_retrained_after_seeing_2024": (
            False
        ),

        "threshold_changed_after_seeing_2024": (
            False
        ),
    }


    # ========================================================
    # 7. PERSISTÊNCIA
    # ========================================================

    with open(
        FINAL_TEST_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_result,
            file,
            indent=2,
            ensure_ascii=False,
        )


    segments.to_csv(
        SEGMENTS_OUTPUT_PATH,
        index=False,
    )


    municipalities.to_csv(
        MUNICIPALITIES_OUTPUT_PATH,
        index=False,
    )


    # ========================================================
    # 8. EXIBIÇÃO
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RESULTADO FINAL — TESTE 2024"
    )

    print(
        "=" * 72
    )


    print(
        f"PR-AUC risco: "
        f"{rank['pr_auc_risk']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{rank['roc_auc_literate']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{threshold_result['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{threshold_result['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision risco: "
        f"{threshold_result['precision_risk']:.4f}"
    )

    print(
        f"Recall risco: "
        f"{threshold_result['recall_risk']:.4f}"
    )

    print(
        f"F1 risco: "
        f"{threshold_result['f1_risk']:.4f}"
    )

    print(
        f"Recall alfabetizado: "
        f"{threshold_result['recall_literate']:.4f}"
    )

    print(
        f"Taxa real de risco: "
        f"{threshold_result['actual_risk_rate']:.4f}"
    )

    print(
        f"Taxa prevista de risco: "
        f"{threshold_result['predicted_risk_rate']:.4f}"
    )


    print(
        "\nMatriz de confusão "
        "(positivo = risco):"
    )

    print(
        f"TN: "
        f"{threshold_result['true_negative']:,}"
    )

    print(
        f"FP: "
        f"{threshold_result['false_positive']:,}"
    )

    print(
        f"FN: "
        f"{threshold_result['false_negative']:,}"
    )

    print(
        f"TP: "
        f"{threshold_result['true_positive']:,}"
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "DELTA TEMPORAL — 2024 MENOS OOF 2023"
    )

    print(
        "=" * 72
    )


    for metric, value in (
        temporal_delta.items()
    ):

        print(
            f"{metric}: "
            f"{value:+.4f}"
        )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "GENERALIZAÇÃO TERRITORIAL"
    )

    print(
        "=" * 72
    )


    print(
        "Municípios vistos em 2023: "
        f"{len(known_municipalities):,}"
    )

    print(
        "Municípios novos em 2024: "
        f"{len(new_municipalities):,}"
    )

    print(
        "UFs não vistas no desenvolvimento: "
        + (
            ", ".join(
                unseen_ufs
            )
            if unseen_ufs
            else "nenhuma"
        )
    )


    # --------------------------------------------------------
    # Mostrar segmentos mais importantes
    # --------------------------------------------------------

    important_segments = (
        segments[
            segments[
                "segment_type"
            ]
            .isin(
                [
                    "overall",
                    "municipality_seen_status",
                    "uf_seen_status",
                    "rede_nome",
                    "unseen_uf",
                ]
            )
        ][
            [
                "segment_type",
                "segment_value",
                "n_rows",
                "pr_auc_risk",
                "roc_auc_literate",
                "balanced_accuracy",
                "precision_risk",
                "recall_risk",
                "f1_risk",
                "actual_risk_rate",
                "predicted_risk_rate",
            ]
        ]
    )


    print(
        "\n"
        + important_segments
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
            FINAL_TEST_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            SEGMENTS_OUTPUT_PATH
            .relative_to(
                PROJECT_ROOT
            )
        )
    )

    print(
        "  ✓ "
        + str(
            MUNICIPALITIES_OUTPUT_PATH
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
        "TESTE TEMPORAL FINAL CONCLUÍDO"
    )

    print(
        "=" * 72
    )

    print(
        "2024 não poderá ser usado para "
        "retuning de features, algoritmo, "
        "hiperparâmetros ou threshold."
    )


    # --------------------------------------------------------
    # Limpeza
    # --------------------------------------------------------

    del pipeline
    del test_df
    del predictions
    del p_literate
    del p_risk

    gc.collect()


if __name__ == "__main__":
    main()