from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Tech Challenge Fase 3
#
# INTELIGÊNCIA EDUCACIONAL
#
# Pergunta:
#
# "Quais municípios podem não atingir metas futuras?"
#
# Método:
# TRIAGEM DE TRAJETÓRIA.
#
# Não é uma previsão robusta de série temporal.
#
# Temos somente:
# - resultado municipal 2023;
# - resultado municipal 2024;
# - metas 2025-2030.
#
# Portanto avaliamos:
#
# 1. situação atual em relação à meta;
# 2. variação observada 2023 -> 2024;
# 3. ritmo anual necessário para atingir a meta;
# 4. distância entre trajetória recente e necessária.
#
# IMPORTANTE:
#
# - HGB NÃO participa do ranking;
# - metas NÃO foram features do modelo supervisionado;
# - clustering entra somente como interpretação pós-hoc;
# - resultados são screening para monitoramento.
# ============================================================


BASE_YEAR = 2024

FUTURE_YEARS = [
    2025,
    2026,
    2027,
    2028,
    2029,
    2030,
]

PRIMARY_TARGET_YEAR = 2025

TOP_N = 50

EPSILON = 1e-9


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# Entradas
# ============================================================


TARGET_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "meta_alfabetizacao_municipio.csv"
)


CLUSTER_PATH = (
    PROJECT_ROOT
    / "reports"
    / "clustering"
    / "kmeans_final_municipalities_2024.csv"
)


# ============================================================
# Saídas
# ============================================================


REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "business"
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


ALL_OUTPUT_PATH = (
    REPORTS_PATH
    / "future_target_risk_all.csv"
)


TOP_2025_OUTPUT_PATH = (
    REPORTS_PATH
    / "future_target_risk_top_2025.csv"
)


NO_HISTORY_OUTPUT_PATH = (
    REPORTS_PATH
    / "future_target_risk_no_2023_history.csv"
)


HORIZON_SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "future_target_risk_horizon_summary.csv"
)


CLUSTER_SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "future_target_risk_cluster_summary.csv"
)


REGION_SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "future_target_risk_region_summary.csv"
)


METADATA_OUTPUT_PATH = (
    REPORTS_PATH
    / "future_target_risk.json"
)


TOP_IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "future_target_risk_top20_2025.png"
)


HORIZON_IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "future_target_risk_horizon_2025_2030.png"
)


# ============================================================
# Clusters já congelados
# ============================================================


CLUSTER_NAMES = {
    0: (
        "Urbano-industrial de maior escala"
    ),

    1: (
        "Pequeno porte agropecuario com maior "
        "infraestrutura educacional"
    ),

    2: (
        "Rural, menor renda e maior presenca "
        "da administracao publica"
    ),
}


# ============================================================
# Helpers
# ============================================================


def normalize_id(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(7)
    )


def json_default(
    value,
):

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )


    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )


    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )


    if pd.isna(
        value
    ):
        return None


    raise TypeError(
        "Tipo não serializável: "
        f"{type(value)}"
    )


# ============================================================
# Fonte municipal de resultados + metas
# ============================================================


def load_target_source() -> pd.DataFrame:

    if not TARGET_SOURCE_PATH.exists():

        raise FileNotFoundError(
            "Fonte de metas municipais não encontrada:\n"
            f"{TARGET_SOURCE_PATH}"
        )


    df = pd.read_csv(
        TARGET_SOURCE_PATH,
        dtype={
            "id_municipio": "string",
        },
    )


    df[
        "id_municipio"
    ] = normalize_id(
        df[
            "id_municipio"
        ]
    )


    required = {
        "ano",
        "id_municipio",
        "rede",
        "taxa_alfabetizacao",
        "percentual_participacao",
    }


    required.update(
        {
            f"meta_alfabetizacao_{year}"
            for year in FUTURE_YEARS
        }
    )


    missing = sorted(
        required
        - set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Colunas ausentes na fonte de metas: "
            f"{missing}"
        )


    networks = set(
        df[
            "rede"
        ]
        .dropna()
        .unique()
    )


    if networks != {
        "Municipal",
    }:

        raise RuntimeError(
            "Universo de rede inesperado: "
            f"{networks}"
        )


    assert (
        df[
            "id_municipio"
        ]
        .nunique()
        == 5_352
    )


    year_counts = (
        df[
            "ano"
        ]
        .value_counts()
    )


    assert (
        int(
            year_counts.get(
                2023,
                0,
            )
        )
        == 5_352
    )


    assert (
        int(
            year_counts.get(
                2024,
                0,
            )
        )
        == 5_352
    )


    duplicate_max = (
        df
        .groupby(
            [
                "ano",
                "id_municipio",
                "rede",
            ]
        )
        .size()
        .max()
    )


    assert (
        duplicate_max
        == 1
    )


    latest = (
        df[
            df[
                "ano"
            ]
            == BASE_YEAR
        ]
    )


    assert (
        latest[
            "taxa_alfabetizacao"
        ]
        .notna()
        .sum()
        == 5_352
    )


    for year in FUTURE_YEARS:

        column = (
            f"meta_alfabetizacao_{year}"
        )


        assert (
            latest[
                column
            ]
            .notna()
            .sum()
            == 5_352
        )


    return df


# ============================================================
# Painel municipal
# ============================================================


def build_target_panel(
    source: pd.DataFrame,
) -> pd.DataFrame:

    observed = (
        source[
            [
                "ano",
                "id_municipio",
                "taxa_alfabetizacao",
                "percentual_participacao",
            ]
        ]
        .pivot(
            index="id_municipio",
            columns="ano",
        )
    )


    observed.columns = [
        (
            f"{metric}_{year}"
        )
        for metric, year in (
            observed.columns
        )
    ]


    observed = (
        observed
        .reset_index()
    )


    latest = (
        source[
            source[
                "ano"
            ]
            == BASE_YEAR
        ][
            [
                "id_municipio",
                "rede",
            ]
            + [
                f"meta_alfabetizacao_{year}"
                for year in FUTURE_YEARS
            ]
        ]
        .copy()
    )


    panel = (
        latest
        .merge(
            observed,
            how="left",
            on="id_municipio",
            validate="one_to_one",
        )
        .rename(
            columns={
                "taxa_alfabetizacao_2023": (
                    "taxa_observada_2023"
                ),

                "taxa_alfabetizacao_2024": (
                    "taxa_observada_2024"
                ),

                "percentual_participacao_2023": (
                    "participacao_2023"
                ),

                "percentual_participacao_2024": (
                    "participacao_2024"
                ),
            }
        )
    )


    assert (
        len(
            panel
        )
        == 5_352
    )


    panel[
        "possui_historico_2023"
    ] = (
        panel[
            "taxa_observada_2023"
        ]
        .notna()
    )


    assert (
        int(
            panel[
                "possui_historico_2023"
            ]
            .sum()
        )
        == 5_232
    )


    assert (
        int(
            (
                ~panel[
                    "possui_historico_2023"
                ]
            )
            .sum()
        )
        == 120
    )


    panel[
        "ganho_observado_2023_2024_pp"
    ] = (
        panel[
            "taxa_observada_2024"
        ]
        - panel[
            "taxa_observada_2023"
        ]
    )


    return panel


# ============================================================
# Trajetória
#
# Distinguimos:
#
# cenário bruto:
#
#   taxa_2024 + ganho_recente * horizonte
#
# Pode ficar abaixo de 0 ou acima de 100.
#
# NÃO é uma taxa prevista.
# É utilizado somente para medir distância de trajetória.
#
# cenário limitado:
#
#   clip(cenário bruto, 0, 100)
#
# É mantido apenas para leitura humana.
#
# déficit de trajetória:
#
#   meta - cenário bruto
#
# > 0:
# trajetória recente insuficiente para a meta.
#
# <= 0:
# trajetória recente seria suficiente.
# ============================================================


def add_trajectory_metrics(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        panel.copy()
    )


    latest_rate = (
        result[
            "taxa_observada_2024"
        ]
    )


    recent_gain = (
        result[
            "ganho_observado_2023_2024_pp"
        ]
    )


    for year in FUTURE_YEARS:

        horizon = (
            year
            - BASE_YEAR
        )


        target_column = (
            f"meta_alfabetizacao_{year}"
        )


        gap_column = (
            f"gap_atual_meta_{year}_pp"
        )


        required_column = (
            f"ganho_anual_requerido_{year}_pp"
        )


        raw_scenario_column = (
            f"cenario_tendencia_bruto_{year}"
        )


        bounded_scenario_column = (
            f"cenario_tendencia_{year}"
        )


        trajectory_deficit_column = (
            f"deficit_trajetoria_{year}_pp"
        )


        bounded_gap_column = (
            f"gap_cenario_limitado_{year}_pp"
        )


        flag_column = (
            f"flag_risco_nao_atingir_{year}"
        )


        target = (
            result[
                target_column
            ]
        )


        # ----------------------------------------------------
        # Distância atual até a meta.
        # Pode ser negativa caso a meta já esteja superada.
        # ----------------------------------------------------

        result[
            gap_column
        ] = (
            target
            - latest_rate
        )


        # ----------------------------------------------------
        # Ganho médio anual necessário.
        #
        # Meta já atingida:
        # necessidade adicional = 0.
        # ----------------------------------------------------

        result[
            required_column
        ] = (
            np.maximum(
                result[
                    gap_column
                ],
                0.0,
            )
            / horizon
        )


        # ----------------------------------------------------
        # Cenário BRUTO.
        #
        # Não interpretar como taxa prevista.
        # ----------------------------------------------------

        result[
            raw_scenario_column
        ] = (
            latest_rate
            + (
                recent_gain
                * horizon
            )
        )


        # ----------------------------------------------------
        # Cenário limitado 0-100 apenas para apresentação.
        # ----------------------------------------------------

        result[
            bounded_scenario_column
        ] = (
            result[
                raw_scenario_column
            ]
            .clip(
                lower=0.0,
                upper=100.0,
            )
        )


        # ----------------------------------------------------
        # Indicador principal.
        #
        # NÃO é taxa.
        #
        # Mede quanto a trajetória recente está distante
        # da trajetória necessária.
        # ----------------------------------------------------

        result[
            trajectory_deficit_column
        ] = (
            target
            - result[
                raw_scenario_column
            ]
        )


        # ----------------------------------------------------
        # Gap do cenário limitado.
        #
        # Apenas para comunicação.
        # ----------------------------------------------------

        result[
            bounded_gap_column
        ] = (
            target
            - result[
                bounded_scenario_column
            ]
        )


        # ----------------------------------------------------
        # Flag de risco.
        #
        # NA sem histórico 2023.
        # ----------------------------------------------------

        flag = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )


        history_mask = (
            result[
                "possui_historico_2023"
            ]
        )


        flag.loc[
            history_mask
        ] = (
            result.loc[
                history_mask,
                trajectory_deficit_column,
            ]
            > EPSILON
        )


        result[
            flag_column
        ] = flag


    # ========================================================
    # Classificação executiva 2025
    # ========================================================

    current_met = (
        result[
            "taxa_observada_2024"
        ]
        >= result[
            "meta_alfabetizacao_2025"
        ]
    )


    result[
        "meta_2025_ja_atingida_em_2024"
    ] = (
        current_met
    )


    history = (
        result[
            "possui_historico_2023"
        ]
    )


    risk = (
        result[
            "flag_risco_nao_atingir_2025"
        ]
        .fillna(
            False
        )
    )


    result[
        "status_meta_2025"
    ] = (
        "SEM_HISTORICO_2023"
    )


    result.loc[
        (
            history
            & current_met
            & ~risk
        ),
        "status_meta_2025",
    ] = (
        "META_ATINGIDA_TRAJETORIA_SUSTENTA"
    )


    result.loc[
        (
            history
            & current_met
            & risk
        ),
        "status_meta_2025",
    ] = (
        "META_ATINGIDA_MAS_TRAJETORIA_RISCO"
    )


    result.loc[
        (
            history
            & ~current_met
            & ~risk
        ),
        "status_meta_2025",
    ] = (
        "ABAIXO_META_EM_TRAJETORIA"
    )


    result.loc[
        (
            history
            & ~current_met
            & risk
        ),
        "status_meta_2025",
    ] = (
        "ABAIXO_META_EM_RISCO"
    )


    # --------------------------------------------------------
    # Ranking principal.
    #
    # SOMENTE municípios sinalizados.
    #
    # Maior déficit de trajetória = maior prioridade.
    #
    # Não existe mais saturação causada pelo clip 0-100.
    # --------------------------------------------------------

    result[
        "rank_risco_meta_2025"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )


    eligible = (
        history
        & risk
    )


    result.loc[
        eligible,
        "rank_risco_meta_2025",
    ] = (
        result.loc[
            eligible,
            "deficit_trajetoria_2025_pp",
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            "Int64"
        )
    )


    # --------------------------------------------------------
    # Ranking complementar apenas do gap atual.
    #
    # Não é o ranking principal.
    # Serve para auditoria.
    # --------------------------------------------------------

    result[
        "rank_gap_atual_2025"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )


    below_target = (
        result[
            "gap_atual_meta_2025_pp"
        ]
        > EPSILON
    )


    result.loc[
        below_target,
        "rank_gap_atual_2025",
    ] = (
        result.loc[
            below_target,
            "gap_atual_meta_2025_pp",
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(
            "Int64"
        )
    )


    return result


# ============================================================
# Contexto de clustering
#
# SOMENTE pós-hoc.
# ============================================================


def load_cluster_context() -> pd.DataFrame:

    if not CLUSTER_PATH.exists():

        raise FileNotFoundError(
            "Resultado final do clustering não encontrado:\n"
            f"{CLUSTER_PATH}"
        )


    df = pd.read_csv(
        CLUSTER_PATH,
        dtype={
            "id_municipio": "string",
        },
    )


    df[
        "id_municipio"
    ] = normalize_id(
        df[
            "id_municipio"
        ]
    )


    required = {
        "id_municipio",
        "municipio",
        "sigla_uf",
        "regiao",
        "cluster_id",
    }


    missing = sorted(
        required
        - set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Contexto de clustering incompleto: "
            f"{missing}"
        )


    result = (
        df[
            [
                "id_municipio",
                "municipio",
                "sigla_uf",
                "regiao",
                "cluster_id",
            ]
        ]
        .copy()
    )


    result[
        "cluster_nome"
    ] = (
        result[
            "cluster_id"
        ]
        .map(
            CLUSTER_NAMES
        )
    )


    assert (
        result[
            "id_municipio"
        ]
        .nunique()
        == 5_517
    )


    return result


def add_posthoc_context(
    panel: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        panel
        .merge(
            context,
            how="left",
            on="id_municipio",
            validate="one_to_one",
        )
    )


    if (
        result[
            "municipio"
        ]
        .isna()
        .any()
    ):

        missing = (
            result.loc[
                result[
                    "municipio"
                ]
                .isna(),
                "id_municipio",
            ]
            .head(20)
            .tolist()
        )


        raise RuntimeError(
            "Municípios sem contexto territorial: "
            f"{missing}"
        )


    return result


# ============================================================
# Resumo por horizonte
# ============================================================


def build_horizon_summary(
    result: pd.DataFrame,
) -> pd.DataFrame:

    historical = (
        result[
            result[
                "possui_historico_2023"
            ]
        ]
        .copy()
    )


    rows = []


    for year in FUTURE_YEARS:

        target_column = (
            f"meta_alfabetizacao_{year}"
        )


        gap_column = (
            f"gap_atual_meta_{year}_pp"
        )


        required_column = (
            f"ganho_anual_requerido_{year}_pp"
        )


        deficit_column = (
            f"deficit_trajetoria_{year}_pp"
        )


        flag_column = (
            f"flag_risco_nao_atingir_{year}"
        )


        risk = (
            historical[
                flag_column
            ]
            .fillna(
                False
            )
        )


        current_met = (
            historical[
                "taxa_observada_2024"
            ]
            >= historical[
                target_column
            ]
        )


        risk_below = (
            risk
            & ~current_met
        )


        risk_current_met = (
            risk
            & current_met
        )


        rows.append(
            {
                "ano_meta": (
                    year
                ),

                "municipios_com_historico": (
                    int(
                        len(
                            historical
                        )
                    )
                ),

                "municipios_meta_ja_atingida_2024": (
                    int(
                        current_met.sum()
                    )
                ),

                "municipios_risco_cenario_tendencia": (
                    int(
                        risk.sum()
                    )
                ),

                "municipios_abaixo_meta_em_risco": (
                    int(
                        risk_below.sum()
                    )
                ),

                "municipios_meta_atingida_mas_trajetoria_risco": (
                    int(
                        risk_current_met.sum()
                    )
                ),

                "share_risco_cenario_tendencia": (
                    float(
                        risk.mean()
                    )
                ),

                "gap_atual_medio_pp": (
                    float(
                        historical[
                            gap_column
                        ]
                        .mean()
                    )
                ),

                "ganho_anual_requerido_medio_pp": (
                    float(
                        historical[
                            required_column
                        ]
                        .mean()
                    )
                ),

                "deficit_trajetoria_medio_entre_risco_pp": (
                    float(
                        historical.loc[
                            risk,
                            deficit_column,
                        ]
                        .mean()
                    )
                    if risk.any()
                    else 0.0
                ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Resumo por cluster
# ============================================================


def build_cluster_summary(
    result: pd.DataFrame,
) -> pd.DataFrame:

    valid = (
        result[
            result[
                "possui_historico_2023"
            ]
        ]
        .copy()
    )


    rows = []


    for (
        cluster_id,
        cluster_name
    ), group in (
        valid.groupby(
            [
                "cluster_id",
                "cluster_nome",
            ],
            dropna=False,
        )
    ):

        risk = (
            group[
                "flag_risco_nao_atingir_2025"
            ]
            .fillna(
                False
            )
        )


        rows.append(
            {
                "cluster_id": (
                    int(
                        cluster_id
                    )
                ),

                "cluster_nome": (
                    cluster_name
                ),

                "n_municipios_com_historico": (
                    int(
                        len(
                            group
                        )
                    )
                ),

                "n_risco_nao_atingir_2025": (
                    int(
                        risk.sum()
                    )
                ),

                "share_risco_nao_atingir_2025": (
                    float(
                        risk.mean()
                    )
                ),

                "taxa_observada_2024_media": (
                    float(
                        group[
                            "taxa_observada_2024"
                        ]
                        .mean()
                    )
                ),

                "meta_2025_media": (
                    float(
                        group[
                            "meta_alfabetizacao_2025"
                        ]
                        .mean()
                    )
                ),

                "ganho_2023_2024_medio_pp": (
                    float(
                        group[
                            "ganho_observado_2023_2024_pp"
                        ]
                        .mean()
                    )
                ),

                "ganho_anual_requerido_2025_medio_pp": (
                    float(
                        group[
                            "ganho_anual_requerido_2025_pp"
                        ]
                        .mean()
                    )
                ),

                "deficit_trajetoria_2025_medio_entre_risco_pp": (
                    float(
                        group.loc[
                            risk,
                            "deficit_trajetoria_2025_pp",
                        ]
                        .mean()
                    )
                    if risk.any()
                    else 0.0
                ),
            }
        )


    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "share_risco_nao_atingir_2025",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Resumo por região
# ============================================================


def build_region_summary(
    result: pd.DataFrame,
) -> pd.DataFrame:

    valid = (
        result[
            result[
                "possui_historico_2023"
            ]
        ]
        .copy()
    )


    rows = []


    for region, group in (
        valid.groupby(
            "regiao",
            dropna=False,
        )
    ):

        risk = (
            group[
                "flag_risco_nao_atingir_2025"
            ]
            .fillna(
                False
            )
        )


        rows.append(
            {
                "regiao": (
                    region
                ),

                "n_municipios_com_historico": (
                    int(
                        len(
                            group
                        )
                    )
                ),

                "n_risco_nao_atingir_2025": (
                    int(
                        risk.sum()
                    )
                ),

                "share_risco_nao_atingir_2025": (
                    float(
                        risk.mean()
                    )
                ),

                "taxa_observada_2024_media": (
                    float(
                        group[
                            "taxa_observada_2024"
                        ]
                        .mean()
                    )
                ),

                "meta_2025_media": (
                    float(
                        group[
                            "meta_alfabetizacao_2025"
                        ]
                        .mean()
                    )
                ),

                "ganho_2023_2024_medio_pp": (
                    float(
                        group[
                            "ganho_observado_2023_2024_pp"
                        ]
                        .mean()
                    )
                ),

                "deficit_trajetoria_2025_medio_entre_risco_pp": (
                    float(
                        group.loc[
                            risk,
                            "deficit_trajetoria_2025_pp",
                        ]
                        .mean()
                    )
                    if risk.any()
                    else 0.0
                ),
            }
        )


    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "share_risco_nao_atingir_2025",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Gráfico ranking 2025
# ============================================================


def create_top_2025_plot(
    top: pd.DataFrame,
) -> None:

    plot_data = (
        top
        .head(20)
        .copy()
    )


    if plot_data.empty:

        raise RuntimeError(
            "Nenhum município sinalizado para 2025."
        )


    plot_data[
        "label"
    ] = (
        plot_data[
            "municipio"
        ]
        .astype(str)
        + " / "
        + plot_data[
            "sigla_uf"
        ]
        .astype(str)
    )


    plot_data = (
        plot_data
        .sort_values(
            "deficit_trajetoria_2025_pp",
            ascending=True,
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            11,
            8,
        )
    )


    ax.barh(
        plot_data[
            "label"
        ],
        plot_data[
            "deficit_trajetoria_2025_pp"
        ],
    )


    ax.set_title(
        "Maior risco de não atingir a meta de 2025"
        "\nDéficit entre trajetória recente e trajetória necessária"
    )


    ax.set_xlabel(
        "Déficit de trajetória "
        "(pontos percentuais)"
    )


    ax.set_ylabel(
        "Município / UF"
    )


    ax.grid(
        axis="x",
        alpha=0.25,
    )


    fig.tight_layout()


    fig.savefig(
        TOP_IMAGE_OUTPUT_PATH,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# Gráfico de horizonte
#
# Agora mostra:
#
# - proporção sinalizada;
# - déficit médio entre os sinalizados.
# ============================================================


def create_horizon_plot(
    summary: pd.DataFrame,
) -> None:

    fig, ax_left = plt.subplots(
        figsize=(
            10,
            6,
        )
    )


    ax_left.plot(
        summary[
            "ano_meta"
        ],
        summary[
            "share_risco_cenario_tendencia"
        ],
        marker="o",
        label=(
            "Proporção sinalizada"
        ),
    )


    ax_left.set_xticks(
        FUTURE_YEARS
    )


    ax_left.set_ylim(
        0.0,
        1.0,
    )


    ax_left.set_xlabel(
        "Ano da meta"
    )


    ax_left.set_ylabel(
        "Proporção de municípios sinalizados"
    )


    ax_left.grid(
        alpha=0.25,
    )


    ax_right = (
        ax_left.twinx()
    )


    ax_right.plot(
        summary[
            "ano_meta"
        ],
        summary[
            "deficit_trajetoria_medio_entre_risco_pp"
        ],
        marker="s",
        linestyle="--",
        label=(
            "Déficit médio de trajetória"
        ),
    )


    ax_right.set_ylabel(
        "Déficit médio entre sinalizados "
        "(p.p.)"
    )


    ax_left.set_title(
        "Triagem de risco de não atingir metas"
        "\nCenário de continuidade da variação 2023→2024"
    )


    lines_left, labels_left = (
        ax_left
        .get_legend_handles_labels()
    )


    lines_right, labels_right = (
        ax_right
        .get_legend_handles_labels()
    )


    ax_left.legend(
        (
            lines_left
            + lines_right
        ),
        (
            labels_left
            + labels_right
        ),
        loc="upper left",
    )


    fig.tight_layout()


    fig.savefig(
        HORIZON_IMAGE_OUTPUT_PATH,
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

    total_start = (
        time.perf_counter()
    )


    print(
        "=" * 72
    )


    print(
        "RISCO DE NÃO ATINGIR METAS FUTURAS"
    )


    print(
        "=" * 72
    )


    print(
        "Método: triagem de trajetória."
    )


    print(
        "Não é previsão de série temporal."
    )


    print(
        "HGB supervisionado NÃO participa do ranking."
    )


    # ========================================================
    # 1. Fonte
    # ========================================================

    source = (
        load_target_source()
    )


    print(
        f"\n✓ Municípios: "
        f"{source['id_municipio'].nunique():,}"
    )


    # ========================================================
    # 2. Painel e trajetória
    # ========================================================

    panel = (
        build_target_panel(
            source
        )
    )


    panel = (
        add_trajectory_metrics(
            panel
        )
    )


    # ========================================================
    # 3. Contexto pós-hoc
    # ========================================================

    context = (
        load_cluster_context()
    )


    result = (
        add_posthoc_context(
            panel=panel,
            context=context,
        )
    )


    # ========================================================
    # 4. Ranking principal
    # ========================================================

    top_2025 = (
        result[
            result[
                "flag_risco_nao_atingir_2025"
            ]
            .fillna(
                False
            )
        ]
        .sort_values(
            [
                "deficit_trajetoria_2025_pp",
                "gap_atual_meta_2025_pp",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            TOP_N
        )
        .copy()
    )


    no_history = (
        result[
            ~result[
                "possui_historico_2023"
            ]
        ]
        .sort_values(
            "gap_atual_meta_2025_pp",
            ascending=False,
        )
        .copy()
    )


    # ========================================================
    # 5. Resumos
    # ========================================================

    horizon_summary = (
        build_horizon_summary(
            result
        )
    )


    cluster_summary = (
        build_cluster_summary(
            result
        )
    )


    region_summary = (
        build_region_summary(
            result
        )
    )


    # ========================================================
    # 6. Persistência
    # ========================================================

    result.to_csv(
        ALL_OUTPUT_PATH,
        index=False,
    )


    top_columns = [
        "rank_risco_meta_2025",
        "rank_gap_atual_2025",
        "id_municipio",
        "municipio",
        "sigla_uf",
        "regiao",
        "cluster_id",
        "cluster_nome",
        "taxa_observada_2023",
        "taxa_observada_2024",
        "ganho_observado_2023_2024_pp",
        "meta_alfabetizacao_2025",
        "gap_atual_meta_2025_pp",
        "ganho_anual_requerido_2025_pp",
        "cenario_tendencia_bruto_2025",
        "cenario_tendencia_2025",
        "deficit_trajetoria_2025_pp",
        "participacao_2024",
        "status_meta_2025",
    ]


    top_2025[
        top_columns
    ].to_csv(
        TOP_2025_OUTPUT_PATH,
        index=False,
    )


    no_history.to_csv(
        NO_HISTORY_OUTPUT_PATH,
        index=False,
    )


    horizon_summary.to_csv(
        HORIZON_SUMMARY_OUTPUT_PATH,
        index=False,
    )


    cluster_summary.to_csv(
        CLUSTER_SUMMARY_OUTPUT_PATH,
        index=False,
    )


    region_summary.to_csv(
        REGION_SUMMARY_OUTPUT_PATH,
        index=False,
    )


    # ========================================================
    # 7. Gráficos
    # ========================================================

    create_top_2025_plot(
        top_2025
    )


    create_horizon_plot(
        horizon_summary
    )


    # ========================================================
    # 8. Metadados
    # ========================================================

    status_counts = {
        str(key): int(
            value
        )
        for key, value in (
            result[
                "status_meta_2025"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    }


    metadata = {
        "analysis": (
            "future_target_risk_screening"
        ),

        "base_year": (
            BASE_YEAR
        ),

        "target_years": (
            FUTURE_YEARS
        ),

        "primary_target_year": (
            PRIMARY_TARGET_YEAR
        ),

        "network_universe": (
            "Municipal"
        ),

        "n_municipalities": (
            int(
                len(
                    result
                )
            )
        ),

        "n_with_2023_history": (
            int(
                result[
                    "possui_historico_2023"
                ]
                .sum()
            )
        ),

        "n_without_2023_history": (
            int(
                (
                    ~result[
                        "possui_historico_2023"
                    ]
                )
                .sum()
            )
        ),

        "method": (
            "trajectory screening based on "
            "observed 2023-2024 change versus "
            "future municipal targets"
        ),

        "bounded_projection_use": (
            "only for human-readable scenario; "
            "not used for ranking"
        ),

        "trajectory_deficit_definition": (
            "target - unbounded linear continuation "
            "of the observed 2023-2024 change"
        ),

        "primary_ranking": (
            "descending trajectory deficit "
            "against 2025 target"
        ),

        "supervised_hgb_used_in_ranking": (
            False
        ),

        "cluster_used_for_ranking": (
            False
        ),

        "cluster_used_posthoc": (
            True
        ),

        "status_2025_counts": (
            status_counts
        ),

        "important_limitations": [
            (
                "Only two observed municipal years "
                "are available."
            ),

            (
                "The trajectory is a screening "
                "scenario and not a robust "
                "time-series forecast."
            ),

            (
                "Large one-year changes may reflect "
                "real change and/or instability in "
                "small municipal cohorts."
            ),

            (
                "The unbounded trajectory is used "
                "only as a distance-to-required-"
                "trajectory index, never as a "
                "literal literacy rate."
            ),

            (
                "Municipalities without a 2023 rate "
                "cannot receive a trend-based flag."
            ),

            (
                "Future targets were not predictors "
                "of the supervised model."
            ),
        ],

        "runtime_seconds": (
            float(
                time.perf_counter()
                - total_start
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
            default=json_default,
        )


    # ========================================================
    # 9. Console
    # ========================================================

    elapsed = (
        time.perf_counter()
        - total_start
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "STATUS META 2025"
    )


    print(
        "=" * 72
    )


    print(
        result[
            "status_meta_2025"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "TOP 20 — PRIORIDADE DE TRAJETÓRIA 2025"
    )


    print(
        "=" * 72
    )


    print(
        top_2025[
            [
                "rank_risco_meta_2025",
                "municipio",
                "sigla_uf",
                "taxa_observada_2023",
                "taxa_observada_2024",
                "ganho_observado_2023_2024_pp",
                "meta_alfabetizacao_2025",
                "gap_atual_meta_2025_pp",
                "deficit_trajetoria_2025_pp",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "POR HORIZONTE"
    )


    print(
        "=" * 72
    )


    print(
        horizon_summary
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "POR CLUSTER"
    )


    print(
        "=" * 72
    )


    print(
        cluster_summary
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "POR REGIÃO"
    )


    print(
        "=" * 72
    )


    print(
        region_summary
        .to_string(
            index=False
        )
    )


    print(
        f"\nTempo total: "
        f"{elapsed:.2f}s"
    )


    print(
        "\nArquivos gerados:"
    )


    paths = [
        ALL_OUTPUT_PATH,
        TOP_2025_OUTPUT_PATH,
        NO_HISTORY_OUTPUT_PATH,
        HORIZON_SUMMARY_OUTPUT_PATH,
        CLUSTER_SUMMARY_OUTPUT_PATH,
        REGION_SUMMARY_OUTPUT_PATH,
        METADATA_OUTPUT_PATH,
        TOP_IMAGE_OUTPUT_PATH,
        HORIZON_IMAGE_OUTPUT_PATH,
    ]


    for path in paths:

        print(
            "  ✓ "
            + str(
                path.relative_to(
                    PROJECT_ROOT
                )
            )
        )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "TRIAGEM DE METAS CONCLUÍDA"
    )


    print(
        "=" * 72
    )


    print(
        "Interpretar como screening de trajetória, "
        "não como forecast determinístico."
    )


if __name__ == "__main__":
    main()