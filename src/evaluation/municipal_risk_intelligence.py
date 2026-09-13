from __future__ import annotations

import gzip
import json
import time
import zlib
from pathlib import Path
from urllib.request import Request, urlopen

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
# Pergunta de negócio:
#
# "Quais municípios apresentam maior risco educacional?"
#
# Regras:
#
# - nenhuma nova modelagem;
# - nenhum retuning;
# - nenhuma alteração de threshold;
# - nenhuma alteração de features;
# - utiliza exclusivamente previsões congeladas de 2024.
#
# Leituras produzidas:
#
# 1. severidade do risco
#    -> probabilidade média prevista de risco
#
# 2. carga estimada de risco
#    -> n_alunos * probabilidade_media_risco
#
# O resultado real de 2024 é mantido somente como referência
# descritiva e NÃO participa da construção dos rankings.
# ============================================================


TOP_N = 30

MIN_STUDENTS_RATE_RANK = 100


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "hgb_final_test_2024_municipalities.csv"
)


REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "business"
)


IMAGES_PATH = (
    PROJECT_ROOT
    / "images"
)


REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference"
)


REPORTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


IMAGES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


REFERENCE_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# API oficial de Localidades do IBGE
#
# É usada exclusivamente para obter o nome dos municípios.
#
# O nome NÃO participa do modelo.
# ============================================================


IBGE_MUNICIPALITIES_URL = (
    "https://servicodados.ibge.gov.br/"
    "api/v1/localidades/municipios"
)


IBGE_CACHE_PATH = (
    REFERENCE_PATH
    / "ibge_municipios.csv"
)


# ============================================================
# Saídas
# ============================================================


ALL_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_risk_2024_all.csv"
)


TOP_PROBABILITY_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_risk_2024_top_probability.csv"
)


TOP_BURDEN_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_risk_2024_top_burden.csv"
)


NEW_TERRITORIES_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_risk_2024_new_territories.csv"
)


REGIONS_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_risk_2024_regions.csv"
)


SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_risk_2024.json"
)


TOP_IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "municipal_risk_top20_2024.png"
)


# ============================================================
# Colunas esperadas
# ============================================================


REQUIRED_COLUMNS = [
    "rank_risco_modelo",
    "id_municipio",
    "sigla_uf",
    "regiao",
    "municipio_visto_2023",
    "uf_vista_2023",
    "n_alunos",
    "taxa_alfabetizacao_real",
    "taxa_risco_real",
    "probabilidade_media_alfabetizado",
    "probabilidade_media_risco",
    "taxa_sinalizada_risco",
]


EXECUTIVE_COLUMNS = [
    "id_municipio",
    "municipio",
    "sigla_uf",
    "regiao",
    "status_dominio_modelo",
    "n_alunos",
    "probabilidade_media_risco",
    "carga_risco_estimada",
    "taxa_sinalizada_risco",
    "taxa_risco_real",
    "taxa_alfabetizacao_real",
    "rank_probabilidade_risco_geral",
    "rank_carga_risco_geral",
    "percentil_risco_modelo",
    "faixa_risco_relativo",
]


# ============================================================
# Helpers
# ============================================================


def normalize_bool(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(
            bool
        )


    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }


    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


    if normalized.isna().any():

        invalid = (
            series.loc[
                normalized.isna()
            ]
            .drop_duplicates()
            .tolist()
        )


        raise ValueError(
            "Valores booleanos inesperados: "
            f"{invalid}"
        )


    return normalized.astype(
        bool
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
        "Objeto não serializável: "
        f"{type(value)}"
    )


# ============================================================
# Carregamento das previsões municipais congeladas
# ============================================================


def load_municipality_predictions() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Resultado municipal do teste final "
            "não encontrado:\n"
            f"{INPUT_PATH}"
        )


    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "id_municipio": "string",
        },
    )


    # --------------------------------------------------------
    # Remover índice acidental salvo em CSV, se existir.
    # --------------------------------------------------------

    index_like_columns = [
        column
        for column in df.columns
        if (
            column.startswith(
                "Unnamed:"
            )
            or column == "index"
        )
    ]


    if index_like_columns:

        df = df.drop(
            columns=index_like_columns,
        )


    missing_columns = sorted(
        set(
            REQUIRED_COLUMNS
        )
        - set(
            df.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            "Colunas ausentes no arquivo municipal: "
            f"{missing_columns}"
        )


    df[
        "id_municipio"
    ] = (
        df[
            "id_municipio"
        ]
        .astype(str)
        .str.zfill(7)
    )


    df[
        "municipio_visto_2023"
    ] = normalize_bool(
        df[
            "municipio_visto_2023"
        ]
    )


    df[
        "uf_vista_2023"
    ] = normalize_bool(
        df[
            "uf_vista_2023"
        ]
    )


    # --------------------------------------------------------
    # Reconciliação com teste temporal final.
    # --------------------------------------------------------

    assert (
        len(df)
        == 5_517
    )


    assert (
        df[
            "id_municipio"
        ]
        .nunique()
        == 5_517
    )


    assert (
        int(
            df[
                "municipio_visto_2023"
            ]
            .sum()
        )
        == 4_841
    )


    assert (
        int(
            (
                ~df[
                    "municipio_visto_2023"
                ]
            )
            .sum()
        )
        == 676
    )


    unseen_ufs = set(
        df.loc[
            ~df[
                "uf_vista_2023"
            ],
            "sigla_uf",
        ]
        .dropna()
        .unique()
    )


    assert unseen_ufs == {
        "AC",
        "DF",
        "SP",
    }


    return df


# ============================================================
# Decodificação robusta da resposta HTTP
#
# A API do IBGE pode retornar conteúdo compactado.
#
# O erro anteriormente observado:
#
# UnicodeDecodeError:
# byte 0x8b in position 1
#
# é compatível com conteúdo GZIP:
#
# magic bytes = 1f 8b
#
# Portanto descompactamos antes do decode.
# ============================================================


def decode_http_payload(
    raw: bytes,
    content_encoding: str | None,
) -> str:

    encoding = (
        content_encoding
        or ""
    ).strip().lower()


    # --------------------------------------------------------
    # GZIP indicado pelo header.
    # --------------------------------------------------------

    if "gzip" in encoding:

        raw = gzip.decompress(
            raw
        )


    # --------------------------------------------------------
    # GZIP identificado pelos magic bytes.
    #
    # Essa proteção adicional é necessária porque alguns
    # servidores/proxies podem devolver gzip sem informar
    # corretamente Content-Encoding.
    # --------------------------------------------------------

    elif raw.startswith(
        b"\x1f\x8b"
    ):

        raw = gzip.decompress(
            raw
        )


    # --------------------------------------------------------
    # Deflate.
    # --------------------------------------------------------

    elif "deflate" in encoding:

        try:

            raw = zlib.decompress(
                raw
            )

        except zlib.error:

            raw = zlib.decompress(
                raw,
                -zlib.MAX_WBITS,
            )


    # --------------------------------------------------------
    # utf-8-sig também remove BOM caso exista.
    # --------------------------------------------------------

    return raw.decode(
        "utf-8-sig"
    )


# ============================================================
# Lookup oficial de nomes de municípios
#
# Depois da primeira execução, o lookup fica salvo em:
#
# data/reference/ibge_municipios.csv
#
# Assim as próximas execuções não dependem da API.
# ============================================================


def download_ibge_lookup() -> pd.DataFrame:

    # --------------------------------------------------------
    # Reutilizar cache local.
    # --------------------------------------------------------

    if IBGE_CACHE_PATH.exists():

        lookup = pd.read_csv(
            IBGE_CACHE_PATH,
            dtype={
                "id_municipio": "string",
            },
        )


        lookup[
            "id_municipio"
        ] = (
            lookup[
                "id_municipio"
            ]
            .astype(str)
            .str.zfill(7)
        )


        assert {
            "id_municipio",
            "municipio",
        }.issubset(
            lookup.columns
        )


        print(
            "✓ Lookup IBGE carregado do cache."
        )


        return lookup


    print(
        "Baixando nomes oficiais dos municípios "
        "na API de Localidades do IBGE..."
    )


    request = Request(
        IBGE_MUNICIPALITIES_URL,
        headers={
            "User-Agent": (
                "FIAP-Tech-Challenge-Fase3/1.0"
            ),

            # O servidor pode escolher gzip.
            # O script sabe descompactar.
            "Accept-Encoding": (
                "gzip, deflate, identity"
            ),

            "Accept": (
                "application/json"
            ),
        },
    )


    try:

        with urlopen(
            request,
            timeout=60,
        ) as response:

            raw = response.read()

            content_encoding = (
                response.headers.get(
                    "Content-Encoding"
                )
            )


    except Exception as exc:

        raise RuntimeError(
            "Não foi possível consultar a API "
            "de Localidades do IBGE.\n"
            "Verifique a conexão com a internet "
            "e execute novamente."
        ) from exc


    try:

        decoded = decode_http_payload(
            raw=raw,
            content_encoding=(
                content_encoding
            ),
        )


        payload = json.loads(
            decoded
        )


    except Exception as exc:

        raise RuntimeError(
            "A API do IBGE respondeu, mas o conteúdo "
            "não pôde ser decodificado como JSON.\n"
            f"Content-Encoding recebido: "
            f"{content_encoding!r}\n"
            f"Primeiros bytes: "
            f"{raw[:20]!r}"
        ) from exc


    if not isinstance(
        payload,
        list,
    ):

        raise RuntimeError(
            "Formato inesperado na resposta do IBGE. "
            "Era esperada uma lista de municípios."
        )


    rows = []


    for item in payload:

        municipality_id = item.get(
            "id"
        )


        municipality_name = item.get(
            "nome"
        )


        if (
            municipality_id is None
            or municipality_name is None
        ):

            continue


        rows.append(
            {
                "id_municipio": (
                    str(
                        municipality_id
                    )
                    .zfill(7)
                ),

                "municipio": (
                    str(
                        municipality_name
                    )
                ),
            }
        )


    lookup = (
        pd.DataFrame(
            rows
        )
        .drop_duplicates(
            subset=[
                "id_municipio"
            ]
        )
        .sort_values(
            "id_municipio"
        )
        .reset_index(
            drop=True
        )
    )


    if (
        len(
            lookup
        )
        < 5_500
    ):

        raise RuntimeError(
            "Lookup retornado pelo IBGE possui "
            "menos municípios que o esperado: "
            f"{len(lookup):,}"
        )


    lookup.to_csv(
        IBGE_CACHE_PATH,
        index=False,
    )


    print(
        f"✓ Lookup IBGE salvo: "
        f"{len(lookup):,} municípios."
    )


    return lookup


# ============================================================
# Status de domínio territorial
#
# DOMINIO_FORTE
#   Município e UF presentes em 2023.
#
# DOMINIO_INTERMEDIARIO
#   Município novo, mas UF presente no desenvolvimento.
#
# FORA_DOMINIO_UF
#   UF completamente ausente de 2023.
#
# Essa classificação NÃO altera previsões.
# ============================================================


def assign_domain_status(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df.copy()
    )


    result[
        "status_dominio_modelo"
    ] = np.select(
        condlist=[
            (
                ~result[
                    "uf_vista_2023"
                ]
            ),

            (
                result[
                    "municipio_visto_2023"
                ]
                & result[
                    "uf_vista_2023"
                ]
            ),
        ],

        choicelist=[
            "FORA_DOMINIO_UF",
            "DOMINIO_FORTE",
        ],

        default=(
            "DOMINIO_INTERMEDIARIO"
        ),
    )


    counts = (
        result[
            "status_dominio_modelo"
        ]
        .value_counts()
    )


    assert (
        int(
            counts.get(
                "DOMINIO_FORTE",
                0,
            )
        )
        == 4_841
    )


    assert (
        int(
            counts.get(
                "FORA_DOMINIO_UF",
                0,
            )
        )
        == 665
    )


    assert (
        int(
            counts.get(
                "DOMINIO_INTERMEDIARIO",
                0,
            )
        )
        == 11
    )


    return result


# ============================================================
# Métricas de negócio
# ============================================================


def add_business_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df.copy()
    )


    # --------------------------------------------------------
    # Carga esperada de risco.
    #
    # n * média das probabilidades
    #
    # equivale à soma das probabilidades individuais.
    #
    # Não deve ser interpretada como quantidade observada
    # de alunos não alfabetizados.
    # --------------------------------------------------------

    result[
        "carga_risco_estimada"
    ] = (
        result[
            "n_alunos"
        ]
        * result[
            "probabilidade_media_risco"
        ]
    )


    # --------------------------------------------------------
    # Apenas diagnóstico.
    # Não participa dos rankings.
    # --------------------------------------------------------

    result[
        "gap_probabilidade_vs_risco_real"
    ] = (
        result[
            "probabilidade_media_risco"
        ]
        - result[
            "taxa_risco_real"
        ]
    )


    # --------------------------------------------------------
    # Ranking geral por severidade.
    # --------------------------------------------------------

    result[
        "rank_probabilidade_risco_geral"
    ] = (
        result[
            "probabilidade_media_risco"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


    # --------------------------------------------------------
    # Ranking geral por carga.
    # --------------------------------------------------------

    result[
        "rank_carga_risco_geral"
    ] = (
        result[
            "carga_risco_estimada"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


    # --------------------------------------------------------
    # Percentil de risco.
    #
    # Valores próximos de 1 representam maior risco relativo.
    # --------------------------------------------------------

    result[
        "percentil_risco_modelo"
    ] = (
        result[
            "probabilidade_media_risco"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )


    result[
        "faixa_risco_relativo"
    ] = pd.cut(
        result[
            "percentil_risco_modelo"
        ],
        bins=[
            0.0,
            0.50,
            0.75,
            0.90,
            1.00,
        ],
        labels=[
            "ate_P50",
            "P50_P75",
            "P75_P90",
            "top_10_percent",
        ],
        include_lowest=True,
    )


    return result


# ============================================================
# Resumo por região
# ============================================================


def summarize_regions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    scopes = {
        "todos_2024": (
            df
        ),

        "dominio_forte": (
            df[
                df[
                    "status_dominio_modelo"
                ]
                == "DOMINIO_FORTE"
            ]
        ),
    }


    for scope_name, frame in (
        scopes.items()
    ):

        for region, group in (
            frame.groupby(
                "regiao",
                dropna=False,
            )
        ):

            total_students = float(
                group[
                    "n_alunos"
                ]
                .sum()
            )


            if (
                total_students
                <= 0
            ):
                continue


            predicted_risk_weighted = float(
                (
                    group[
                        "probabilidade_media_risco"
                    ]
                    * group[
                        "n_alunos"
                    ]
                )
                .sum()
                / total_students
            )


            actual_risk_weighted = float(
                (
                    group[
                        "taxa_risco_real"
                    ]
                    * group[
                        "n_alunos"
                    ]
                )
                .sum()
                / total_students
            )


            signaled_risk_weighted = float(
                (
                    group[
                        "taxa_sinalizada_risco"
                    ]
                    * group[
                        "n_alunos"
                    ]
                )
                .sum()
                / total_students
            )


            rows.append(
                {
                    "escopo": (
                        scope_name
                    ),

                    "regiao": (
                        region
                    ),

                    "n_municipios": (
                        int(
                            group[
                                "id_municipio"
                            ]
                            .nunique()
                        )
                    ),

                    "n_alunos": (
                        int(
                            total_students
                        )
                    ),

                    "probabilidade_media_risco_ponderada": (
                        predicted_risk_weighted
                    ),

                    "taxa_risco_real_ponderada": (
                        actual_risk_weighted
                    ),

                    "taxa_sinalizada_risco_ponderada": (
                        signaled_risk_weighted
                    ),

                    "carga_risco_estimada": (
                        float(
                            group[
                                "carga_risco_estimada"
                            ]
                            .sum()
                        )
                    ),

                    "media_probabilidade_municipios": (
                        float(
                            group[
                                "probabilidade_media_risco"
                            ]
                            .mean()
                        )
                    ),

                    "mediana_probabilidade_municipios": (
                        float(
                            group[
                                "probabilidade_media_risco"
                            ]
                            .median()
                        )
                    ),
                }
            )


    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "escopo",
                "probabilidade_media_risco_ponderada",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    return result


# ============================================================
# Gráfico executivo
# ============================================================


def create_top_risk_plot(
    top_probability: pd.DataFrame,
) -> None:

    plot_data = (
        top_probability
        .head(20)
        .copy()
    )


    if plot_data.empty:

        raise RuntimeError(
            "Ranking executivo vazio."
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
            "probabilidade_media_risco",
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
            "probabilidade_media_risco"
        ],
    )


    ax.set_title(
        "Municípios com maior risco previsto — 2024"
        "\nDomínio territorial forte e mínimo de 100 alunos"
    )


    ax.set_xlabel(
        "Probabilidade média prevista de risco"
    )


    ax.set_ylabel(
        "Município / UF"
    )


    max_probability = float(
        plot_data[
            "probabilidade_media_risco"
        ]
        .max()
    )


    ax.set_xlim(
        left=0.0,
        right=max(
            0.80,
            max_probability
            * 1.05,
        ),
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
# Main
# ============================================================


def main() -> None:

    start = (
        time.perf_counter()
    )


    print(
        "=" * 72
    )

    print(
        "INTELIGÊNCIA MUNICIPAL DE RISCO — 2024"
    )

    print(
        "=" * 72
    )

    print(
        "Nenhuma alteração será realizada "
        "no modelo final."
    )


    # ========================================================
    # 1. Carregar previsões municipais congeladas
    # ========================================================

    print(
        "\nCarregando resultado municipal..."
    )


    df = (
        load_municipality_predictions()
    )


    print(
        f"✓ Municípios: "
        f"{len(df):,}"
    )


    # ========================================================
    # 2. Obter nomes oficiais dos municípios
    # ========================================================

    lookup = (
        download_ibge_lookup()
    )


    df = (
        df
        .merge(
            lookup,
            how="left",
            on="id_municipio",
            validate="one_to_one",
        )
    )


    missing_names = (
        df[
            "municipio"
        ]
        .isna()
    )


    if missing_names.any():

        missing_ids = (
            df.loc[
                missing_names,
                [
                    "id_municipio",
                    "sigla_uf",
                ],
            ]
            .head(20)
            .to_dict(
                orient="records"
            )
        )


        raise RuntimeError(
            "Existem municípios sem nome após "
            "o join com IBGE:\n"
            f"{missing_ids}"
        )


    print(
        "✓ Todos os municípios possuem nome."
    )


    # ========================================================
    # 3. Status territorial
    # ========================================================

    df = assign_domain_status(
        df
    )


    print(
        "\nDomínio do modelo:"
    )


    print(
        df[
            "status_dominio_modelo"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # 4. Métricas de inteligência municipal
    # ========================================================

    df = add_business_metrics(
        df
    )


    # ========================================================
    # 5. Ranking por severidade
    #
    # Ranking executivo:
    #
    # - somente DOMINIO_FORTE;
    # - pelo menos 100 alunos;
    # - somente probabilidade prevista;
    # - resultado real não participa.
    # ========================================================

    strong_domain = (
        df[
            df[
                "status_dominio_modelo"
            ]
            == "DOMINIO_FORTE"
        ]
        .copy()
    )


    rate_eligible = (
        strong_domain[
            strong_domain[
                "n_alunos"
            ]
            >= MIN_STUDENTS_RATE_RANK
        ]
        .copy()
    )


    rate_eligible[
        "rank_executivo_probabilidade"
    ] = (
        rate_eligible[
            "probabilidade_media_risco"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


    top_probability = (
        rate_eligible
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
        .head(
            TOP_N
        )
        .copy()
    )


    # ========================================================
    # 6. Ranking por carga estimada
    # ========================================================

    strong_domain[
        "rank_executivo_carga"
    ] = (
        strong_domain[
            "carga_risco_estimada"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


    top_burden = (
        strong_domain
        .sort_values(
            [
                "carga_risco_estimada",
                "probabilidade_media_risco",
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


    # ========================================================
    # 7. Territórios novos / fora do domínio
    # ========================================================

    new_territories = (
        df[
            df[
                "status_dominio_modelo"
            ]
            != "DOMINIO_FORTE"
        ]
        .sort_values(
            [
                "status_dominio_modelo",
                "probabilidade_media_risco",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .copy()
    )


    # ========================================================
    # 8. Resumo regional
    # ========================================================

    regions = summarize_regions(
        df
    )


    # ========================================================
    # 9. Diagnóstico municipal
    #
    # A correlação usa o resultado real de 2024 apenas para
    # avaliar se o ranking previsto acompanha a ordenação
    # observada.
    #
    # Ela NÃO participa do ranking.
    # ========================================================

    spearman_strong = float(
        strong_domain[
            "probabilidade_media_risco"
        ]
        .corr(
            strong_domain[
                "taxa_risco_real"
            ],
            method="spearman",
        )
    )


    # ========================================================
    # 10. Persistência
    # ========================================================

    all_columns = [
        "id_municipio",
        "municipio",
        "sigla_uf",
        "regiao",
        "municipio_visto_2023",
        "uf_vista_2023",
        "status_dominio_modelo",
        "n_alunos",
        "taxa_alfabetizacao_real",
        "taxa_risco_real",
        "probabilidade_media_alfabetizado",
        "probabilidade_media_risco",
        "taxa_sinalizada_risco",
        "carga_risco_estimada",
        "gap_probabilidade_vs_risco_real",
        "rank_probabilidade_risco_geral",
        "rank_carga_risco_geral",
        "percentil_risco_modelo",
        "faixa_risco_relativo",
    ]


    (
        df[
            all_columns
        ]
        .sort_values(
            "rank_probabilidade_risco_geral"
        )
        .to_csv(
            ALL_OUTPUT_PATH,
            index=False,
        )
    )


    probability_columns = (
        [
            "rank_executivo_probabilidade",
        ]
        + EXECUTIVE_COLUMNS
    )


    top_probability[
        probability_columns
    ].to_csv(
        TOP_PROBABILITY_OUTPUT_PATH,
        index=False,
    )


    burden_columns = (
        [
            "rank_executivo_carga",
        ]
        + EXECUTIVE_COLUMNS
    )


    top_burden[
        burden_columns
    ].to_csv(
        TOP_BURDEN_OUTPUT_PATH,
        index=False,
    )


    new_territories[
        EXECUTIVE_COLUMNS
    ].to_csv(
        NEW_TERRITORIES_OUTPUT_PATH,
        index=False,
    )


    regions.to_csv(
        REGIONS_OUTPUT_PATH,
        index=False,
    )


    create_top_risk_plot(
        top_probability
    )


    # ========================================================
    # 11. Resumo JSON
    # ========================================================

    status_counts = {
        str(key): int(
            value
        )
        for key, value in (
            df[
                "status_dominio_modelo"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    }


    summary = {
        "analysis": (
            "municipal_risk_intelligence"
        ),

        "year": 2024,

        "ranking_uses_frozen_predictions": True,

        "model_changed": False,

        "threshold_changed": False,

        "n_municipalities": (
            int(
                len(df)
            )
        ),

        "n_students": (
            int(
                df[
                    "n_alunos"
                ]
                .sum()
            )
        ),

        "domain_status_counts": (
            status_counts
        ),

        "main_probability_ranking": {
            "scope": (
                "DOMINIO_FORTE"
            ),

            "minimum_students": (
                MIN_STUDENTS_RATE_RANK
            ),

            "criterion": (
                "descending mean predicted "
                "risk probability"
            ),

            "observed_2024_target_used_for_ranking": (
                False
            ),
        },

        "burden_ranking": {
            "scope": (
                "DOMINIO_FORTE"
            ),

            "criterion": (
                "descending estimated risk burden"
            ),

            "definition": (
                "n_alunos * "
                "probabilidade_media_risco"
            ),

            "observed_2024_target_used_for_ranking": (
                False
            ),
        },

        "spearman_predicted_vs_actual_risk_"
        "strong_domain": (
            spearman_strong
        ),

        "top_10_probability": (
            top_probability[
                [
                    "rank_executivo_probabilidade",
                    "id_municipio",
                    "municipio",
                    "sigla_uf",
                    "regiao",
                    "n_alunos",
                    "probabilidade_media_risco",
                    "carga_risco_estimada",
                    "taxa_risco_real",
                ]
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        ),

        "top_10_burden": (
            top_burden[
                [
                    "rank_executivo_carga",
                    "id_municipio",
                    "municipio",
                    "sigla_uf",
                    "regiao",
                    "n_alunos",
                    "probabilidade_media_risco",
                    "carga_risco_estimada",
                    "taxa_risco_real",
                ]
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        ),

        "interpretation_notes": [
            (
                "Risk ranking represents model "
                "predictions and not causal effects."
            ),

            (
                "Observed 2024 literacy outcomes are "
                "reported only as descriptive reference "
                "and do not determine ranking."
            ),

            (
                "UFs absent from 2023 are reported "
                "separately because temporal validation "
                "showed weaker generalization."
            ),

            (
                "Probability ranking represents severity; "
                "burden ranking represents estimated "
                "absolute volume."
            ),
        ],
    }


    with open(
        SUMMARY_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


    # ========================================================
    # 12. Console
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP 20 — SEVERIDADE DO RISCO"
    )

    print(
        "=" * 72
    )


    print(
        top_probability[
            [
                "rank_executivo_probabilidade",
                "municipio",
                "sigla_uf",
                "n_alunos",
                "probabilidade_media_risco",
                "taxa_risco_real",
                "carga_risco_estimada",
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
        "TOP 20 — CARGA ESTIMADA DE RISCO"
    )

    print(
        "=" * 72
    )


    print(
        top_burden[
            [
                "rank_executivo_carga",
                "municipio",
                "sigla_uf",
                "n_alunos",
                "probabilidade_media_risco",
                "carga_risco_estimada",
                "taxa_risco_real",
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
        "RESUMO REGIONAL"
    )

    print(
        "=" * 72
    )


    print(
        regions.to_string(
            index=False
        )
    )


    print(
        "\nSpearman município — "
        "risco previsto x risco real "
        "(domínio forte): "
        f"{spearman_strong:.4f}"
    )


    print(
        f"\nTempo total: "
        f"{elapsed:.2f}s"
    )


    print(
        "\nArquivos gerados:"
    )


    output_paths = [
        ALL_OUTPUT_PATH,
        TOP_PROBABILITY_OUTPUT_PATH,
        TOP_BURDEN_OUTPUT_PATH,
        NEW_TERRITORIES_OUTPUT_PATH,
        REGIONS_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
        TOP_IMAGE_OUTPUT_PATH,
        IBGE_CACHE_PATH,
    ]


    for path in output_paths:

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
        "ANÁLISE MUNICIPAL CONCLUÍDA"
    )

    print(
        "=" * 72
    )

    print(
        "Nenhum resultado foi utilizado "
        "para retuning do modelo."
    )


if __name__ == "__main__":
    main()