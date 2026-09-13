from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Tech Challenge Fase 3
#
# APRENDIZAGEM NÃO SUPERVISIONADA
#
# Etapa:
# clustering municipal final com K = 3.
#
# Objetivo:
#
# responder à pergunta:
#
# "Quais municípios e regiões apresentam perfis semelhantes?"
#
# IMPORTANTE:
#
# O clustering NÃO utiliza:
#
# - alfabetização;
# - risco previsto;
# - meta;
# - UF;
# - região;
# - id_municipio.
#
# Essas informações são adicionadas somente DEPOIS
# da formação dos clusters, para interpretação.
#
# O PCA também NÃO é usado para formar os clusters.
#
# K = 3 foi escolhido após avaliação conjunta de:
#
# - Silhouette Score;
# - Davies-Bouldin Index;
# - Inertia;
# - tamanho dos clusters;
# - interpretabilidade.
# ============================================================


RANDOM_STATE = 42

FINAL_K = 3

N_INIT = 30


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# Entradas
# ============================================================


PROFILE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "clustering"
    / "municipal_profiles_2024.csv"
)


PCA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "clustering"
    / "municipal_profiles_pca_2024.csv"
)


CANDIDATE_LABELS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "clustering"
    / "kmeans_candidate_labels_2024.csv"
)


K_EVALUATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "clustering"
    / "kmeans_k_evaluation_2024.csv"
)


RISK_PATH = (
    PROJECT_ROOT
    / "reports"
    / "business"
    / "municipal_risk_2024_all.csv"
)


# ============================================================
# Saídas
# ============================================================


REPORTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "clustering"
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


FINAL_MUNICIPAL_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_final_municipalities_2024.csv"
)


CLUSTER_SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_final_cluster_summary_2024.csv"
)


FEATURE_PROFILE_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_final_feature_profiles_2024.csv"
)


REGION_COMPOSITION_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_final_region_composition_2024.csv"
)


UF_COMPOSITION_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_final_uf_composition_2024.csv"
)


METADATA_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_final_2024.json"
)


PCA_IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "kmeans_final_pca_2024.png"
)


PROFILE_IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "kmeans_final_profile_heatmap_2024.png"
)


# ============================================================
# Features usadas na clusterização
# ============================================================


SOCIOECONOMIC_FEATURES = [
    "populacao_2020",
    "pib_por_habitante_2020",
    "participacao_agropecuaria_2020",
    "participacao_industria_2020",
    "participacao_administracao_publica_2020",
]


EDUCATIONAL_FEATURES = [
    "quantidade_escolas_anos_iniciais",
    "proporcao_escolas_rurais",
    "proporcao_internet_aprendizagem",
    "proporcao_biblioteca_sala_leitura",
    "proporcao_laboratorio_informatica",
    "alunos_por_turma_anos_iniciais",
    "razao_matriculas_docentes_anos_iniciais",
    "proporcao_matriculas_integral_anos_iniciais",
]


CLUSTER_FEATURES = (
    SOCIOECONOMIC_FEATURES
    + EDUCATIONAL_FEATURES
)


# ============================================================
# Variáveis transformadas com log1p antes do StandardScaler.
#
# Devem ser idênticas à etapa de diagnóstico de K.
# ============================================================


LOG_FEATURES = [
    "populacao_2020",
    "pib_por_habitante_2020",
    "quantidade_escolas_anos_iniciais",
    "alunos_por_turma_anos_iniciais",
]


# ============================================================
# Variáveis explicitamente proibidas na formação dos clusters.
# ============================================================


FORBIDDEN_CLUSTER_FEATURES = {
    "id_municipio",
    "sigla_uf",
    "regiao",
    "capital_uf",
    "amazonia_legal",
    "alfabetizado",
    "taxa_risco_real",
    "taxa_alfabetizacao_real",
    "probabilidade_media_risco",
    "probabilidade_media_alfabetizado",
    "taxa_sinalizada_risco",
    "meta_alfabetizacao",
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
# Auditoria metodológica
# ============================================================


def audit_cluster_features() -> None:

    overlap = (
        set(
            CLUSTER_FEATURES
        )
        & FORBIDDEN_CLUSTER_FEATURES
    )


    if overlap:

        raise RuntimeError(
            "Feature proibida presente no clustering: "
            f"{sorted(overlap)}"
        )


    assert (
        len(
            CLUSTER_FEATURES
        )
        == 13
    )


# ============================================================
# Carregamento dos perfis
# ============================================================


def load_profiles() -> pd.DataFrame:

    if not PROFILE_PATH.exists():

        raise FileNotFoundError(
            "Perfil municipal não encontrado: "
            f"{PROFILE_PATH}"
        )


    df = pd.read_csv(
        PROFILE_PATH,
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


    missing_features = sorted(
        set(
            CLUSTER_FEATURES
        )
        - set(
            df.columns
        )
    )


    if missing_features:

        raise RuntimeError(
            "Features ausentes no perfil municipal: "
            f"{missing_features}"
        )


    return df


# ============================================================
# Preprocessing
#
# Deve reproduzir exatamente a etapa anterior:
#
# 1. log1p;
# 2. mediana;
# 3. StandardScaler.
# ============================================================


def prepare_matrix(
    profiles: pd.DataFrame,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    Pipeline,
]:

    X = (
        profiles[
            CLUSTER_FEATURES
        ]
        .copy()
    )


    for feature in LOG_FEATURES:

        negative_mask = (
            X[
                feature
            ]
            .dropna()
            < 0
        )


        if negative_mask.any():

            raise RuntimeError(
                "Valor negativo em feature log1p: "
                f"{feature}"
            )


        X[
            feature
        ] = np.log1p(
            X[
                feature
            ]
        )


    preprocessing = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


    X_scaled = (
        preprocessing
        .fit_transform(
            X
        )
    )


    if not np.isfinite(
        X_scaled
    ).all():

        raise RuntimeError(
            "Matriz final possui NaN ou infinito."
        )


    scaled_df = pd.DataFrame(
        X_scaled,
        columns=[
            f"scaled__{feature}"
            for feature in CLUSTER_FEATURES
        ],
        index=profiles.index,
    )


    return (
        X_scaled,
        scaled_df,
        preprocessing,
    )


# ============================================================
# Ajuste final do K-Means
# ============================================================


def fit_final_kmeans(
    X_scaled: np.ndarray,
) -> tuple[
    KMeans,
    np.ndarray,
]:

    model = KMeans(
        n_clusters=FINAL_K,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )


    labels = (
        model.fit_predict(
            X_scaled
        )
    )


    assert (
        len(
            np.unique(
                labels
            )
        )
        == FINAL_K
    )


    return (
        model,
        labels,
    )


# ============================================================
# Auditoria contra labels do diagnóstico anterior
#
# Cluster IDs podem, em princípio, trocar de número.
# Por isso usamos Adjusted Rand Index.
#
# ARI = 1 significa partição idêntntica.
# ============================================================


def audit_candidate_labels(
    profiles: pd.DataFrame,
    final_labels: np.ndarray,
) -> float:

    if not CANDIDATE_LABELS_PATH.exists():

        raise FileNotFoundError(
            "Labels candidatos não encontrados: "
            f"{CANDIDATE_LABELS_PATH}"
        )


    candidates = pd.read_csv(
        CANDIDATE_LABELS_PATH,
        dtype={
            "id_municipio": "string",
        },
    )


    candidates[
        "id_municipio"
    ] = normalize_id(
        candidates[
            "id_municipio"
        ]
    )


    if (
        "cluster_k3"
        not in candidates.columns
    ):

        raise RuntimeError(
            "Coluna cluster_k3 não encontrada."
        )


    final_frame = pd.DataFrame(
        {
            "id_municipio": (
                profiles[
                    "id_municipio"
                ]
                .to_numpy()
            ),

            "cluster_final": (
                final_labels
            ),
        }
    )


    comparison = (
        final_frame
        .merge(
            candidates[
                [
                    "id_municipio",
                    "cluster_k3",
                ]
            ],
            how="inner",
            on="id_municipio",
            validate="one_to_one",
        )
    )


    assert (
        len(
            comparison
        )
        == 5_517
    )


    ari = float(
        adjusted_rand_score(
            comparison[
                "cluster_k3"
            ],
            comparison[
                "cluster_final"
            ],
        )
    )


    if not np.isclose(
        ari,
        1.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            "Clustering final não reproduziu "
            "a partição candidata K=3. "
            f"ARI={ari}"
        )


    return ari


# ============================================================
# Auditoria das métricas K=3
# ============================================================


def calculate_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
) -> dict:

    return {
        "inertia": (
            float(
                model.inertia_
            )
        ),

        "silhouette_score": (
            float(
                silhouette_score(
                    X_scaled,
                    labels,
                )
            )
        ),

        "davies_bouldin_score": (
            float(
                davies_bouldin_score(
                    X_scaled,
                    labels,
                )
            )
        ),

        "calinski_harabasz_score": (
            float(
                calinski_harabasz_score(
                    X_scaled,
                    labels,
                )
            )
        ),
    }


def audit_previous_metrics(
    metrics: dict,
) -> dict:

    if not K_EVALUATION_PATH.exists():

        raise FileNotFoundError(
            "Avaliação de K não encontrada: "
            f"{K_EVALUATION_PATH}"
        )


    previous = pd.read_csv(
        K_EVALUATION_PATH
    )


    row = (
        previous[
            previous[
                "k"
            ]
            == FINAL_K
        ]
    )


    if (
        len(
            row
        )
        != 1
    ):

        raise RuntimeError(
            "K=3 não localizado unicamente "
            "na avaliação anterior."
        )


    row = (
        row.iloc[0]
    )


    checks = {
        "inertia": (
            "inertia"
        ),

        "silhouette_score": (
            "silhouette_score"
        ),

        "davies_bouldin_score": (
            "davies_bouldin_score"
        ),

        "calinski_harabasz_score": (
            "calinski_harabasz_score"
        ),
    }


    differences = {}


    for current_name, previous_name in (
        checks.items()
    ):

        difference = float(
            abs(
                metrics[
                    current_name
                ]
                - row[
                    previous_name
                ]
            )
        )


        differences[
            current_name
        ] = (
            difference
        )


        if (
            difference
            > 1e-8
        ):

            raise RuntimeError(
                "Métrica final diferente "
                "da avaliação anterior.\n"
                f"{current_name}: "
                f"{difference}"
            )


    return differences


# ============================================================
# PCA já calculado anteriormente
#
# PCA é usado SOMENTE para visualização.
# ============================================================


def load_pca() -> pd.DataFrame:

    if not PCA_PATH.exists():

        raise FileNotFoundError(
            "PCA municipal não encontrado: "
            f"{PCA_PATH}"
        )


    pca = pd.read_csv(
        PCA_PATH,
        dtype={
            "id_municipio": "string",
        },
    )


    pca[
        "id_municipio"
    ] = normalize_id(
        pca[
            "id_municipio"
        ]
    )


    required = {
        "id_municipio",
        "pca_1",
        "pca_2",
    }


    if not required.issubset(
        pca.columns
    ):

        raise RuntimeError(
            "Colunas PCA ausentes."
        )


    assert (
        pca[
            "id_municipio"
        ]
        .nunique()
        == 5_517
    )


    return pca[
        [
            "id_municipio",
            "pca_1",
            "pca_2",
        ]
    ]


# ============================================================
# Inteligência municipal pós-clustering
#
# IMPORTANTE:
#
# risco e alfabetização são adicionados somente depois
# dos clusters já estarem formados.
# ============================================================


def load_risk_context() -> pd.DataFrame:

    if not RISK_PATH.exists():

        raise FileNotFoundError(
            "Arquivo de inteligência municipal "
            "não encontrado: "
            f"{RISK_PATH}"
        )


    risk = pd.read_csv(
        RISK_PATH,
        dtype={
            "id_municipio": "string",
        },
    )


    risk[
        "id_municipio"
    ] = normalize_id(
        risk[
            "id_municipio"
        ]
    )


    required = [
        "id_municipio",
        "municipio",
        "n_alunos",
        "probabilidade_media_risco",
        "taxa_risco_real",
        "taxa_alfabetizacao_real",
        "status_dominio_modelo",
    ]


    missing = sorted(
        set(
            required
        )
        - set(
            risk.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Colunas ausentes no arquivo de risco: "
            f"{missing}"
        )


    assert (
        risk[
            "id_municipio"
        ]
        .nunique()
        == 5_517
    )


    return risk[
        required
    ]


# ============================================================
# Tabela final por município
# ============================================================


def build_final_municipality_table(
    profiles: pd.DataFrame,
    labels: np.ndarray,
    pca: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:

    final = (
        profiles.copy()
    )


    final[
        "cluster_id"
    ] = (
        labels.astype(int)
    )


    final[
        "cluster_codigo"
    ] = (
        final[
            "cluster_id"
        ]
        .map(
            lambda value: (
                f"cluster_{value}"
            )
        )
    )


    final = (
        final
        .merge(
            pca,
            how="left",
            on="id_municipio",
            validate="one_to_one",
        )
        .merge(
            risk,
            how="left",
            on="id_municipio",
            validate="one_to_one",
        )
    )


    assert (
        len(
            final
        )
        == 5_517
    )


    assert (
        final[
            "cluster_id"
        ]
        .nunique()
        == FINAL_K
    )


    assert (
        final[
            "municipio"
        ]
        .isna()
        .sum()
        == 0
    )


    return final


# ============================================================
# Perfil das features por cluster
#
# Para cada feature:
#
# - média bruta;
# - mediana;
# - Q25;
# - Q75;
# - média padronizada no espaço usado pelo K-Means.
#
# centroid_z_model_space:
#
# > 0  = acima da média geral;
# < 0  = abaixo da média geral.
#
# Para features log1p, o centroid_z se refere ao espaço
# após log + padronização.
# ============================================================


def build_feature_profiles(
    profiles: pd.DataFrame,
    scaled_df: pd.DataFrame,
    labels: np.ndarray,
    model: KMeans,
) -> pd.DataFrame:

    rows = []


    labels_series = pd.Series(
        labels,
        index=profiles.index,
        name="cluster_id",
    )


    for cluster_id in range(
        FINAL_K
    ):

        mask = (
            labels_series
            == cluster_id
        )


        for feature_index, feature in enumerate(
            CLUSTER_FEATURES
        ):

            raw_values = (
                profiles.loc[
                    mask,
                    feature,
                ]
            )


            scaled_values = (
                scaled_df.loc[
                    mask,
                    f"scaled__{feature}",
                ]
            )


            rows.append(
                {
                    "cluster_id": (
                        cluster_id
                    ),

                    "cluster_codigo": (
                        f"cluster_{cluster_id}"
                    ),

                    "feature": (
                        feature
                    ),

                    "grupo_feature": (
                        "socioeconomica"
                        if feature
                        in SOCIOECONOMIC_FEATURES
                        else "educacional"
                    ),

                    "n_municipios": (
                        int(
                            mask.sum()
                        )
                    ),

                    "raw_mean": (
                        float(
                            raw_values.mean()
                        )
                    ),

                    "raw_median": (
                        float(
                            raw_values.median()
                        )
                    ),

                    "raw_q25": (
                        float(
                            raw_values.quantile(
                                0.25
                            )
                        )
                    ),

                    "raw_q75": (
                        float(
                            raw_values.quantile(
                                0.75
                            )
                        )
                    ),

                    "scaled_mean": (
                        float(
                            scaled_values.mean()
                        )
                    ),

                    "centroid_z_model_space": (
                        float(
                            model
                            .cluster_centers_[
                                cluster_id,
                                feature_index,
                            ]
                        )
                    ),
                }
            )


    result = pd.DataFrame(
        rows
    )


    return result


# ============================================================
# Resumo dos clusters
# ============================================================


def build_cluster_summary(
    final: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    for cluster_id, group in (
        final.groupby(
            "cluster_id"
        )
    ):

        n_municipalities = int(
            len(
                group
            )
        )


        total_students = float(
            group[
                "n_alunos"
            ]
            .sum()
        )


        weighted_predicted_risk = float(
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


        weighted_actual_risk = float(
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


        region_counts = (
            group[
                "regiao"
            ]
            .value_counts()
        )


        dominant_region = str(
            region_counts.index[0]
        )


        dominant_region_share = float(
            region_counts.iloc[0]
            / n_municipalities
        )


        strong_domain_share = float(
            (
                group[
                    "status_dominio_modelo"
                ]
                == "DOMINIO_FORTE"
            )
            .mean()
        )


        rows.append(
            {
                "cluster_id": (
                    int(
                        cluster_id
                    )
                ),

                "cluster_codigo": (
                    f"cluster_{cluster_id}"
                ),

                "n_municipios": (
                    n_municipalities
                ),

                "share_municipios": (
                    float(
                        n_municipalities
                        / len(
                            final
                        )
                    )
                ),

                "n_ufs": (
                    int(
                        group[
                            "sigla_uf"
                        ]
                        .nunique()
                    )
                ),

                "n_alunos": (
                    int(
                        total_students
                    )
                ),

                "regiao_predominante": (
                    dominant_region
                ),

                "share_regiao_predominante": (
                    dominant_region_share
                ),

                "probabilidade_risco_ponderada": (
                    weighted_predicted_risk
                ),

                "taxa_risco_real_ponderada": (
                    weighted_actual_risk
                ),

                "probabilidade_risco_media_municipios": (
                    float(
                        group[
                            "probabilidade_media_risco"
                        ]
                        .mean()
                    )
                ),

                "probabilidade_risco_mediana_municipios": (
                    float(
                        group[
                            "probabilidade_media_risco"
                        ]
                        .median()
                    )
                ),

                "taxa_alfabetizacao_real_ponderada": (
                    float(
                        1.0
                        - weighted_actual_risk
                    )
                ),

                "share_dominio_forte": (
                    strong_domain_share
                ),
            }
        )


    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "cluster_id"
        )
        .reset_index(
            drop=True
        )
    )


    return result


# ============================================================
# Composição regional
# ============================================================


def build_region_composition(
    final: pd.DataFrame,
) -> pd.DataFrame:

    counts = (
        final
        .groupby(
            [
                "cluster_id",
                "regiao",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "n_municipios"
        )
        .reset_index()
    )


    cluster_totals = (
        counts
        .groupby(
            "cluster_id"
        )[
            "n_municipios"
        ]
        .sum()
        .rename(
            "cluster_total"
        )
        .reset_index()
    )


    region_totals = (
        counts
        .groupby(
            "regiao"
        )[
            "n_municipios"
        ]
        .sum()
        .rename(
            "region_total"
        )
        .reset_index()
    )


    result = (
        counts
        .merge(
            cluster_totals,
            on="cluster_id",
            how="left",
        )
        .merge(
            region_totals,
            on="regiao",
            how="left",
        )
    )


    result[
        "share_dentro_cluster"
    ] = (
        result[
            "n_municipios"
        ]
        / result[
            "cluster_total"
        ]
    )


    result[
        "share_da_regiao_no_cluster"
    ] = (
        result[
            "n_municipios"
        ]
        / result[
            "region_total"
        ]
    )


    result[
        "cluster_codigo"
    ] = (
        result[
            "cluster_id"
        ]
        .map(
            lambda value: (
                f"cluster_{value}"
            )
        )
    )


    return (
        result
        .sort_values(
            [
                "cluster_id",
                "n_municipios",
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


# ============================================================
# Composição por UF
# ============================================================


def build_uf_composition(
    final: pd.DataFrame,
) -> pd.DataFrame:

    counts = (
        final
        .groupby(
            [
                "cluster_id",
                "sigla_uf",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "n_municipios"
        )
        .reset_index()
    )


    cluster_totals = (
        counts
        .groupby(
            "cluster_id"
        )[
            "n_municipios"
        ]
        .sum()
        .rename(
            "cluster_total"
        )
        .reset_index()
    )


    result = (
        counts
        .merge(
            cluster_totals,
            how="left",
            on="cluster_id",
        )
    )


    result[
        "share_dentro_cluster"
    ] = (
        result[
            "n_municipios"
        ]
        / result[
            "cluster_total"
        ]
    )


    result[
        "cluster_codigo"
    ] = (
        result[
            "cluster_id"
        ]
        .map(
            lambda value: (
                f"cluster_{value}"
            )
        )
    )


    return (
        result
        .sort_values(
            [
                "cluster_id",
                "n_municipios",
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


# ============================================================
# Visualização PCA
# ============================================================


def create_pca_plot(
    final: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
    )


    for cluster_id in sorted(
        final[
            "cluster_id"
        ]
        .unique()
    ):

        subset = (
            final[
                final[
                    "cluster_id"
                ]
                == cluster_id
            ]
        )


        ax.scatter(
            subset[
                "pca_1"
            ],
            subset[
                "pca_2"
            ],
            s=14,
            alpha=0.55,
            label=(
                f"Cluster {cluster_id} "
                f"(n={len(subset):,})"
            ),
        )


    ax.set_title(
        "Perfis municipais — K-Means final (K=3)"
        "\nPCA utilizado apenas para visualização"
    )


    ax.set_xlabel(
        "Componente principal 1"
    )


    ax.set_ylabel(
        "Componente principal 2"
    )


    ax.legend()


    ax.grid(
        alpha=0.20,
    )


    fig.tight_layout()


    fig.savefig(
        PCA_IMAGE_OUTPUT_PATH,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# Heatmap do perfil padronizado
#
# Valor positivo:
# acima da média geral.
#
# Valor negativo:
# abaixo da média geral.
#
# Ainda NÃO damos nomes semânticos aos clusters.
# ============================================================


def create_profile_heatmap(
    feature_profiles: pd.DataFrame,
) -> None:

    matrix = (
        feature_profiles
        .pivot(
            index="feature",
            columns="cluster_id",
            values="centroid_z_model_space",
        )
        .loc[
            CLUSTER_FEATURES
        ]
    )


    fig, ax = plt.subplots(
        figsize=(
            10,
            9,
        )
    )


    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
    )


    ax.set_xticks(
        np.arange(
            FINAL_K
        )
    )


    ax.set_xticklabels(
        [
            f"Cluster {value}"
            for value in matrix.columns
        ]
    )


    ax.set_yticks(
        np.arange(
            len(
                matrix.index
            )
        )
    )


    ax.set_yticklabels(
        matrix.index
    )


    for row_index in range(
        matrix.shape[0]
    ):

        for column_index in range(
            matrix.shape[1]
        ):

            value = (
                matrix.iloc[
                    row_index,
                    column_index,
                ]
            )


            ax.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )


    colorbar = (
        fig.colorbar(
            image,
            ax=ax,
        )
    )


    colorbar.set_label(
        "Centroide padronizado "
        "(espaço usado pelo K-Means)"
    )


    ax.set_title(
        "Perfis estruturais dos clusters municipais"
        "\nValores padronizados em relação ao conjunto de 2024"
    )


    ax.set_xlabel(
        "Cluster"
    )


    ax.set_ylabel(
        "Feature"
    )


    fig.tight_layout()


    fig.savefig(
        PROFILE_IMAGE_OUTPUT_PATH,
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
        "K-MEANS FINAL — PERFIS MUNICIPAIS"
    )


    print(
        "=" * 72
    )


    print(
        "K congelado: 3"
    )


    print(
        "Alfabetização, risco, meta, UF e região "
        "NÃO entram na formação dos clusters."
    )


    # ========================================================
    # 1. Auditoria de features
    # ========================================================

    audit_cluster_features()


    # ========================================================
    # 2. Perfis municipais
    # ========================================================

    profiles = (
        load_profiles()
    )


    print(
        f"\n✓ Municípios: "
        f"{len(profiles):,}"
    )


    print(
        f"✓ Features: "
        f"{len(CLUSTER_FEATURES)}"
    )


    # ========================================================
    # 3. Preprocessing
    # ========================================================

    (
        X_scaled,
        scaled_df,
        preprocessing,
    ) = prepare_matrix(
        profiles
    )


    print(
        "✓ Preprocessing reproduzido."
    )


    # ========================================================
    # 4. K-Means final
    # ========================================================

    start_fit = (
        time.perf_counter()
    )


    (
        model,
        labels,
    ) = fit_final_kmeans(
        X_scaled
    )


    fit_seconds = (
        time.perf_counter()
        - start_fit
    )


    print(
        f"✓ K-Means K=3 ajustado em "
        f"{fit_seconds:.2f}s"
    )


    # ========================================================
    # 5. Auditorias de reprodutibilidade
    # ========================================================

    ari = audit_candidate_labels(
        profiles=profiles,
        final_labels=labels,
    )


    metrics = calculate_metrics(
        X_scaled=X_scaled,
        labels=labels,
        model=model,
    )


    metric_differences = (
        audit_previous_metrics(
            metrics
        )
    )


    print(
        f"✓ ARI contra cluster_k3 anterior: "
        f"{ari:.6f}"
    )


    print(
        "✓ Métricas reproduzem "
        "o diagnóstico anterior."
    )


    # ========================================================
    # 6. PCA e contexto de risco
    # ========================================================

    pca = load_pca()


    risk = load_risk_context()


    final = (
        build_final_municipality_table(
            profiles=profiles,
            labels=labels,
            pca=pca,
            risk=risk,
        )
    )


    # ========================================================
    # 7. Perfis dos clusters
    # ========================================================

    feature_profiles = (
        build_feature_profiles(
            profiles=profiles,
            scaled_df=scaled_df,
            labels=labels,
            model=model,
        )
    )


    cluster_summary = (
        build_cluster_summary(
            final
        )
    )


    region_composition = (
        build_region_composition(
            final
        )
    )


    uf_composition = (
        build_uf_composition(
            final
        )
    )


    # ========================================================
    # 8. Persistência
    # ========================================================

    final.to_csv(
        FINAL_MUNICIPAL_OUTPUT_PATH,
        index=False,
    )


    cluster_summary.to_csv(
        CLUSTER_SUMMARY_OUTPUT_PATH,
        index=False,
    )


    feature_profiles.to_csv(
        FEATURE_PROFILE_OUTPUT_PATH,
        index=False,
    )


    region_composition.to_csv(
        REGION_COMPOSITION_OUTPUT_PATH,
        index=False,
    )


    uf_composition.to_csv(
        UF_COMPOSITION_OUTPUT_PATH,
        index=False,
    )


    # ========================================================
    # 9. Visualizações
    # ========================================================

    create_pca_plot(
        final
    )


    create_profile_heatmap(
        feature_profiles
    )


    # ========================================================
    # 10. Metadados
    # ========================================================

    cluster_sizes = {
        str(
            cluster_id
        ): int(
            count
        )

        for cluster_id, count in (
            pd.Series(
                labels
            )
            .value_counts()
            .sort_index()
            .items()
        )
    }


    elapsed = (
        time.perf_counter()
        - total_start
    )


    metadata = {
        "analysis": (
            "final_kmeans_municipal_profiles"
        ),

        "year": 2024,

        "algorithm": (
            "KMeans"
        ),

        "final_k": (
            FINAL_K
        ),

        "selection_rationale": [
            (
                "K=2 had the highest Silhouette "
                "but K=3 remained close."
            ),

            (
                "K=3 achieved the lowest "
                "Davies-Bouldin score."
            ),

            (
                "The inertia curve showed a strong "
                "gain up to K=3 followed by smaller "
                "incremental improvements."
            ),

            (
                "K=3 produced three clusters with "
                "substantial and interpretable sizes."
            ),
        ],

        "cluster_features": (
            CLUSTER_FEATURES
        ),

        "forbidden_variables_not_used": (
            sorted(
                FORBIDDEN_CLUSTER_FEATURES
            )
        ),

        "log1p_features": (
            LOG_FEATURES
        ),

        "imputation": (
            "median"
        ),

        "scaling": (
            "StandardScaler"
        ),

        "n_init": (
            N_INIT
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "pca_used_for_clustering": (
            False
        ),

        "posthoc_interpretation_variables": [
            "sigla_uf",
            "regiao",
            "probabilidade_media_risco",
            "taxa_risco_real",
            "taxa_alfabetizacao_real",
        ],

        "n_municipalities": (
            int(
                len(
                    profiles
                )
            )
        ),

        "cluster_sizes": (
            cluster_sizes
        ),

        "validation_metrics": (
            metrics
        ),

        "adjusted_rand_vs_candidate_k3": (
            ari
        ),

        "metric_reproduction_differences": (
            metric_differences
        ),

        "clusters_semantically_named": (
            False
        ),

        "important_note": (
            "Cluster IDs are arbitrary numeric identifiers. "
            "Semantic names will be assigned only after "
            "examining the final structural profiles."
        ),

        "runtime_seconds": (
            float(
                elapsed
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
    # 11. Console
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )


    print(
        "MÉTRICAS FINAIS"
    )


    print(
        "=" * 72
    )


    print(
        f"Inertia: "
        f"{metrics['inertia']:.2f}"
    )


    print(
        f"Silhouette: "
        f"{metrics['silhouette_score']:.4f}"
    )


    print(
        f"Davies-Bouldin: "
        f"{metrics['davies_bouldin_score']:.4f}"
    )


    print(
        f"Calinski-Harabasz: "
        f"{metrics['calinski_harabasz_score']:.2f}"
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "TAMANHO DOS CLUSTERS"
    )


    print(
        "=" * 72
    )


    print(
        pd.Series(
            labels
        )
        .value_counts()
        .sort_index()
        .rename(
            "n_municipios"
        )
        .to_string()
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "RESUMO PÓS-HOC"
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
        "COMPOSIÇÃO REGIONAL"
    )


    print(
        "=" * 72
    )


    print(
        region_composition[
            [
                "cluster_id",
                "regiao",
                "n_municipios",
                "share_dentro_cluster",
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
        "CLUSTERING FINAL CONCLUÍDO"
    )


    print(
        "=" * 72
    )


    print(
        "Os clusters ainda NÃO receberam "
        "nomes semânticos."
    )


    print(
        "A próxima etapa será interpretar "
        "os perfis estruturais."
    )


    print(
        f"\nTempo total: "
        f"{elapsed:.2f}s"
    )


    print(
        "\nArquivos gerados:"
    )


    paths = [
        FINAL_MUNICIPAL_OUTPUT_PATH,
        CLUSTER_SUMMARY_OUTPUT_PATH,
        FEATURE_PROFILE_OUTPUT_PATH,
        REGION_COMPOSITION_OUTPUT_PATH,
        UF_COMPOSITION_OUTPUT_PATH,
        METADATA_OUTPUT_PATH,
        PCA_IMAGE_OUTPUT_PATH,
        PROFILE_IMAGE_OUTPUT_PATH,
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


if __name__ == "__main__":
    main()