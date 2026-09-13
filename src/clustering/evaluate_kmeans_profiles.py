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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
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
# Etapa 1:
# avaliação do número de clusters para perfis municipais.
#
# Objetivo:
#
# responder futuramente:
#
# "Quais municípios/regiões possuem perfis semelhantes?"
#
# Nesta etapa:
#
# - NÃO usamos alfabetização como feature;
# - NÃO usamos risco previsto como feature;
# - NÃO usamos metas como feature;
# - NÃO usamos UF como feature;
# - NÃO usamos região como feature;
# - NÃO usamos id_municipio como feature.
#
# O clustering utiliza somente características estruturais,
# socioeconômicas e educacionais.
#
# PCA será usado SOMENTE para visualização.
#
# K-Means será ajustado no espaço completo padronizado.
# ============================================================


RANDOM_STATE = 42

K_MIN = 2
K_MAX = 10

N_INIT = 30


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
    / "ano=2024"
)


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


# ============================================================
# Saídas
# ============================================================


PROFILE_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_profiles_2024.csv"
)


PCA_OUTPUT_PATH = (
    REPORTS_PATH
    / "municipal_profiles_pca_2024.csv"
)


EVALUATION_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_k_evaluation_2024.csv"
)


LABELS_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_candidate_labels_2024.csv"
)


METADATA_OUTPUT_PATH = (
    REPORTS_PATH
    / "kmeans_k_evaluation_2024.json"
)


INERTIA_IMAGE_PATH = (
    IMAGES_PATH
    / "kmeans_inertia_2024.png"
)


SILHOUETTE_IMAGE_PATH = (
    IMAGES_PATH
    / "kmeans_silhouette_2024.png"
)


DBI_IMAGE_PATH = (
    IMAGES_PATH
    / "kmeans_davies_bouldin_2024.png"
)


# ============================================================
# Variáveis de clustering
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
# Variáveis com transformação log1p
#
# São variáveis positivas e assimétricas.
#
# Seguimos a mesma lógica já utilizada na modelagem
# supervisionada para reduzir o efeito de caudas longas.
# ============================================================


LOG_FEATURES = [
    "populacao_2020",
    "pib_por_habitante_2020",
    "quantidade_escolas_anos_iniciais",
    "alunos_por_turma_anos_iniciais",
]


# ============================================================
# Colunas apenas descritivas
#
# Elas NÃO entram no K-Means.
# ============================================================


DESCRIPTOR_COLUMNS = [
    "id_municipio",
    "rede_nome",
    "sigla_uf",
    "regiao",
    "capital_uf",
    "amazonia_legal",
]


READ_COLUMNS = list(
    dict.fromkeys(
        DESCRIPTOR_COLUMNS
        + CLUSTER_FEATURES
    )
)


PUBLIC_NETWORKS = [
    "Municipal",
    "Estadual",
]


# ============================================================
# Carregamento
# ============================================================


def load_2024() -> pd.DataFrame:

    files = sorted(
        DATA_PATH.glob(
            "*.parquet"
        )
    )


    if not files:

        raise FileNotFoundError(
            "Nenhum arquivo Parquet encontrado em "
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


    assert (
        len(df)
        == 1_851_852
    )


    assert (
        df[
            "id_municipio"
        ]
        .isna()
        .sum()
        == 0
    )


    print(
        f"✓ Linhas 2024: {len(df):,}"
    )


    print(
        "✓ Municípios 2024: "
        f"{df['id_municipio'].nunique():,}"
    )


    return df


# ============================================================
# Construção do perfil municipal
#
# Problema:
#
# as features contextuais aparecem repetidas em cada aluno.
#
# Além disso, algumas features educacionais são específicas
# da combinação município + rede.
#
# Estratégia:
#
# 1. manter apenas redes públicas Municipal e Estadual;
# 2. reduzir para uma linha única por município + rede;
# 3. verificar que as features são constantes dentro dessa
#    unidade;
# 4. calcular média NÃO ponderada entre redes públicas;
# 5. obter uma única linha estrutural por município.
#
# Dessa maneira a quantidade de alunos da avaliação não
# determina artificialmente o perfil estrutural do município.
# ============================================================


def build_municipal_profiles(
    df: pd.DataFrame,
) -> pd.DataFrame:

    public = (
        df[
            df[
                "rede_nome"
            ]
            .isin(
                PUBLIC_NETWORKS
            )
        ]
        .copy()
    )


    print(
        "\nRedes utilizadas no perfil:"
    )


    print(
        public[
            "rede_nome"
        ]
        .value_counts()
        .to_string()
    )


    context_columns = (
        DESCRIPTOR_COLUMNS
        + CLUSTER_FEATURES
    )


    unique_context = (
        public[
            context_columns
        ]
        .drop_duplicates()
        .copy()
    )


    # --------------------------------------------------------
    # Cada município + rede deveria possuir apenas um contexto.
    # --------------------------------------------------------

    network_context_counts = (
        unique_context
        .groupby(
            [
                "id_municipio",
                "rede_nome",
            ],
            dropna=False,
        )
        .size()
    )


    inconsistent = (
        network_context_counts[
            network_context_counts
            > 1
        ]
    )


    if (
        len(
            inconsistent
        )
        > 0
    ):

        raise RuntimeError(
            "Foram encontrados contextos diferentes "
            "dentro da mesma combinação município/rede.\n"
            f"Primeiros casos:\n"
            f"{inconsistent.head(20)}"
        )


    # --------------------------------------------------------
    # Contexto territorial:
    # deve ser único por município.
    # --------------------------------------------------------

    territory = (
        public[
            [
                "id_municipio",
                "sigla_uf",
                "regiao",
                "capital_uf",
                "amazonia_legal",
            ]
        ]
        .drop_duplicates()
    )


    territory_counts = (
        territory
        .groupby(
            "id_municipio"
        )
        .size()
    )


    if (
        (
            territory_counts
            > 1
        )
        .any()
    ):

        raise RuntimeError(
            "Município com mais de um contexto territorial."
        )


    territory = (
        territory
        .drop_duplicates(
            subset=[
                "id_municipio"
            ]
        )
    )


    # --------------------------------------------------------
    # Média entre redes públicas.
    #
    # Variáveis socioeconômicas são municipais e, portanto,
    # permanecem iguais.
    #
    # Indicadores educacionais podem diferir entre redes.
    # A média é NÃO ponderada pelo número de alunos avaliados.
    # --------------------------------------------------------

    profile_features = (
        unique_context
        .groupby(
            "id_municipio",
            as_index=False,
        )[
            CLUSTER_FEATURES
        ]
        .mean()
    )


    network_counts = (
        unique_context
        .groupby(
            "id_municipio"
        )[
            "rede_nome"
        ]
        .nunique()
        .rename(
            "n_redes_publicas"
        )
        .reset_index()
    )


    profiles = (
        profile_features
        .merge(
            territory,
            how="left",
            on="id_municipio",
            validate="one_to_one",
        )
        .merge(
            network_counts,
            how="left",
            on="id_municipio",
            validate="one_to_one",
        )
    )


    # --------------------------------------------------------
    # Esperamos preservar os 5.517 municípios.
    # --------------------------------------------------------

    if (
        len(
            profiles
        )
        != 5_517
    ):

        raise RuntimeError(
            "O perfil municipal não preservou "
            "todos os municípios de 2024.\n"
            f"Esperado: 5.517\n"
            f"Obtido: {len(profiles):,}"
        )


    assert (
        profiles[
            "id_municipio"
        ]
        .nunique()
        == 5_517
    )


    print(
        "\n✓ Perfil municipal construído."
    )


    print(
        f"✓ Municípios: {len(profiles):,}"
    )


    print(
        "✓ Features de clustering: "
        f"{len(CLUSTER_FEATURES)}"
    )


    print(
        "\nQuantidade de redes públicas por município:"
    )


    print(
        profiles[
            "n_redes_publicas"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    return profiles


# ============================================================
# Pré-processamento
#
# 1. log1p em variáveis assimétricas;
# 2. imputação por mediana;
# 3. StandardScaler.
#
# O K-Means utiliza distância Euclidiana, então trabalhar
# com variáveis em escalas muito diferentes distorceria
# a formação dos clusters.
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


    # --------------------------------------------------------
    # Auditoria das features destinadas ao log.
    # --------------------------------------------------------

    for feature in (
        LOG_FEATURES
    ):

        negative = (
            X[
                feature
            ]
            .dropna()
            < 0
        )


        if negative.any():

            raise RuntimeError(
                "Valor negativo encontrado em feature "
                f"logarítmica: {feature}"
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


    assert (
        X_scaled.shape
        == (
            len(
                profiles
            ),
            len(
                CLUSTER_FEATURES
            ),
        )
    )


    if not np.isfinite(
        X_scaled
    ).all():

        raise RuntimeError(
            "Matriz padronizada contém NaN ou infinito."
        )


    transformed = pd.DataFrame(
        X_scaled,
        columns=[
            f"scaled__{feature}"
            for feature in CLUSTER_FEATURES
        ],
    )


    return (
        X_scaled,
        transformed,
        preprocessing,
    )


# ============================================================
# PCA
#
# PCA NÃO define os clusters.
#
# Serve apenas para:
#
# - entender a estrutura dos dados;
# - visualizar posteriormente os grupos em 2D.
# ============================================================


def calculate_pca(
    X_scaled: np.ndarray,
    profiles: pd.DataFrame,
) -> tuple[
    PCA,
    pd.DataFrame,
]:

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )


    coordinates = (
        pca.fit_transform(
            X_scaled
        )
    )


    pca_df = profiles[
        [
            "id_municipio",
            "sigla_uf",
            "regiao",
            "capital_uf",
            "amazonia_legal",
            "n_redes_publicas",
        ]
    ].copy()


    pca_df[
        "pca_1"
    ] = coordinates[
        :,
        0
    ]


    pca_df[
        "pca_2"
    ] = coordinates[
        :,
        1
    ]


    print(
        "\nPCA somente para visualização:"
    )


    print(
        "  Variância explicada PC1: "
        f"{pca.explained_variance_ratio_[0]:.4f}"
    )


    print(
        "  Variância explicada PC2: "
        f"{pca.explained_variance_ratio_[1]:.4f}"
    )


    print(
        "  Variância acumulada PC1+PC2: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )


    return (
        pca,
        pca_df,
    )


# ============================================================
# Avaliação de K
#
# Métricas:
#
# Inertia:
#   menor é melhor, mas sempre cai com K.
#   usamos para observar o "cotovelo".
#
# Silhouette:
#   maior é melhor.
#
# Davies-Bouldin:
#   menor é melhor.
#
# Calinski-Harabasz:
#   maior é melhor.
#
# Também registramos tamanhos dos clusters para evitar
# soluções numericamente atraentes, mas operacionalmente
# pouco úteis.
# ============================================================


def evaluate_k_values(
    X_scaled: np.ndarray,
    profiles: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    rows = []


    labels_output = profiles[
        [
            "id_municipio",
            "sigla_uf",
            "regiao",
        ]
    ].copy()


    for k in range(
        K_MIN,
        K_MAX + 1,
    ):

        print(
            "\n"
            + "=" * 72
        )


        print(
            f"K = {k}"
        )


        print(
            "=" * 72
        )


        start = (
            time.perf_counter()
        )


        model = KMeans(
            n_clusters=k,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )


        labels = (
            model.fit_predict(
                X_scaled
            )
        )


        runtime = (
            time.perf_counter()
            - start
        )


        unique_labels = np.unique(
            labels
        )


        assert (
            len(
                unique_labels
            )
            == k
        )


        silhouette = float(
            silhouette_score(
                X_scaled,
                labels,
                metric="euclidean",
            )
        )


        davies_bouldin = float(
            davies_bouldin_score(
                X_scaled,
                labels,
            )
        )


        calinski_harabasz = float(
            calinski_harabasz_score(
                X_scaled,
                labels,
            )
        )


        cluster_sizes = (
            pd.Series(
                labels
            )
            .value_counts()
            .sort_index()
        )


        min_size = int(
            cluster_sizes.min()
        )


        max_size = int(
            cluster_sizes.max()
        )


        median_size = float(
            cluster_sizes.median()
        )


        smallest_share = float(
            min_size
            / len(
                labels
            )
        )


        largest_share = float(
            max_size
            / len(
                labels
            )
        )


        rows.append(
            {
                "k": (
                    k
                ),

                "inertia": (
                    float(
                        model.inertia_
                    )
                ),

                "silhouette_score": (
                    silhouette
                ),

                "davies_bouldin_score": (
                    davies_bouldin
                ),

                "calinski_harabasz_score": (
                    calinski_harabasz
                ),

                "min_cluster_size": (
                    min_size
                ),

                "median_cluster_size": (
                    median_size
                ),

                "max_cluster_size": (
                    max_size
                ),

                "smallest_cluster_share": (
                    smallest_share
                ),

                "largest_cluster_share": (
                    largest_share
                ),

                "runtime_seconds": (
                    runtime
                ),
            }
        )


        labels_output[
            f"cluster_k{k}"
        ] = (
            labels
        )


        print(
            f"Inertia: "
            f"{model.inertia_:.2f}"
        )


        print(
            f"Silhouette: "
            f"{silhouette:.4f}"
        )


        print(
            f"Davies-Bouldin: "
            f"{davies_bouldin:.4f}"
        )


        print(
            f"Calinski-Harabasz: "
            f"{calinski_harabasz:.2f}"
        )


        print(
            "Tamanho clusters: "
            f"min={min_size:,} "
            f"| mediana={median_size:,.1f} "
            f"| max={max_size:,}"
        )


        print(
            f"Tempo: "
            f"{runtime:.2f}s"
        )


    evaluation = (
        pd.DataFrame(
            rows
        )
    )


    return (
        evaluation,
        labels_output,
    )


# ============================================================
# Gráficos de diagnóstico
# ============================================================


def create_line_plot(
    evaluation: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )


    ax.plot(
        evaluation[
            "k"
        ],
        evaluation[
            column
        ],
        marker="o",
    )


    ax.set_xticks(
        evaluation[
            "k"
        ]
    )


    ax.set_xlabel(
        "Número de clusters (K)"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
    )


    ax.grid(
        alpha=0.25,
    )


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


def create_diagnostic_plots(
    evaluation: pd.DataFrame,
) -> None:

    create_line_plot(
        evaluation=(
            evaluation
        ),

        column="inertia",

        ylabel="Inertia",

        title=(
            "K-Means — método do cotovelo"
            "\nPerfis municipais de 2024"
        ),

        output_path=(
            INERTIA_IMAGE_PATH
        ),
    )


    create_line_plot(
        evaluation=(
            evaluation
        ),

        column="silhouette_score",

        ylabel="Silhouette Score",

        title=(
            "K-Means — Silhouette Score"
            "\nMaior valor indica melhor coesão/separação"
        ),

        output_path=(
            SILHOUETTE_IMAGE_PATH
        ),
    )


    create_line_plot(
        evaluation=(
            evaluation
        ),

        column="davies_bouldin_score",

        ylabel="Davies-Bouldin Index",

        title=(
            "K-Means — Davies-Bouldin Index"
            "\nMenor valor indica melhor separação relativa"
        ),

        output_path=(
            DBI_IMAGE_PATH
        ),
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
        "CLUSTERING MUNICIPAL — "
        "AVALIAÇÃO DE K"
    )


    print(
        "=" * 72
    )


    print(
        "Clustering NÃO utiliza alfabetização, "
        "risco previsto, metas, UF ou região."
    )


    # ========================================================
    # 1. Dados
    # ========================================================

    df = load_2024()


    # ========================================================
    # 2. Perfil municipal
    # ========================================================

    profiles = (
        build_municipal_profiles(
            df
        )
    )


    del df


    # ========================================================
    # 3. Matriz padronizada
    # ========================================================

    (
        X_scaled,
        transformed,
        preprocessing,
    ) = prepare_matrix(
        profiles
    )


    # ========================================================
    # 4. PCA apenas para visualização
    # ========================================================

    (
        pca,
        pca_df,
    ) = calculate_pca(
        X_scaled=(
            X_scaled
        ),
        profiles=(
            profiles
        ),
    )


    # ========================================================
    # 5. Avaliar K
    # ========================================================

    (
        evaluation,
        labels_output,
    ) = evaluate_k_values(
        X_scaled=(
            X_scaled
        ),
        profiles=(
            profiles
        ),
    )


    # ========================================================
    # 6. Persistir dados
    # ========================================================

    profiles.to_csv(
        PROFILE_OUTPUT_PATH,
        index=False,
    )


    pca_df.to_csv(
        PCA_OUTPUT_PATH,
        index=False,
    )


    evaluation.to_csv(
        EVALUATION_OUTPUT_PATH,
        index=False,
    )


    labels_output.to_csv(
        LABELS_OUTPUT_PATH,
        index=False,
    )


    # ========================================================
    # 7. Gráficos
    # ========================================================

    create_diagnostic_plots(
        evaluation
    )


    # ========================================================
    # 8. Metadados
    # ========================================================

    best_silhouette_row = (
        evaluation
        .sort_values(
            "silhouette_score",
            ascending=False,
        )
        .iloc[0]
    )


    best_dbi_row = (
        evaluation
        .sort_values(
            "davies_bouldin_score",
            ascending=True,
        )
        .iloc[0]
    )


    metadata = {
        "analysis": (
            "kmeans_k_evaluation"
        ),

        "year": 2024,

        "n_municipalities": (
            int(
                len(
                    profiles
                )
            )
        ),

        "n_cluster_features": (
            len(
                CLUSTER_FEATURES
            )
        ),

        "cluster_features": (
            CLUSTER_FEATURES
        ),

        "excluded_from_clustering": [
            "id_municipio",
            "sigla_uf",
            "regiao",
            "capital_uf",
            "amazonia_legal",
            "alfabetizado",
            "taxa_risco_real",
            "probabilidade_media_risco",
            "metas",
        ],

        "public_networks_used": (
            PUBLIC_NETWORKS
        ),

        "network_aggregation": (
            "unweighted mean of unique "
            "municipality-network contexts"
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

        "algorithm": (
            "KMeans"
        ),

        "k_range": [
            K_MIN,
            K_MAX,
        ],

        "n_init": (
            N_INIT
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "pca_used_for_clustering": (
            False
        ),

        "pca_used_for_visualization": (
            True
        ),

        "pca_explained_variance_ratio": (
            pca
            .explained_variance_ratio_
            .tolist()
        ),

        "pca_explained_variance_2d": (
            float(
                pca
                .explained_variance_ratio_
                .sum()
            )
        ),

        "best_k_by_silhouette": (
            int(
                best_silhouette_row[
                    "k"
                ]
            )
        ),

        "best_silhouette_score": (
            float(
                best_silhouette_row[
                    "silhouette_score"
                ]
            )
        ),

        "best_k_by_davies_bouldin": (
            int(
                best_dbi_row[
                    "k"
                ]
            )
        ),

        "best_davies_bouldin_score": (
            float(
                best_dbi_row[
                    "davies_bouldin_score"
                ]
            )
        ),

        "important_note": (
            "K will not be selected automatically "
            "from a single metric. Silhouette, "
            "Davies-Bouldin, inertia, cluster sizes "
            "and interpretability will be evaluated "
            "together before freezing the final K."
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
        "AVALIAÇÃO DE K"
    )


    print(
        "=" * 72
    )


    print(
        evaluation[
            [
                "k",
                "inertia",
                "silhouette_score",
                "davies_bouldin_score",
                "calinski_harabasz_score",
                "min_cluster_size",
                "max_cluster_size",
                "smallest_cluster_share",
                "largest_cluster_share",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nMelhor K isoladamente por Silhouette:"
    )


    print(
        f"  K = "
        f"{int(best_silhouette_row['k'])}"
        f" | "
        f"{best_silhouette_row['silhouette_score']:.4f}"
    )


    print(
        "\nMelhor K isoladamente por Davies-Bouldin:"
    )


    print(
        f"  K = "
        f"{int(best_dbi_row['k'])}"
        f" | "
        f"{best_dbi_row['davies_bouldin_score']:.4f}"
    )


    print(
        "\nIMPORTANTE:"
    )


    print(
        "Ainda NÃO estamos selecionando o K final."
    )


    print(
        "A decisão será feita após analisar conjuntamente:"
    )


    print(
        "- Silhouette"
    )


    print(
        "- Davies-Bouldin"
    )


    print(
        "- Inertia / cotovelo"
    )


    print(
        "- tamanhos dos clusters"
    )


    print(
        "- utilidade interpretativa"
    )


    print(
        f"\nTempo total: "
        f"{elapsed:.2f}s"
    )


    print(
        "\nArquivos gerados:"
    )


    paths = [
        PROFILE_OUTPUT_PATH,
        PCA_OUTPUT_PATH,
        EVALUATION_OUTPUT_PATH,
        LABELS_OUTPUT_PATH,
        METADATA_OUTPUT_PATH,
        INERTIA_IMAGE_PATH,
        SILHOUETTE_IMAGE_PATH,
        DBI_IMAGE_PATH,
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
        "DIAGNÓSTICO DE K CONCLUÍDO"
    )


    print(
        "=" * 72
    )


    print(
        "Nenhum K foi congelado nesta execução."
    )


if __name__ == "__main__":
    main()