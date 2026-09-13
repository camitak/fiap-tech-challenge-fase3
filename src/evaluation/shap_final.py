from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from scipy import sparse

from sklearn.ensemble import HistGradientBoostingClassifier
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
# INTERPRETABILIDADE — SHAP VALUES
#
# Modelo:
# HistGradientBoostingClassifier final já congelado.
#
# Objetivos:
#
# 1. calcular SHAP Values do modelo final;
# 2. explicar as previsões na perspectiva da classe de risco;
# 3. medir importância global;
# 4. identificar direção dos padrões aprendidos;
# 5. gerar perfis interpretáveis das features.
#
# IMPORTANTE:
#
# SHAP explica o comportamento preditivo do modelo.
#
# NÃO representa causalidade.
#
# Nenhuma feature, algoritmo, hiperparâmetro ou threshold
# será alterado em função desta análise.
# ============================================================


RANDOM_STATE = 42

SHAP_SAMPLE_SIZE = 20_000

CHECK_ADDITIVITY_ROWS = 500


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
# Configuração final congelada
# ============================================================


FINAL_CONFIG_PATH = (
    REPORTS_PATH
    / "hgb_final_threshold_2023.json"
)


# ============================================================
# Saídas
# ============================================================


RAW_SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_shap_2024_raw_summary.csv"
)


TRANSFORMED_SUMMARY_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_shap_2024_transformed_summary.csv"
)


PROFILES_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_shap_2024_profiles.csv"
)


SAMPLE_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_shap_2024_sample.parquet"
)


METADATA_OUTPUT_PATH = (
    REPORTS_PATH
    / "hgb_shap_2024.json"
)


GLOBAL_IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "hgb_shap_global_importance_2024.png"
)


BEESWARM_IMAGE_OUTPUT_PATH = (
    IMAGES_PATH
    / "hgb_shap_beeswarm_2024.png"
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
# Features que serão tratadas como categorias nos perfis SHAP
# ============================================================


CATEGORICAL_PROFILE_FEATURES = {
    "rede_nome",
    "regiao",
    "sigla_uf",
    "capital_uf",
    "amazonia_legal",
}


# ============================================================
# Configuração final
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


    assert (
        config[
            "development_year"
        ]
        == 2023
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


def load_dataset(
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
            "Nenhum arquivo Parquet "
            f"encontrado em {path}"
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
# Pipeline final congelado
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
# Amostra estratificada
#
# Preservamos:
# - alfabetizado
# - UF
#
# Isso mantém a composição territorial do teste e também
# a proporção da classe.
# ============================================================


def build_shap_sample(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if (
        SHAP_SAMPLE_SIZE
        >= len(df)
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
            train_size=SHAP_SAMPLE_SIZE,
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
        == SHAP_SAMPLE_SIZE
    )


    return sample


# ============================================================
# Mapeamento:
#
# 45 features transformadas
#        ↓
# 18 features originais
#
# Exemplo:
#
# categorical__sigla_uf_AL
# categorical__sigla_uf_BA
# categorical__sigla_uf_CE
#
#          ↓
#
# sigla_uf
# ============================================================


def build_transformed_feature_mapping(
    transformed_names: np.ndarray,
) -> pd.DataFrame:

    raw_features_by_length = sorted(
        FEATURES,
        key=len,
        reverse=True,
    )


    rows = []


    for transformed_index, transformed_name in enumerate(
        transformed_names
    ):

        transformed_name = str(
            transformed_name
        )


        # ----------------------------------------------------
        # Remove prefixo do ColumnTransformer:
        #
        # categorical__sigla_uf_AL
        #              ↓
        # sigla_uf_AL
        # ----------------------------------------------------

        if "__" in transformed_name:

            local_name = (
                transformed_name
                .split(
                    "__",
                    1,
                )[1]
            )

        else:

            local_name = (
                transformed_name
            )


        raw_feature = None


        for candidate in (
            raw_features_by_length
        ):

            if (
                local_name
                == candidate
            ):

                raw_feature = (
                    candidate
                )

                break


            if (
                local_name
                .startswith(
                    candidate
                    + "_"
                )
            ):

                raw_feature = (
                    candidate
                )

                break


        if raw_feature is None:

            raise RuntimeError(
                "Não foi possível mapear "
                "feature transformada para "
                "feature original: "
                f"{transformed_name}"
            )


        rows.append(
            {
                "transformed_index": (
                    transformed_index
                ),

                "transformed_feature": (
                    transformed_name
                ),

                "raw_feature": (
                    raw_feature
                ),
            }
        )


    mapping = pd.DataFrame(
        rows
    )


    assert (
        len(mapping)
        == len(
            transformed_names
        )
    )


    assert (
        mapping[
            "raw_feature"
        ]
        .isna()
        .sum()
        == 0
    )


    return mapping


# ============================================================
# Extrair SHAP da classe alfabetizado=1
#
# Compatibilidade com diferentes versões da biblioteca shap.
# ============================================================


def extract_class_one_shap(
    explainer,
    X_transformed: np.ndarray,
    model: HistGradientBoostingClassifier,
) -> tuple[
    np.ndarray,
    float,
]:

    class_positions = np.where(
        model.classes_
        == 1
    )[0]


    if (
        len(
            class_positions
        )
        != 1
    ):

        raise RuntimeError(
            "Classe alfabetizado=1 "
            "não localizada no modelo."
        )


    class_one_index = int(
        class_positions[0]
    )


    raw_shap = (
        explainer.shap_values(
            X_transformed,
            check_additivity=False,
        )
    )


    expected_value = (
        explainer.expected_value
    )


    # --------------------------------------------------------
    # Formato legado:
    #
    # list[
    #     array classe 0,
    #     array classe 1
    # ]
    # --------------------------------------------------------

    if isinstance(
        raw_shap,
        list,
    ):

        if (
            len(
                raw_shap
            )
            <= class_one_index
        ):

            raise RuntimeError(
                "Formato SHAP inesperado: "
                "lista sem a classe 1."
            )


        values = np.asarray(
            raw_shap[
                class_one_index
            ]
        )


        expected_array = np.asarray(
            expected_value
        ).reshape(-1)


        if (
            len(
                expected_array
            )
            > class_one_index
        ):

            expected = float(
                expected_array[
                    class_one_index
                ]
            )

        else:

            expected = float(
                expected_array[0]
            )


        return (
            values,
            expected,
        )


    # --------------------------------------------------------
    # Formato ndarray
    # --------------------------------------------------------

    raw_array = np.asarray(
        raw_shap
    )


    # --------------------------------------------------------
    # n x features
    # --------------------------------------------------------

    if (
        raw_array.ndim
        == 2
    ):

        values = (
            raw_array
        )


        expected_array = np.asarray(
            expected_value
        ).reshape(-1)


        if (
            len(
                expected_array
            )
            == 1
        ):

            expected = float(
                expected_array[0]
            )

        elif (
            len(
                expected_array
            )
            > class_one_index
        ):

            expected = float(
                expected_array[
                    class_one_index
                ]
            )

        else:

            raise RuntimeError(
                "expected_value SHAP "
                "com formato inesperado."
            )


        return (
            values,
            expected,
        )


    # --------------------------------------------------------
    # n x features x classes
    # --------------------------------------------------------

    if (
        raw_array.ndim
        == 3
    ):

        values = (
            raw_array[
                :,
                :,
                class_one_index,
            ]
        )


        expected_array = np.asarray(
            expected_value
        ).reshape(-1)


        expected = float(
            expected_array[
                class_one_index
            ]
        )


        return (
            values,
            expected,
        )


    raise RuntimeError(
        "Formato SHAP não suportado. "
        f"Shape recebido: "
        f"{raw_array.shape}"
    )


# ============================================================
# Calcular SHAP
#
# O TreeExplainer explica o score bruto da classe alfabetizado.
#
# Para risco:
#
# score_risco = -score_alfabetizado
#
# Portanto:
#
# shap_risco = -shap_alfabetizado
#
# SHAP risco > 0:
#     empurra previsão em direção ao risco.
#
# SHAP risco < 0:
#     empurra previsão em direção à alfabetização.
# ============================================================


def calculate_shap_values(
    model: HistGradientBoostingClassifier,
    X_transformed: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
    float,
]:

    print(
        "\nCriando TreeExplainer..."
    )


    try:

        explainer = (
            shap.TreeExplainer(
                model,
                feature_perturbation=(
                    "tree_path_dependent"
                ),
                model_output="raw",
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Não foi possível criar "
            "shap.TreeExplainer para o "
            "HistGradientBoostingClassifier. "
            f"Versão shap: {shap.__version__}. "
            "Não utilizar KernelExplainer como "
            "fallback sem revisar a metodologia."
        ) from exc


    print(
        "✓ TreeExplainer criado."
    )


    start_time = (
        time.perf_counter()
    )


    (
        shap_literate,
        expected_literate,
    ) = extract_class_one_shap(
        explainer=explainer,
        X_transformed=X_transformed,
        model=model,
    )


    elapsed = (
        time.perf_counter()
        - start_time
    )


    assert (
        shap_literate.shape
        == X_transformed.shape
    )


    # --------------------------------------------------------
    # Auditoria de aditividade no score bruto.
    # --------------------------------------------------------

    n_check = min(
        CHECK_ADDITIVITY_ROWS,
        len(
            X_transformed
        ),
    )


    decision = np.asarray(
        model.decision_function(
            X_transformed[
                :n_check
            ]
        )
    ).reshape(-1)


    reconstructed = (
        expected_literate
        + shap_literate[
            :n_check
        ].sum(
            axis=1
        )
    )


    max_additivity_error = float(
        np.max(
            np.abs(
                decision
                - reconstructed
            )
        )
    )


    if (
        max_additivity_error
        > 1e-3
    ):

        raise RuntimeError(
            "Falha na auditoria de aditividade "
            "dos SHAP values. "
            "Erro máximo: "
            f"{max_additivity_error:.8f}"
        )


    # --------------------------------------------------------
    # Perspectiva da classe de risco.
    # --------------------------------------------------------

    shap_risk = (
        -shap_literate
    )


    expected_risk = (
        -expected_literate
    )


    print(
        f"✓ SHAP calculado em "
        f"{elapsed:.2f}s"
    )


    print(
        "✓ Erro máximo de aditividade: "
        f"{max_additivity_error:.10f}"
    )


    return (
        shap_risk,
        float(
            expected_risk
        ),
        max_additivity_error,
    )


# ============================================================
# Agregar SHAP transformado -> feature original
#
# Para cada observação:
#
# shap(feature original)
# =
# soma dos SHAPs das colunas transformadas daquela feature.
#
# Exemplo:
#
# UF_AL + UF_BA + UF_CE + ...
#               ↓
#            sigla_uf
# ============================================================


def aggregate_raw_shap(
    shap_risk_transformed: np.ndarray,
    mapping: pd.DataFrame,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:

    raw_matrix = np.zeros(
        (
            shap_risk_transformed.shape[0],
            len(
                FEATURES
            ),
        ),
        dtype=np.float64,
    )


    rows = []


    for raw_index, raw_feature in enumerate(
        FEATURES
    ):

        transformed_indices = (
            mapping.loc[
                mapping[
                    "raw_feature"
                ]
                == raw_feature,
                "transformed_index",
            ]
            .astype(int)
            .to_numpy()
        )


        if (
            len(
                transformed_indices
            )
            == 0
        ):

            raise RuntimeError(
                "Feature original sem coluna "
                "transformada: "
                f"{raw_feature}"
            )


        raw_values = (
            shap_risk_transformed[
                :,
                transformed_indices,
            ]
            .sum(
                axis=1
            )
        )


        raw_matrix[
            :,
            raw_index
        ] = (
            raw_values
        )


        rows.append(
            {
                "feature": (
                    raw_feature
                ),

                "transformed_components": (
                    len(
                        transformed_indices
                    )
                ),

                "mean_abs_shap_risk": (
                    float(
                        np.mean(
                            np.abs(
                                raw_values
                            )
                        )
                    )
                ),

                "mean_shap_risk": (
                    float(
                        np.mean(
                            raw_values
                        )
                    )
                ),

                "median_shap_risk": (
                    float(
                        np.median(
                            raw_values
                        )
                    )
                ),

                "std_shap_risk": (
                    float(
                        np.std(
                            raw_values,
                            ddof=1,
                        )
                    )
                ),

                "p05_shap_risk": (
                    float(
                        np.quantile(
                            raw_values,
                            0.05,
                        )
                    )
                ),

                "p95_shap_risk": (
                    float(
                        np.quantile(
                            raw_values,
                            0.95,
                        )
                    )
                ),

                "positive_share": (
                    float(
                        np.mean(
                            raw_values
                            > 0
                        )
                    )
                ),

                "negative_share": (
                    float(
                        np.mean(
                            raw_values
                            < 0
                        )
                    )
                ),
            }
        )


    summary = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "mean_abs_shap_risk",
            ascending=False,
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


    return (
        raw_matrix,
        summary,
    )


# ============================================================
# Resumo das 45 features transformadas
# ============================================================


def build_transformed_summary(
    shap_risk: np.ndarray,
    mapping: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    for column_index in range(
        shap_risk.shape[1]
    ):

        values = (
            shap_risk[
                :,
                column_index
            ]
        )


        mapping_row = (
            mapping.loc[
                mapping[
                    "transformed_index"
                ]
                == column_index
            ]
            .iloc[0]
        )


        rows.append(
            {
                "transformed_feature": (
                    mapping_row[
                        "transformed_feature"
                    ]
                ),

                "raw_feature": (
                    mapping_row[
                        "raw_feature"
                    ]
                ),

                "mean_abs_shap_risk": (
                    float(
                        np.mean(
                            np.abs(
                                values
                            )
                        )
                    )
                ),

                "mean_shap_risk": (
                    float(
                        np.mean(
                            values
                        )
                    )
                ),

                "std_shap_risk": (
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                ),

                "positive_share": (
                    float(
                        np.mean(
                            values
                            > 0
                        )
                    )
                ),

                "negative_share": (
                    float(
                        np.mean(
                            values
                            < 0
                        )
                    )
                ),
            }
        )


    summary = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "mean_abs_shap_risk",
            ascending=False,
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
# Probabilidade de risco da amostra
# ============================================================


def calculate_sample_risk_probability(
    model: HistGradientBoostingClassifier,
    X_transformed: np.ndarray,
) -> np.ndarray:

    probabilities = (
        model.predict_proba(
            X_transformed
        )
    )


    class_positions = np.where(
        model.classes_
        == 1
    )[0]


    if (
        len(
            class_positions
        )
        != 1
    ):

        raise RuntimeError(
            "Classe alfabetizado=1 "
            "não localizada."
        )


    p_literate = (
        probabilities[
            :,
            int(
                class_positions[0]
            ),
        ]
    )


    return (
        1.0
        - p_literate
    )


# ============================================================
# Perfis direcionais
#
# Numéricas:
# quintis da variável.
#
# Categóricas:
# categoria.
#
# O objetivo é permitir interpretar:
#
# - quais valores tendem a empurrar o score para risco;
# - quais tendem a empurrar para alfabetização.
#
# Ainda assim:
#
# direção SHAP != causalidade.
# ============================================================


def build_feature_profiles(
    sample: pd.DataFrame,
    raw_shap_matrix: np.ndarray,
    p_risk: np.ndarray,
) -> pd.DataFrame:

    risk_real = (
        1
        - sample[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy()
    )


    rows = []


    for feature_index, feature in enumerate(
        FEATURES
    ):

        shap_values = (
            raw_shap_matrix[
                :,
                feature_index
            ]
        )


        feature_values = (
            sample[
                feature
            ]
        )


        # ====================================================
        # Categórica
        # ====================================================

        if (
            feature
            in CATEGORICAL_PROFILE_FEATURES
        ):

            categories = (
                feature_values
                .fillna(
                    "__MISSING__"
                )
                .astype(str)
            )


            temporary = pd.DataFrame(
                {
                    "group": (
                        categories
                    ),

                    "shap_risk": (
                        shap_values
                    ),

                    "risk_real": (
                        risk_real
                    ),

                    "p_risk": (
                        p_risk
                    ),
                }
            )


            grouped = (
                temporary
                .groupby(
                    "group",
                    dropna=False,
                )
            )


            for group_name, group in grouped:

                rows.append(
                    {
                        "feature": (
                            feature
                        ),

                        "profile_type": (
                            "category"
                        ),

                        "level_or_bin": (
                            str(
                                group_name
                            )
                        ),

                        "n": (
                            int(
                                len(
                                    group
                                )
                            )
                        ),

                        "value_min": (
                            np.nan
                        ),

                        "value_mean": (
                            np.nan
                        ),

                        "value_max": (
                            np.nan
                        ),

                        "mean_shap_risk": (
                            float(
                                group[
                                    "shap_risk"
                                ]
                                .mean()
                            )
                        ),

                        "median_shap_risk": (
                            float(
                                group[
                                    "shap_risk"
                                ]
                                .median()
                            )
                        ),

                        "mean_abs_shap_risk": (
                            float(
                                group[
                                    "shap_risk"
                                ]
                                .abs()
                                .mean()
                            )
                        ),

                        "actual_risk_rate": (
                            float(
                                group[
                                    "risk_real"
                                ]
                                .mean()
                            )
                        ),

                        "predicted_risk_mean": (
                            float(
                                group[
                                    "p_risk"
                                ]
                                .mean()
                            )
                        ),
                    }
                )


            continue


        # ====================================================
        # Numérica
        # ====================================================

        numeric_values = pd.to_numeric(
            feature_values,
            errors="coerce",
        )


        valid_mask = (
            numeric_values
            .notna()
            .to_numpy()
        )


        if (
            valid_mask.sum()
            > 0
        ):

            valid_values = (
                numeric_values.loc[
                    valid_mask
                ]
            )


            try:

                bins = pd.qcut(
                    valid_values,
                    q=5,
                    duplicates="drop",
                )

            except ValueError:

                bins = pd.Series(
                    "all",
                    index=(
                        valid_values
                        .index
                    ),
                )


            temporary = pd.DataFrame(
                {
                    "value": (
                        valid_values
                    ),

                    "group": (
                        bins
                        .astype(str)
                    ),

                    "shap_risk": (
                        shap_values[
                            valid_mask
                        ]
                    ),

                    "risk_real": (
                        risk_real[
                            valid_mask
                        ]
                    ),

                    "p_risk": (
                        p_risk[
                            valid_mask
                        ]
                    ),
                }
            )


            grouped = (
                temporary
                .groupby(
                    "group",
                    dropna=False,
                )
            )


            for group_name, group in grouped:

                rows.append(
                    {
                        "feature": (
                            feature
                        ),

                        "profile_type": (
                            "quantile"
                        ),

                        "level_or_bin": (
                            str(
                                group_name
                            )
                        ),

                        "n": (
                            int(
                                len(
                                    group
                                )
                            )
                        ),

                        "value_min": (
                            float(
                                group[
                                    "value"
                                ]
                                .min()
                            )
                        ),

                        "value_mean": (
                            float(
                                group[
                                    "value"
                                ]
                                .mean()
                            )
                        ),

                        "value_max": (
                            float(
                                group[
                                    "value"
                                ]
                                .max()
                            )
                        ),

                        "mean_shap_risk": (
                            float(
                                group[
                                    "shap_risk"
                                ]
                                .mean()
                            )
                        ),

                        "median_shap_risk": (
                            float(
                                group[
                                    "shap_risk"
                                ]
                                .median()
                            )
                        ),

                        "mean_abs_shap_risk": (
                            float(
                                group[
                                    "shap_risk"
                                ]
                                .abs()
                                .mean()
                            )
                        ),

                        "actual_risk_rate": (
                            float(
                                group[
                                    "risk_real"
                                ]
                                .mean()
                            )
                        ),

                        "predicted_risk_mean": (
                            float(
                                group[
                                    "p_risk"
                                ]
                                .mean()
                            )
                        ),
                    }
                )


        # ----------------------------------------------------
        # Missing como grupo separado
        # ----------------------------------------------------

        missing_mask = (
            ~valid_mask
        )


        if (
            missing_mask.sum()
            > 0
        ):

            rows.append(
                {
                    "feature": (
                        feature
                    ),

                    "profile_type": (
                        "missing"
                    ),

                    "level_or_bin": (
                        "__MISSING__"
                    ),

                    "n": (
                        int(
                            missing_mask.sum()
                        )
                    ),

                    "value_min": (
                        np.nan
                    ),

                    "value_mean": (
                        np.nan
                    ),

                    "value_max": (
                        np.nan
                    ),

                    "mean_shap_risk": (
                        float(
                            np.mean(
                                shap_values[
                                    missing_mask
                                ]
                            )
                        )
                    ),

                    "median_shap_risk": (
                        float(
                            np.median(
                                shap_values[
                                    missing_mask
                                ]
                            )
                        )
                    ),

                    "mean_abs_shap_risk": (
                        float(
                            np.mean(
                                np.abs(
                                    shap_values[
                                        missing_mask
                                    ]
                                )
                            )
                        )
                    ),

                    "actual_risk_rate": (
                        float(
                            np.mean(
                                risk_real[
                                    missing_mask
                                ]
                            )
                        )
                    ),

                    "predicted_risk_mean": (
                        float(
                            np.mean(
                                p_risk[
                                    missing_mask
                                ]
                            )
                        )
                    ),
                }
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Gráfico global — 18 features originais
# ============================================================


def create_global_importance_plot(
    raw_summary: pd.DataFrame,
) -> None:

    plot_data = (
        raw_summary
        .sort_values(
            "mean_abs_shap_risk",
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
            "mean_abs_shap_risk"
        ],
    )


    ax.set_title(
        "SHAP — importância global do HGB final"
        "\nPerspectiva da classe de risco — amostra 2024"
    )


    ax.set_xlabel(
        "Média do |SHAP de risco| "
        "(score bruto / log-odds)"
    )


    ax.set_ylabel(
        "Feature original"
    )


    fig.tight_layout()


    fig.savefig(
        GLOBAL_IMAGE_OUTPUT_PATH,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# Beeswarm
#
# Mostra as 45 features transformadas.
#
# Eixo horizontal:
#
# SHAP positivo -> maior risco
# SHAP negativo -> maior alfabetização
#
# A cor é o valor da feature transformada.
# ============================================================


def create_beeswarm_plot(
    shap_risk: np.ndarray,
    X_transformed: np.ndarray,
    transformed_names: np.ndarray,
) -> None:

    plt.figure()


    shap.summary_plot(
        shap_values=shap_risk,
        features=X_transformed,
        feature_names=(
            transformed_names
        ),
        max_display=20,
        show=False,
        plot_size=(
            11,
            8,
        ),
    )


    ax = plt.gca()


    ax.set_xlabel(
        "SHAP de risco "
        "(positivo = maior risco; "
        "negativo = maior alfabetização)"
    )


    plt.tight_layout()


    plt.savefig(
        BEESWARM_IMAGE_OUTPUT_PATH,
        dpi=160,
        bbox_inches="tight",
    )


    plt.close(
        "all"
    )


# ============================================================
# Amostra auditável
#
# Salva:
#
# - features originais;
# - probabilidade prevista de risco;
# - SHAP agregado para cada uma das 18 features.
#
# Isso permite análises locais posteriores sem precisar
# recalcular os SHAP values.
# ============================================================


def build_sample_output(
    sample: pd.DataFrame,
    p_risk: np.ndarray,
    raw_shap_matrix: np.ndarray,
) -> pd.DataFrame:

    output = (
        sample[
            CONTROL_COLUMNS
            + FEATURES
        ]
        .copy()
    )


    output[
        "p_risco"
    ] = (
        p_risk
    )


    for feature_index, feature in enumerate(
        FEATURES
    ):

        output[
            "shap_risk__"
            + feature
        ] = (
            raw_shap_matrix[
                :,
                feature_index
            ]
        )


    return output


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
        "SHAP VALUES — MODELO FINAL"
    )

    print(
        "=" * 72
    )

    print(
        "Modelo congelado."
    )

    print(
        "Nenhum retuning será realizado."
    )

    print(
        f"Versão shap: "
        f"{shap.__version__}"
    )


    # ========================================================
    # 1. Configuração
    # ========================================================

    config = (
        load_final_config()
    )


    print(
        "\nConfiguração final:"
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
        "  Threshold risco: "
        f"{config['selected_threshold_risk']:.2f}"
    )

    print(
        f"  Features brutas: "
        f"{len(FEATURES)}"
    )


    # ========================================================
    # 2. Treino final em 2023
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TREINANDO PIPELINE FINAL EM TODO 2023"
    )

    print(
        "=" * 72
    )


    train_df = (
        load_dataset(
            path=TRAIN_PATH,
            expected_rows=1_502_809,
        )
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
        train_df[
            "alfabetizado"
        ]
        .astype(int)
        .to_numpy(),
    )


    fit_seconds = (
        time.perf_counter()
        - fit_start
    )


    preprocessor = (
        pipeline
        .named_steps[
            "preprocessor"
        ]
    )


    model = (
        pipeline
        .named_steps[
            "model"
        ]
    )


    transformed_names = np.asarray(
        preprocessor
        .get_feature_names_out(),
        dtype=str,
    )


    assert (
        len(
            transformed_names
        )
        == 45
    )


    print(
        f"✓ Treino: "
        f"{fit_seconds:.2f}s"
    )

    print(
        "✓ Features transformadas: "
        f"{len(transformed_names)}"
    )


    mapping = (
        build_transformed_feature_mapping(
            transformed_names
        )
    )


    assert (
        mapping[
            "raw_feature"
        ]
        .nunique()
        == len(
            FEATURES
        )
    )


    del train_df

    gc.collect()


    # ========================================================
    # 3. Carregar 2024 e criar amostra
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CRIANDO AMOSTRA SHAP — 2024"
    )

    print(
        "=" * 72
    )


    test_df = (
        load_dataset(
            path=TEST_PATH,
            expected_rows=1_851_852,
        )
    )


    full_risk_rate = float(
        1.0
        - test_df[
            "alfabetizado"
        ]
        .mean()
    )


    sample = (
        build_shap_sample(
            test_df
        )
    )


    sample_risk_rate = float(
        1.0
        - sample[
            "alfabetizado"
        ]
        .mean()
    )


    print(
        f"✓ Base completa: "
        f"{len(test_df):,}"
    )

    print(
        f"✓ Amostra SHAP: "
        f"{len(sample):,}"
    )

    print(
        f"✓ UFs na amostra: "
        f"{sample['sigla_uf'].nunique()}"
    )

    print(
        f"✓ Taxa risco completa: "
        f"{full_risk_rate:.4f}"
    )

    print(
        f"✓ Taxa risco amostra: "
        f"{sample_risk_rate:.4f}"
    )


    del test_df

    gc.collect()


    # ========================================================
    # 4. Preprocessing da amostra
    # ========================================================

    print(
        "\nTransformando amostra..."
    )


    X_transformed = (
        preprocessor
        .transform(
            sample[
                FEATURES
            ]
        )
    )


    if sparse.issparse(
        X_transformed
    ):

        X_transformed = (
            X_transformed
            .toarray()
        )


    X_transformed = np.asarray(
        X_transformed
    )


    assert (
        X_transformed.shape
        == (
            len(sample),
            45,
        )
    )


    print(
        "✓ Shape transformado: "
        f"{X_transformed.shape}"
    )


    # ========================================================
    # 5. SHAP
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CALCULANDO SHAP VALUES"
    )

    print(
        "=" * 72
    )


    (
        shap_risk,
        expected_risk,
        max_additivity_error,
    ) = calculate_shap_values(
        model=model,
        X_transformed=(
            X_transformed
        ),
    )


    # ========================================================
    # 6. Probabilidade de risco
    # ========================================================

    p_risk = (
        calculate_sample_risk_probability(
            model=model,
            X_transformed=(
                X_transformed
            ),
        )
    )


    # ========================================================
    # 7. Resumo das 18 features originais
    # ========================================================

    (
        raw_shap_matrix,
        raw_summary,
    ) = aggregate_raw_shap(
        shap_risk_transformed=(
            shap_risk
        ),
        mapping=mapping,
    )


    # ========================================================
    # 8. Resumo das 45 features transformadas
    # ========================================================

    transformed_summary = (
        build_transformed_summary(
            shap_risk=shap_risk,
            mapping=mapping,
        )
    )


    # ========================================================
    # 9. Perfis direcionais
    # ========================================================

    profiles = (
        build_feature_profiles(
            sample=sample,
            raw_shap_matrix=(
                raw_shap_matrix
            ),
            p_risk=p_risk,
        )
    )


    # ========================================================
    # 10. Amostra auditável
    # ========================================================

    sample_output = (
        build_sample_output(
            sample=sample,
            p_risk=p_risk,
            raw_shap_matrix=(
                raw_shap_matrix
            ),
        )
    )


    # ========================================================
    # 11. Gráficos
    # ========================================================

    print(
        "\nGerando gráficos..."
    )


    create_global_importance_plot(
        raw_summary
    )


    create_beeswarm_plot(
        shap_risk=shap_risk,
        X_transformed=(
            X_transformed
        ),
        transformed_names=(
            transformed_names
        ),
    )


    print(
        "✓ Gráficos gerados."
    )


    # ========================================================
    # 12. Persistência
    # ========================================================

    raw_summary.to_csv(
        RAW_SUMMARY_OUTPUT_PATH,
        index=False,
    )


    transformed_summary.to_csv(
        TRANSFORMED_SUMMARY_OUTPUT_PATH,
        index=False,
    )


    profiles.to_csv(
        PROFILES_OUTPUT_PATH,
        index=False,
    )


    sample_output.to_parquet(
        SAMPLE_OUTPUT_PATH,
        index=False,
    )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    metadata = {
        "analysis": (
            "shap_values"
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

        "features_changed": False,

        "hyperparameters_changed": False,

        "threshold_changed": False,

        "threshold_risk": (
            float(
                config[
                    "selected_threshold_risk"
                ]
            )
        ),

        "shap_version": (
            shap.__version__
        ),

        "explainer": (
            "TreeExplainer"
        ),

        "model_output": (
            "raw"
        ),

        "explained_class": (
            "risk = not literate"
        ),

        "risk_shap_definition": (
            "negative of SHAP values "
            "for class alfabetizado=1"
        ),

        "positive_shap_interpretation": (
            "pushes prediction toward risk"
        ),

        "negative_shap_interpretation": (
            "pushes prediction toward literacy"
        ),

        "shap_sample_size": (
            int(
                len(sample)
            )
        ),

        "sample_strategy": (
            "StratifiedShuffleSplit by "
            "alfabetizado + sigla_uf"
        ),

        "sample_actual_risk_rate": (
            sample_risk_rate
        ),

        "full_2024_actual_risk_rate": (
            full_risk_rate
        ),

        "raw_features": (
            len(
                FEATURES
            )
        ),

        "transformed_features": (
            len(
                transformed_names
            )
        ),

        "expected_value_risk_raw": (
            float(
                expected_risk
            )
        ),

        "max_additivity_error": (
            float(
                max_additivity_error
            )
        ),

        "top_raw_features": (
            raw_summary[
                [
                    "rank",
                    "feature",
                    "transformed_components",
                    "mean_abs_shap_risk",
                    "mean_shap_risk",
                    "positive_share",
                    "negative_share",
                ]
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        ),

        "interpretation_note": (
            "SHAP explains predictive behavior, "
            "not causal effects. Positive SHAP "
            "values in this analysis push the "
            "prediction toward risk; negative "
            "values push toward literacy."
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
    # 13. Resultado
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP FEATURES — SHAP"
    )

    print(
        "=" * 72
    )


    print(
        raw_summary[
            [
                "rank",
                "feature",
                "transformed_components",
                "mean_abs_shap_risk",
                "mean_shap_risk",
                "positive_share",
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
        "COMO INTERPRETAR"
    )

    print(
        "=" * 72
    )


    print(
        "SHAP risco > 0:"
    )

    print(
        "  empurra a previsão para "
        "maior risco de não alfabetização."
    )


    print(
        "\nSHAP risco < 0:"
    )

    print(
        "  empurra a previsão para "
        "maior probabilidade de alfabetização."
    )


    print(
        "\nmean_abs_shap_risk:"
    )

    print(
        "  magnitude média da influência "
        "da feature sobre o score do modelo."
    )


    print(
        "\nIMPORTANTE:"
    )

    print(
        "  direção SHAP não significa "
        "efeito causal."
    )


    print(
        f"\nTempo total: "
        f"{total_seconds / 60:.2f} min"
    )


    print(
        "\nArquivos gerados:"
    )


    output_paths = [
        RAW_SUMMARY_OUTPUT_PATH,
        TRANSFORMED_SUMMARY_OUTPUT_PATH,
        PROFILES_OUTPUT_PATH,
        SAMPLE_OUTPUT_PATH,
        METADATA_OUTPUT_PATH,
        GLOBAL_IMAGE_OUTPUT_PATH,
        BEESWARM_IMAGE_OUTPUT_PATH,
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
        "SHAP CONCLUÍDO"
    )

    print(
        "=" * 72
    )

    print(
        "Nenhuma decisão de modelagem "
        "foi alterada."
    )


    del pipeline
    del preprocessor
    del model
    del sample
    del X_transformed
    del shap_risk
    del raw_shap_matrix
    del sample_output

    gc.collect()


if __name__ == "__main__":
    main()