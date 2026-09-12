from __future__ import annotations

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


# ---------------------------------------------------------------------
# Features do baseline
# ---------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "rede_nome",
    "regiao",
]

BINARY_FEATURES = [
    "capital_uf",
    "amazonia_legal",
]

LOG_NUMERIC_FEATURES = [
    "populacao_2020",
    "pib_por_habitante_2020",
]

NUMERIC_FEATURES = [
    "participacao_agropecuaria_2020",
    "participacao_industria_2020",
    "participacao_administracao_publica_2020",
]

# Feature experimental.
# Será avaliada posteriormente, usando apenas dados de desenvolvimento.
EXPERIMENTAL_UF_FEATURE = "sigla_uf"


BASELINE_FEATURES = (
    CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + LOG_NUMERIC_FEATURES
    + NUMERIC_FEATURES
)


def build_preprocessor(
    include_uf: bool = False,
) -> ColumnTransformer:
    """
    Constrói o pré-processamento da classificação supervisionada.

    Parâmetros
    ----------
    include_uf:
        Se True, adiciona sigla_uf às variáveis categóricas.
        O baseline padrão não utiliza UF.

    Retorno
    -------
    ColumnTransformer
        Pré-processador compatível com Pipeline do Scikit-learn.
    """

    categorical_features = CATEGORICAL_FEATURES.copy()

    if include_uf:
        categorical_features.append(
            EXPERIMENTAL_UF_FEATURE
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    log_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES,
            ),
            (
                "log_numeric",
                log_numeric_pipeline,
                LOG_NUMERIC_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )