from __future__ import annotations

from typing import Literal

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# Tech Challenge Fase 3
# Preprocessing supervisionado
#
# Objetivos:
# - preservar compatibilidade com o baseline v1;
# - suportar o conjunto enriquecido v2;
# - manter toda imputação e transformação dentro da pipeline;
# - permitir experimento separado com sigla_uf;
# - evitar data leakage.
#
# IMPORTANTE:
# O preprocessor deve ser ajustado somente nos dados de treino.
# ============================================================


FeatureSet = Literal[
    "baseline",
    "enriched",
]


# ============================================================
# FEATURES ORIGINAIS — V1
#
# Estas constantes são preservadas com os mesmos nomes para
# manter compatibilidade com scripts já existentes.
# ============================================================

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


EXPERIMENTAL_UF_FEATURE = "sigla_uf"


BASELINE_FEATURES = (
    CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + LOG_NUMERIC_FEATURES
    + NUMERIC_FEATURES
)


# ============================================================
# FEATURES EDUCACIONAIS — V2
#
# Seleção definida após EDA exclusivamente com 2023.
#
# Mantidas:
# - 1 variável representando escala;
# - infraestrutura;
# - ruralidade;
# - indicadores de organização da oferta;
# - ensino integral.
#
# Excluídas do conjunto principal por redundância:
# - total_matriculas_anos_iniciais;
# - total_docentes_anos_iniciais;
# - total_turmas_anos_iniciais.
#
# Essas colunas continuam existindo na base v2 e podem ser
# utilizadas em experimentos futuros.
# ============================================================


EDUCATIONAL_LOG_NUMERIC_FEATURES = [
    "quantidade_escolas_anos_iniciais",
    "alunos_por_turma_anos_iniciais",
]


EDUCATIONAL_NUMERIC_FEATURES = [
    "proporcao_escolas_rurais",
    "proporcao_internet_aprendizagem",
    "proporcao_biblioteca_sala_leitura",
    "proporcao_laboratorio_informatica",
    "razao_matriculas_docentes_anos_iniciais",
    "proporcao_matriculas_integral_anos_iniciais",
]


EDUCATIONAL_SELECTED_FEATURES = (
    EDUCATIONAL_LOG_NUMERIC_FEATURES
    + EDUCATIONAL_NUMERIC_FEATURES
)


ENRICHED_FEATURES = (
    BASELINE_FEATURES
    + EDUCATIONAL_SELECTED_FEATURES
)


# ============================================================
# FEATURES EDUCACIONAIS DISPONÍVEIS, MAS NÃO SELECIONADAS
#
# Mantidas explicitamente para auditoria e documentação.
# ============================================================

EDUCATIONAL_REDUNDANT_FEATURES = [
    "total_matriculas_anos_iniciais",
    "total_docentes_anos_iniciais",
    "total_turmas_anos_iniciais",
]


# ============================================================
# COLUNAS PROIBIDAS COMO PREDITOR
#
# Essas colunas podem existir no dataframe para controle,
# avaliação ou rastreabilidade, mas jamais devem ser
# selecionadas automaticamente pela pipeline.
# ============================================================

FORBIDDEN_MODEL_FEATURES = [
    "ano",
    "aluno_key",
    "id_municipio",
    "alfabetizado",
]


# ============================================================
# Configurações válidas
# ============================================================

VALID_FEATURE_SETS = {
    "baseline",
    "enriched",
}


# ============================================================
# Helpers
# ============================================================


def _validate_feature_set(
    feature_set: str,
) -> None:
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(
            "feature_set inválido. "
            f"Recebido={feature_set!r}. "
            f"Valores permitidos={sorted(VALID_FEATURE_SETS)}"
        )


def get_model_features(
    feature_set: FeatureSet = "baseline",
    include_uf: bool = False,
) -> list[str]:
    """
    Retorna a lista de features brutas esperadas pelo modelo.

    Parâmetros
    ----------
    feature_set:
        "baseline":
            conjunto original de 9 features.

        "enriched":
            9 features baseline +
            8 features educacionais selecionadas.

    include_uf:
        Quando True, adiciona sigla_uf ao conjunto categórico.

    Retorno
    -------
    list[str]
        Lista ordenada de features.
    """

    _validate_feature_set(
        feature_set
    )

    if feature_set == "baseline":
        features = list(
            BASELINE_FEATURES
        )

    else:
        features = list(
            ENRICHED_FEATURES
        )

    if include_uf:
        features.append(
            EXPERIMENTAL_UF_FEATURE
        )

    return features


def get_feature_groups(
    feature_set: FeatureSet = "baseline",
    include_uf: bool = False,
) -> dict[str, list[str]]:
    """
    Retorna os grupos de features utilizados pelo ColumnTransformer.
    """

    _validate_feature_set(
        feature_set
    )

    categorical = list(
        CATEGORICAL_FEATURES
    )

    if include_uf:
        categorical.append(
            EXPERIMENTAL_UF_FEATURE
        )

    binary = list(
        BINARY_FEATURES
    )

    log_numeric = list(
        LOG_NUMERIC_FEATURES
    )

    numeric = list(
        NUMERIC_FEATURES
    )

    if feature_set == "enriched":

        log_numeric.extend(
            EDUCATIONAL_LOG_NUMERIC_FEATURES
        )

        numeric.extend(
            EDUCATIONAL_NUMERIC_FEATURES
        )

    return {
        "categorical": categorical,
        "binary": binary,
        "log_numeric": log_numeric,
        "numeric": numeric,
    }


# ============================================================
# Pipelines internas
# ============================================================


def _build_categorical_pipeline() -> Pipeline:
    """
    Pipeline para variáveis categóricas nominais.

    - imputação pela categoria mais frequente;
    - One-Hot Encoding;
    - categorias desconhecidas são ignoradas.

    handle_unknown="ignore" é especialmente importante para
    avaliação temporal com categorias que possam surgir em
    períodos futuros.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


def _build_binary_pipeline() -> Pipeline:
    """
    Pipeline para features binárias já representadas
    numericamente.

    Não há scaling específico neste grupo.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )


def _build_log_numeric_pipeline() -> Pipeline:
    """
    Pipeline para variáveis numéricas positivas e assimétricas.

    Ordem:
    1. imputação pela mediana;
    2. log1p;
    3. StandardScaler.

    A mediana é calculada somente durante fit(), portanto
    permanece protegida contra leakage quando o preprocessor
    está dentro da pipeline de ML.
    """

    return Pipeline(
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
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


def _build_numeric_pipeline() -> Pipeline:
    """
    Pipeline para variáveis numéricas que permanecem
    em sua escala funcional original.

    Ordem:
    1. imputação pela mediana;
    2. StandardScaler.
    """

    return Pipeline(
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


# ============================================================
# Preprocessor principal
# ============================================================


def build_preprocessor(
    include_uf: bool = False,
    feature_set: FeatureSet = "baseline",
) -> ColumnTransformer:
    """
    Constrói o preprocessor supervisionado.

    Compatibilidade
    ---------------
    O primeiro argumento permanece `include_uf`, preservando
    compatibilidade com scripts da modelagem v1 que utilizam:

        build_preprocessor(include_uf=False)

    Exemplos conceituais
    --------------------

    Baseline original:
        build_preprocessor()

    Baseline + UF:
        build_preprocessor(include_uf=True)

    Enriquecido:
        build_preprocessor(
            feature_set="enriched"
        )

    Enriquecido + UF:
        build_preprocessor(
            include_uf=True,
            feature_set="enriched"
        )

    Importante
    ----------
    O ColumnTransformer seleciona explicitamente somente
    as features autorizadas.

    remainder="drop" garante que IDs, target, ano e outras
    colunas presentes no dataframe não entrem acidentalmente
    no modelo.
    """

    groups = get_feature_groups(
        feature_set=feature_set,
        include_uf=include_uf,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                _build_categorical_pipeline(),
                groups["categorical"],
            ),
            (
                "binary",
                _build_binary_pipeline(),
                groups["binary"],
            ),
            (
                "log_numeric",
                _build_log_numeric_pipeline(),
                groups["log_numeric"],
            ),
            (
                "numeric",
                _build_numeric_pipeline(),
                groups["numeric"],
            ),
        ],
        remainder="drop",

        # Mantém saída sparse quando houver One-Hot Encoding.
        # É adequado aos modelos já utilizados na fase atual.
        sparse_threshold=1.0,

        verbose_feature_names_out=True,
    )

    return preprocessor