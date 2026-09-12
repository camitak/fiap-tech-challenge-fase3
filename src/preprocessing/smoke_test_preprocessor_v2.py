from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse

from src.preprocessing.preprocessor import (
    BASELINE_FEATURES,
    EDUCATIONAL_REDUNDANT_FEATURES,
    EDUCATIONAL_SELECTED_FEATURES,
    ENRICHED_FEATURES,
    EXPERIMENTAL_UF_FEATURE,
    FORBIDDEN_MODEL_FEATURES,
    build_preprocessor,
    get_model_features,
)


# ============================================================
# Tech Challenge Fase 3
# Smoke test do preprocessing v2
#
# Objetivos:
# - validar compatibilidade do baseline;
# - validar conjunto enriquecido;
# - validar conjunto enriquecido + UF;
# - verificar imputação das novas features;
# - garantir que colunas proibidas não entram no modelo;
# - garantir que features redundantes não entram no conjunto
#   enriquecido principal.
#
# Somente 2023 é utilizado.
# ============================================================


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


SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


# ============================================================
# Colunas necessárias para o smoke test
# ============================================================


READ_COLUMNS = list(
    dict.fromkeys(
        [
            "aluno_key",
            "alfabetizado",
        ]
        + ENRICHED_FEATURES
        + [
            EXPERIMENTAL_UF_FEATURE
        ]
        + EDUCATIONAL_REDUNDANT_FEATURES
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
            "Nenhum arquivo Parquet encontrado em: "
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

    expected_rows = 1_502_809

    assert len(df) == expected_rows, (
        "Quantidade de linhas inesperada. "
        f"Esperado={expected_rows:,}; "
        f"obtido={len(df):,}"
    )

    return df


# ============================================================
# Amostra
# ============================================================


def build_test_sample(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria amostra determinística e força a presença de registros
    com nulls educacionais para validar a imputação.
    """

    random_sample = df.sample(
        n=min(
            SAMPLE_SIZE,
            len(df),
        ),
        random_state=RANDOM_STATE,
    )

    rows_with_nulls = df[
        EDUCATIONAL_SELECTED_FEATURES
    ].isna().any(
        axis=1
    )

    null_sample = (
        df.loc[
            rows_with_nulls
        ]
        .head(500)
    )

    sample = (
        pd.concat(
            [
                random_sample,
                null_sample,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "aluno_key"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return sample


# ============================================================
# Helpers de validação
# ============================================================


def assert_finite_matrix(
    matrix,
    label: str,
) -> None:
    if sparse.issparse(
        matrix
    ):
        values = matrix.data

    else:
        values = np.asarray(
            matrix
        )

    assert np.isfinite(
        values
    ).all(), (
        f"{label}: matriz transformada contém "
        "NaN ou infinito."
    )


def validate_feature_lists() -> None:

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = get_model_features(
        feature_set="baseline",
        include_uf=False,
    )

    assert baseline == BASELINE_FEATURES

    assert len(
        baseline
    ) == 9


    # --------------------------------------------------------
    # Enriched
    # --------------------------------------------------------

    enriched = get_model_features(
        feature_set="enriched",
        include_uf=False,
    )

    assert enriched == ENRICHED_FEATURES

    assert len(
        enriched
    ) == 17


    # --------------------------------------------------------
    # Enriched + UF
    # --------------------------------------------------------

    enriched_uf = get_model_features(
        feature_set="enriched",
        include_uf=True,
    )

    assert len(
        enriched_uf
    ) == 18

    assert (
        EXPERIMENTAL_UF_FEATURE
        in enriched_uf
    )


    # --------------------------------------------------------
    # Colunas proibidas
    # --------------------------------------------------------

    for feature in (
        FORBIDDEN_MODEL_FEATURES
    ):
        assert (
            feature
            not in enriched_uf
        ), (
            "Feature proibida presente: "
            f"{feature}"
        )


    # --------------------------------------------------------
    # Features redundantes
    # --------------------------------------------------------

    for feature in (
        EDUCATIONAL_REDUNDANT_FEATURES
    ):
        assert (
            feature
            not in enriched_uf
        ), (
            "Feature educacional redundante entrou "
            f"no conjunto principal: {feature}"
        )


# ============================================================
# Teste de um conjunto
# ============================================================


def test_configuration(
    sample: pd.DataFrame,
    *,
    label: str,
    feature_set: str,
    include_uf: bool,
) -> dict:
    features = get_model_features(
        feature_set=feature_set,
        include_uf=include_uf,
    )

    X = sample[
        features
    ].copy()

    preprocessor = (
        build_preprocessor(
            feature_set=feature_set,
            include_uf=include_uf,
        )
    )

    transformed = (
        preprocessor.fit_transform(
            X
        )
    )

    assert (
        transformed.shape[0]
        == len(X)
    ), (
        f"{label}: número de linhas mudou "
        "durante transformação."
    )

    assert_finite_matrix(
        transformed,
        label,
    )

    output_names = (
        preprocessor
        .get_feature_names_out()
    )

    assert (
        len(output_names)
        == transformed.shape[1]
    ), (
        f"{label}: quantidade de nomes de saída "
        "não corresponde à matriz transformada."
    )

    return {
        "label": label,
        "raw_features": len(
            features
        ),
        "rows": len(
            X
        ),
        "transformed_features": (
            transformed.shape[1]
        ),
        "sparse": sparse.issparse(
            transformed
        ),
        "output_names": (
            output_names
        ),
    }


# ============================================================
# Main
# ============================================================


def main() -> None:

    print(
        "=" * 72
    )

    print(
        "SMOKE TEST PREPROCESSOR V2"
    )

    print(
        "=" * 72
    )


    # --------------------------------------------------------
    # Listas de features
    # --------------------------------------------------------

    validate_feature_lists()

    print(
        "✓ Listas de features validadas."
    )

    print(
        f"  Baseline: "
        f"{len(BASELINE_FEATURES)}"
    )

    print(
        f"  Educacionais selecionadas: "
        f"{len(EDUCATIONAL_SELECTED_FEATURES)}"
    )

    print(
        f"  Enriched: "
        f"{len(ENRICHED_FEATURES)}"
    )

    print(
        f"  Enriched + UF: "
        f"{len(ENRICHED_FEATURES) + 1}"
    )


    # --------------------------------------------------------
    # Dados
    # --------------------------------------------------------

    print(
        "\nCarregando base v2 de 2023..."
    )

    df = load_2023()

    print(
        f"✓ Base carregada: "
        f"{len(df):,} linhas."
    )


    # --------------------------------------------------------
    # Amostra
    # --------------------------------------------------------

    sample = build_test_sample(
        df
    )

    print(
        f"✓ Amostra de teste: "
        f"{len(sample):,} linhas."
    )

    null_rows = int(
        sample[
            EDUCATIONAL_SELECTED_FEATURES
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print(
        f"✓ Linhas com algum null educacional "
        f"incluídas: {null_rows:,}"
    )

    assert null_rows > 0, (
        "A amostra não contém nulls educacionais; "
        "a imputação não foi exercitada."
    )


    # --------------------------------------------------------
    # Configuração 1 — baseline
    # --------------------------------------------------------

    baseline_result = (
        test_configuration(
            sample,
            label="BASELINE",
            feature_set="baseline",
            include_uf=False,
        )
    )


    # --------------------------------------------------------
    # Configuração 2 — enriched
    # --------------------------------------------------------

    enriched_result = (
        test_configuration(
            sample,
            label="ENRICHED",
            feature_set="enriched",
            include_uf=False,
        )
    )


    # --------------------------------------------------------
    # Configuração 3 — enriched + UF
    # --------------------------------------------------------

    enriched_uf_result = (
        test_configuration(
            sample,
            label="ENRICHED + UF",
            feature_set="enriched",
            include_uf=True,
        )
    )


    # --------------------------------------------------------
    # Compatibilidade baseline
    #
    # Com 2023:
    # - rede_nome: 2 categorias;
    # - regiao: 5 categorias;
    # - 2 binárias;
    # - 5 numéricas.
    #
    # Total transformado esperado = 14.
    # --------------------------------------------------------

    assert (
        baseline_result[
            "transformed_features"
        ]
        == 14
    ), (
        "Baseline deixou de produzir 14 features "
        "transformadas. Verificar compatibilidade "
        "com a modelagem anterior."
    )


    # --------------------------------------------------------
    # Enriched deve adicionar exatamente 8 features numéricas.
    # --------------------------------------------------------

    assert (
        enriched_result[
            "transformed_features"
        ]
        == (
            baseline_result[
                "transformed_features"
            ]
            + len(
                EDUCATIONAL_SELECTED_FEATURES
            )
        )
    ), (
        "O conjunto enriquecido não adicionou "
        "exatamente as 8 features educacionais esperadas."
    )


    # --------------------------------------------------------
    # UF deve acrescentar pelo menos uma coluna OHE.
    # Não hardcodamos quantidade de UFs para evitar depender
    # da amostra do smoke test.
    # --------------------------------------------------------

    assert (
        enriched_uf_result[
            "transformed_features"
        ]
        >
        enriched_result[
            "transformed_features"
        ]
    ), (
        "A inclusão de sigla_uf não aumentou "
        "a dimensionalidade transformada."
    )


    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 72
    )

    print(
        "RESULTADOS"
    )

    print(
        "-" * 72
    )

    for result in [
        baseline_result,
        enriched_result,
        enriched_uf_result,
    ]:

        print(
            f"\n{result['label']}"
        )

        print(
            f"  Features brutas: "
            f"{result['raw_features']}"
        )

        print(
            f"  Linhas: "
            f"{result['rows']:,}"
        )

        print(
            f"  Features transformadas: "
            f"{result['transformed_features']}"
        )

        print(
            f"  Matriz sparse: "
            f"{result['sparse']}"
        )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "SMOKE TEST V2 CONCLUÍDO COM SUCESSO"
    )

    print(
        "=" * 72
    )

    print(
        "✓ Compatibilidade do baseline preservada."
    )

    print(
        "✓ Features educacionais selecionadas corretamente."
    )

    print(
        "✓ Features redundantes excluídas."
    )

    print(
        "✓ Imputação exercitada com registros nulos."
    )

    print(
        "✓ Enriched e Enriched + UF transformados sem NaN."
    )


if __name__ == "__main__":
    main()