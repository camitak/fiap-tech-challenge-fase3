from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "base_modelagem"
    / "v2"
)


EXPECTED_COLUMNS = [
    "ano",
    "aluno_key",
    "id_municipio",
    "alfabetizado",
    "rede_nome",
    "regiao",
    "sigla_uf",
    "capital_uf",
    "amazonia_legal",
    "populacao_2020",
    "pib_por_habitante_2020",
    "participacao_agropecuaria_2020",
    "participacao_industria_2020",
    "participacao_administracao_publica_2020",
    "quantidade_escolas_anos_iniciais",
    "total_matriculas_anos_iniciais",
    "total_docentes_anos_iniciais",
    "total_turmas_anos_iniciais",
    "proporcao_escolas_rurais",
    "proporcao_internet_aprendizagem",
    "proporcao_biblioteca_sala_leitura",
    "proporcao_laboratorio_informatica",
    "alunos_por_turma_anos_iniciais",
    "razao_matriculas_docentes_anos_iniciais",
    "proporcao_matriculas_integral_anos_iniciais",
]


EXPECTED_COUNTS = {
    2023: {
        "rows": 1_502_809,
        "non_literate": 625_382,
        "literate": 877_427,
    },
    2024: {
        "rows": 1_851_852,
        "non_literate": 744_733,
        "literate": 1_107_119,
    },
}


EXPECTED_EDUCATIONAL_NULLS = {
    2023: {
        "quantidade_escolas_anos_iniciais": 75,
        "total_matriculas_anos_iniciais": 75,
        "total_docentes_anos_iniciais": 75,
        "total_turmas_anos_iniciais": 75,
        "proporcao_escolas_rurais": 75,
        "proporcao_internet_aprendizagem": 75,
        "proporcao_biblioteca_sala_leitura": 75,
        "proporcao_laboratorio_informatica": 75,
        "alunos_por_turma_anos_iniciais": 86,
        "razao_matriculas_docentes_anos_iniciais": 86,
        "proporcao_matriculas_integral_anos_iniciais": 75,
    },
    2024: {
        "quantidade_escolas_anos_iniciais": 0,
        "total_matriculas_anos_iniciais": 0,
        "total_docentes_anos_iniciais": 0,
        "total_turmas_anos_iniciais": 0,
        "proporcao_escolas_rurais": 0,
        "proporcao_internet_aprendizagem": 0,
        "proporcao_biblioteca_sala_leitura": 0,
        "proporcao_laboratorio_informatica": 0,
        "alunos_por_turma_anos_iniciais": 9,
        "razao_matriculas_docentes_anos_iniciais": 9,
        "proporcao_matriculas_integral_anos_iniciais": 0,
    },
}


PROPORTION_COLUMNS = [
    "proporcao_escolas_rurais",
    "proporcao_internet_aprendizagem",
    "proporcao_biblioteca_sala_leitura",
    "proporcao_laboratorio_informatica",
    "proporcao_matriculas_integral_anos_iniciais",
]


NON_NEGATIVE_COLUMNS = [
    "populacao_2020",
    "pib_por_habitante_2020",
    "participacao_agropecuaria_2020",
    "participacao_industria_2020",
    "participacao_administracao_publica_2020",
    "quantidade_escolas_anos_iniciais",
    "total_matriculas_anos_iniciais",
    "total_docentes_anos_iniciais",
    "total_turmas_anos_iniciais",
    "proporcao_escolas_rurais",
    "proporcao_internet_aprendizagem",
    "proporcao_biblioteca_sala_leitura",
    "proporcao_laboratorio_informatica",
    "alunos_por_turma_anos_iniciais",
    "razao_matriculas_docentes_anos_iniciais",
    "proporcao_matriculas_integral_anos_iniciais",
]


def load_year(year: int) -> pd.DataFrame:
    path = DATA_ROOT / f"ano={year}"

    files = sorted(
        path.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo Parquet encontrado em: {path}"
        )

    frames = [
        pd.read_parquet(file)
        for file in files
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def validate_schema(
    df: pd.DataFrame,
    year: int,
) -> None:
    actual_columns = list(
        df.columns
    )

    if actual_columns != EXPECTED_COLUMNS:
        missing = [
            column
            for column in EXPECTED_COLUMNS
            if column not in actual_columns
        ]

        unexpected = [
            column
            for column in actual_columns
            if column not in EXPECTED_COLUMNS
        ]

        raise AssertionError(
            f"{year}: schema diferente do esperado.\n"
            f"Faltantes: {missing}\n"
            f"Extras: {unexpected}\n"
            f"Ordem atual: {actual_columns}"
        )


def validate_counts(
    df: pd.DataFrame,
    year: int,
) -> None:
    expected = EXPECTED_COUNTS[
        year
    ]

    total_rows = len(df)

    unique_students = (
        df["aluno_key"]
        .nunique()
    )

    non_literate = int(
        (df["alfabetizado"] == 0)
        .sum()
    )

    literate = int(
        (df["alfabetizado"] == 1)
        .sum()
    )

    assert (
        total_rows
        == expected["rows"]
    ), (
        f"{year}: total incorreto. "
        f"Esperado={expected['rows']:,}; "
        f"obtido={total_rows:,}"
    )

    assert (
        unique_students
        == expected["rows"]
    ), (
        f"{year}: aluno_key não é único. "
        f"Linhas={total_rows:,}; "
        f"chaves={unique_students:,}"
    )

    assert (
        non_literate
        == expected["non_literate"]
    ), (
        f"{year}: não alfabetizados incorretos. "
        f"Esperado={expected['non_literate']:,}; "
        f"obtido={non_literate:,}"
    )

    assert (
        literate
        == expected["literate"]
    ), (
        f"{year}: alfabetizados incorretos. "
        f"Esperado={expected['literate']:,}; "
        f"obtido={literate:,}"
    )

    invalid_targets = int(
        (~df["alfabetizado"].isin([0, 1]))
        .sum()
    )

    assert invalid_targets == 0, (
        f"{year}: encontrados "
        f"{invalid_targets:,} targets inválidos."
    )


def validate_year(
    df: pd.DataFrame,
    year: int,
) -> None:
    unique_years = (
        df["ano"]
        .dropna()
        .unique()
        .tolist()
    )

    assert unique_years == [year], (
        f"{year}: coluna ano contém valores inesperados: "
        f"{unique_years}"
    )


def validate_required_columns(
    df: pd.DataFrame,
    year: int,
) -> None:
    required = [
        "ano",
        "aluno_key",
        "id_municipio",
        "alfabetizado",
        "rede_nome",
        "regiao",
        "sigla_uf",
        "capital_uf",
        "amazonia_legal",
        "populacao_2020",
        "pib_por_habitante_2020",
        "participacao_agropecuaria_2020",
        "participacao_industria_2020",
        "participacao_administracao_publica_2020",
    ]

    null_counts = (
        df[required]
        .isna()
        .sum()
    )

    invalid = (
        null_counts[
            null_counts > 0
        ]
    )

    assert invalid.empty, (
        f"{year}: nulls inesperados nas colunas base:\n"
        f"{invalid}"
    )


def validate_educational_nulls(
    df: pd.DataFrame,
    year: int,
) -> None:
    expected = (
        EXPECTED_EDUCATIONAL_NULLS[
            year
        ]
    )

    for (
        column,
        expected_nulls,
    ) in expected.items():

        actual_nulls = int(
            df[column]
            .isna()
            .sum()
        )

        assert (
            actual_nulls
            == expected_nulls
        ), (
            f"{year}: nulls inesperados em {column}. "
            f"Esperado={expected_nulls:,}; "
            f"obtido={actual_nulls:,}"
        )


def validate_proportions(
    df: pd.DataFrame,
    year: int,
) -> None:
    for column in PROPORTION_COLUMNS:

        valid = (
            df[column]
            .dropna()
        )

        invalid = (
            (valid < 0)
            | (valid > 1)
        )

        invalid_count = int(
            invalid.sum()
        )

        assert invalid_count == 0, (
            f"{year}: {column} possui "
            f"{invalid_count:,} valores fora de [0, 1]."
        )


def validate_non_negative(
    df: pd.DataFrame,
    year: int,
) -> None:
    for column in NON_NEGATIVE_COLUMNS:

        valid = (
            df[column]
            .dropna()
        )

        invalid_count = int(
            (valid < 0)
            .sum()
        )

        assert invalid_count == 0, (
            f"{year}: {column} possui "
            f"{invalid_count:,} valores negativos."
        )


def print_summary(
    df: pd.DataFrame,
    year: int,
) -> None:
    print(
        "\n"
        + "=" * 70
    )

    print(
        f"BASE MODELAGEM V2 - {year}"
    )

    print(
        "=" * 70
    )

    print(
        f"Linhas: {len(df):,}"
    )

    print(
        "Alunos distintos:",
        f"{df['aluno_key'].nunique():,}",
    )

    print(
        "Municípios:",
        f"{df['id_municipio'].nunique():,}",
    )

    print(
        "Não alfabetizados:",
        f"{(df['alfabetizado'] == 0).sum():,}",
    )

    print(
        "Alfabetizados:",
        f"{(df['alfabetizado'] == 1).sum():,}",
    )

    print(
        "Número de colunas:",
        len(df.columns),
    )

    print(
        "\nNulls educacionais:"
    )

    educational_columns = (
        list(
            EXPECTED_EDUCATIONAL_NULLS[
                year
            ].keys()
        )
    )

    print(
        df[
            educational_columns
        ]
        .isna()
        .sum()
        .to_string()
    )


def main() -> None:
    for year in (
        2023,
        2024,
    ):
        print(
            f"\nCarregando {year}..."
        )

        df = load_year(
            year
        )

        validate_schema(
            df,
            year,
        )

        validate_counts(
            df,
            year,
        )

        validate_year(
            df,
            year,
        )

        validate_required_columns(
            df,
            year,
        )

        validate_educational_nulls(
            df,
            year,
        )

        validate_proportions(
            df,
            year,
        )

        validate_non_negative(
            df,
            year,
        )

        print_summary(
            df,
            year,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDAÇÃO CONCLUÍDA COM SUCESSO"
    )

    print(
        "=" * 70
    )

    print(
        "A base_modelagem/v2 preserva a população original "
        "e o enriquecimento educacional passou nas regras "
        "de qualidade esperadas."
    )


if __name__ == "__main__":
    main()