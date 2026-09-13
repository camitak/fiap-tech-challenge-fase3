from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "base_modelagem"
    / "v1"
)


EXPECTED = {
    2023: {
        "files": 8,
        "total": 1_502_809,
        "alfabetizados": 877_427,
        "nao_alfabetizados": 625_382,
    },
    2024: {
        "files": 11,
        "total": 1_851_852,
        "alfabetizados": 1_107_119,
        "nao_alfabetizados": 744_733,
    },
}


REQUIRED_COLUMNS = {
    "ano",
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
    "participacao_servicos_2020",
    "participacao_administracao_publica_2020",
    "id_municipio",
}


def validate_year(year: int) -> None:
    year_path = DATA_ROOT / f"ano={year}"
    files = sorted(year_path.glob("*.parquet"))

    assert files, f"Nenhum arquivo encontrado para {year}."

    expected = EXPECTED[year]

    assert len(files) == expected["files"], (
        f"{year}: número inesperado de arquivos."
    )

    total = 0
    alfabetizados = 0
    nao_alfabetizados = 0

    for index, file_path in enumerate(files):
        df = pd.read_parquet(file_path)

        if index == 0:
            missing_columns = REQUIRED_COLUMNS - set(df.columns)

            assert not missing_columns, (
                f"{year}: colunas ausentes: "
                f"{sorted(missing_columns)}"
            )

        assert set(df["ano"].dropna().unique()) == {year}, (
            f"{year}: arquivo contém ano incompatível."
        )

        valid_targets = set(
            df["alfabetizado"].dropna().unique()
        )

        assert valid_targets.issubset({0, 1}), (
            f"{year}: target possui valores inválidos."
        )

        total += len(df)

        alfabetizados += int(
            (df["alfabetizado"] == 1).sum()
        )

        nao_alfabetizados += int(
            (df["alfabetizado"] == 0).sum()
        )

    assert total == expected["total"], (
        f"{year}: total esperado {expected['total']:,}, "
        f"obtido {total:,}."
    )

    assert alfabetizados == expected["alfabetizados"], (
        f"{year}: quantidade de alfabetizados divergente."
    )

    assert nao_alfabetizados == expected["nao_alfabetizados"], (
        f"{year}: quantidade de não alfabetizados divergente."
    )

    assert alfabetizados + nao_alfabetizados == total, (
        f"{year}: existem targets nulos ou inválidos."
    )

    print(f"\nAno: {year}")
    print(f"Arquivos: {len(files)}")
    print(f"Registros: {total:,}")
    print(f"Alfabetizados: {alfabetizados:,}")
    print(f"Não alfabetizados: {nao_alfabetizados:,}")
    print("✓ Reconciliação OK")


def main() -> None:
    print("Validando base de modelagem supervisionada...")

    for year in sorted(EXPECTED):
        validate_year(year)

    print("\n✓ Base completa validada com sucesso.")


if __name__ == "__main__":
    main()