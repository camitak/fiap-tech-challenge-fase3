from __future__ import annotations

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


def load_modeling_data(year: int) -> pd.DataFrame:
    """
    Carrega todos os shards Parquet de um ano.

    Parâmetros
    ----------
    year:
        Ano da base. Valores permitidos: 2023 ou 2024.
    """

    if year not in {2023, 2024}:
        raise ValueError(
            "year deve ser 2023 ou 2024."
        )

    year_path = DATA_ROOT / f"ano={year}"

    files = sorted(
        year_path.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum Parquet encontrado em {year_path}"
        )

    frames = [
        pd.read_parquet(file_path)
        for file_path in files
    ]

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    if set(df["ano"].unique()) != {year}:
        raise ValueError(
            f"A base contém registros fora de {year}."
        )

    return df