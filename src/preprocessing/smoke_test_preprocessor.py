from src.preprocessing.data_loader import load_modeling_data
from src.preprocessing.preprocessor import (
    BASELINE_FEATURES,
    build_preprocessor,
)


def main() -> None:
    df = load_modeling_data(2023)

    print("Shape da base:", df.shape)

    X = df[BASELINE_FEATURES]
    y = df["alfabetizado"]

    print("Features baseline:", len(BASELINE_FEATURES))
    print("Target:")
    print(y.value_counts().sort_index())

    sample = X.sample(
        n=10_000,
        random_state=42,
    )

    print("\nCategorias presentes no smoke test:")
    print(
        "Rede:",
        sorted(sample["rede_nome"].unique()),
    )
    print(
        "Região:",
        sorted(sample["regiao"].unique()),
    )

    preprocessor = build_preprocessor()

    Xt = preprocessor.fit_transform(sample)

    print("\nShape antes:", sample.shape)
    print("Shape depois:", Xt.shape)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print("\nFeatures transformadas:")
    for name in feature_names:
        print("-", name)

    feature_names_text = " ".join(feature_names)

    forbidden_features = [
        "ano",
        "alfabetizado",
        "id_municipio",
        "participacao_servicos_2020",
        "sigla_uf",
    ]

    for feature in forbidden_features:
        assert feature not in feature_names_text, (
            f"Feature proibida encontrada: {feature}"
        )

    assert {
        "Centro-Oeste",
        "Nordeste",
        "Norte",
        "Sudeste",
        "Sul",
    }.issubset(set(sample["regiao"].unique()))

    assert {
        "Estadual",
        "Municipal",
    }.issubset(set(sample["rede_nome"].unique()))

    print(
        "\n✓ Pré-processamento executado com sucesso."
    )
    print(
        "✓ Auditoria de features proibidas OK."
    )


if __name__ == "__main__":
    main()