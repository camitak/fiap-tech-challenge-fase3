-- Exportação da base oficial de modelagem supervisionada - Fase 3
--
-- Origem:
-- fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno
--
-- Estratégia temporal:
-- 2023 = desenvolvimento / validação
-- 2024 = teste temporal final
--
-- Os arquivos são exportados em Parquet para um bucket específico da Fase 3.
-- O conjunto de 2024 deve permanecer intocado durante seleção de features,
-- comparação de modelos e ajuste de hiperparâmetros.

EXPORT DATA OPTIONS (
  uri = 'gs://fiap-tc-f2-camila-takemoto-fase3-ml/base_modelagem/v1/ano=2023/part-*.parquet',
  format = 'PARQUET',
  overwrite = TRUE
)
AS
SELECT
    ano,
    alfabetizado,

    rede_nome,
    regiao,
    sigla_uf,

    capital_uf,
    amazonia_legal,

    populacao_2020,
    pib_por_habitante_2020,
    participacao_agropecuaria_2020,
    participacao_industria_2020,
    participacao_servicos_2020,
    participacao_administracao_publica_2020,

    -- Apoio à avaliação territorial; não é feature do modelo.
    id_municipio

FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

WHERE
    ano = 2023;


EXPORT DATA OPTIONS (
  uri = 'gs://fiap-tc-f2-camila-takemoto-fase3-ml/base_modelagem/v1/ano=2024/part-*.parquet',
  format = 'PARQUET',
  overwrite = TRUE
)
AS
SELECT
    ano,
    alfabetizado,

    rede_nome,
    regiao,
    sigla_uf,

    capital_uf,
    amazonia_legal,

    populacao_2020,
    pib_por_habitante_2020,
    participacao_agropecuaria_2020,
    participacao_industria_2020,
    participacao_servicos_2020,
    participacao_administracao_publica_2020,

    -- Apoio à avaliação territorial; não é feature do modelo.
    id_municipio

FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

WHERE
    ano = 2024;