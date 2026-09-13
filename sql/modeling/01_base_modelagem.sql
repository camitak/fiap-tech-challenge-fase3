-- Base oficial para modelagem supervisionada - Fase 3
--
-- Fonte:
-- fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno
--
-- Estratégia temporal:
-- 2023 = desenvolvimento / validação
-- 2024 = teste temporal final
--
-- Variáveis com risco de leakage, identificadores e metadados técnicos
-- são deliberadamente excluídos desta seleção.

SELECT
    ano,

    -- Target
    alfabetizado,

    -- Variáveis categóricas
    rede_nome,
    regiao,
    sigla_uf,

    -- Variáveis binárias
    capital_uf,
    amazonia_legal,

    -- Variáveis numéricas
    populacao_2020,
    pib_por_habitante_2020,
    participacao_agropecuaria_2020,
    participacao_industria_2020,
    participacao_servicos_2020,
    participacao_administracao_publica_2020,

    -- Campo de apoio para avaliação por município.
    -- Não será utilizado como feature do modelo.
    id_municipio

FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

WHERE
    ano IN (2023, 2024);