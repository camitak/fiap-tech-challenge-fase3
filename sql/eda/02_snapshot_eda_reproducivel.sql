-- ============================================================
-- Tech Challenge Fase 3
-- Snapshot agregado para reprodução da EDA
--
-- Objetivo:
-- gerar um conjunto pequeno e não individualizado que possa
-- ser versionado no GitHub e utilizado pelo notebook sem
-- necessidade de autenticação no Google Cloud.
--
-- Granularidade:
-- município + ano + rede
--
-- Fonte oficial:
-- alfabetizacao_gold.gold_ml_aluno
-- ============================================================


SELECT
  ano,

  id_municipio,

  ANY_VALUE(municipio) AS municipio,
  ANY_VALUE(sigla_uf) AS sigla_uf,
  ANY_VALUE(regiao) AS regiao,

  ANY_VALUE(capital_uf) AS capital_uf,
  ANY_VALUE(amazonia_legal) AS amazonia_legal,

  rede_nome,

  COUNT(*) AS quantidade_alunos,

  SUM(alfabetizado)
    AS quantidade_alfabetizados,

  COUNT(*) - SUM(alfabetizado)
    AS quantidade_nao_alfabetizados,

  AVG(alfabetizado)
    AS taxa_alfabetizacao,

  ANY_VALUE(populacao_2020)
    AS populacao_2020,

  ANY_VALUE(pib_por_habitante_2020)
    AS pib_por_habitante_2020,

  ANY_VALUE(
    participacao_agropecuaria_2020
  ) AS participacao_agropecuaria_2020,

  ANY_VALUE(
    participacao_industria_2020
  ) AS participacao_industria_2020,

  ANY_VALUE(
    participacao_servicos_2020
  ) AS participacao_servicos_2020,

  ANY_VALUE(
    participacao_administracao_publica_2020
  ) AS participacao_administracao_publica_2020

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  id_municipio,
  rede_nome

ORDER BY
  ano,
  id_municipio,
  rede_nome;