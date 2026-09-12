-- ============================================================
-- Tech Challenge Fase 3
-- Validação da tabela gold_ml_aluno
--
-- Tabela:
-- alfabetizacao_gold.gold_ml_aluno
-- ============================================================


-- ============================================================
-- VALIDAÇÃO 1
--
-- Verifica:
-- - quantidade de linhas
-- - unicidade
-- - distribuição do target
-- - valores nulos nas principais features
-- ============================================================

SELECT
  ano,

  COUNT(*) AS total_linhas,

  COUNT(DISTINCT aluno_key)
    AS alunos_distintos,

  COUNTIF(alfabetizado = 0)
    AS nao_alfabetizados,

  COUNTIF(alfabetizado = 1)
    AS alfabetizados,

  COUNTIF(alfabetizado IS NULL)
    AS target_nulo,

  COUNTIF(id_municipio IS NULL)
    AS municipio_nulo,

  COUNTIF(sigla_uf IS NULL)
    AS uf_nula,

  COUNTIF(regiao IS NULL)
    AS regiao_nula,

  COUNTIF(populacao_2020 IS NULL)
    AS populacao_nula,

  COUNTIF(pib_por_habitante_2020 IS NULL)
    AS pib_por_habitante_nulo,

  COUNTIF(
    participacao_agropecuaria_2020 IS NULL
  ) AS agro_nulo,

  COUNTIF(
    participacao_industria_2020 IS NULL
  ) AS industria_nula,

  COUNTIF(
    participacao_servicos_2020 IS NULL
  ) AS servicos_nulo,

  COUNTIF(
    participacao_administracao_publica_2020 IS NULL
  ) AS administracao_publica_nula

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano

ORDER BY
  ano;


-- ============================================================
-- RESULTADO ESPERADO / OBTIDO
--
-- 2023
-- total_linhas:     1.502.809
-- alunos_distintos: 1.502.809
-- não alfabetizados: 625.382
-- alfabetizados:     877.427
--
-- 2024
-- total_linhas:     1.851.852
-- alunos_distintos: 1.851.852
-- não alfabetizados: 744.733
-- alfabetizados:   1.107.119
--
-- Valores nulos verificados: 0
-- ============================================================



-- ============================================================
-- VALIDAÇÃO 2
--
-- Verifica:
-- - unicidade global da chave
-- - domínio do target
-- - soma das participações econômicas
-- ============================================================

SELECT

  COUNT(*) AS total_linhas,

  COUNT(
    DISTINCT CONCAT(
      CAST(ano AS STRING),
      '|',
      aluno_key
    )
  ) AS chaves_distintas,

  COUNTIF(
    alfabetizado NOT IN (0, 1)
    OR alfabetizado IS NULL
  ) AS targets_invalidos,

  MIN(
    participacao_agropecuaria_2020
    + participacao_industria_2020
    + participacao_servicos_2020
    + participacao_administracao_publica_2020
  ) AS menor_soma_participacoes,

  MAX(
    participacao_agropecuaria_2020
    + participacao_industria_2020
    + participacao_servicos_2020
    + participacao_administracao_publica_2020
  ) AS maior_soma_participacoes

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`;


-- ============================================================
-- RESULTADO OBTIDO
--
-- total_linhas:
-- 3.354.661
--
-- chaves_distintas:
-- 3.354.661
--
-- targets_invalidos:
-- 0
--
-- menor_soma_participacoes:
-- 0.99993449924674138
--
-- maior_soma_participacoes:
-- 1.0000436185989705
--
-- Conclusão:
-- A Gold preserva a granularidade aluno-ano,
-- possui target válido e não apresenta duplicidade
-- na chave analítica.
-- ============================================================