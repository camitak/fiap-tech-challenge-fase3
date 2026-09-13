-- ============================================================
-- Tech Challenge Fase 3
-- Descoberta e auditoria inicial do Censo Escolar
--
-- Objetivo:
--   identificar tabelas, schema e cobertura temporal antes
--   de definir as features educacionais da Gold de ML.
--
-- Regra temporal pretendida:
--   alvo 2023 -> contexto educacional 2022
--   alvo 2024 -> contexto educacional 2023
--
-- Nenhuma feature é criada nesta etapa.
-- ============================================================


-- ============================================================
-- QUERY 01
-- Quais tabelas existem no dataset do Censo Escolar?
-- ============================================================

SELECT
  table_name,
  table_type
FROM
  `basedosdados.br_inep_censo_escolar.INFORMATION_SCHEMA.TABLES`
ORDER BY
  table_name;


-- ============================================================
-- QUERY 02
-- Schema completo da tabela escola
-- ============================================================

SELECT
  ordinal_position,
  column_name,
  data_type
FROM
  `basedosdados.br_inep_censo_escolar.INFORMATION_SCHEMA.COLUMNS`
WHERE
  table_name = 'escola'
ORDER BY
  ordinal_position;


-- ============================================================
-- QUERY 03
-- Cobertura temporal da tabela escola
-- ============================================================

SELECT
  MIN(ano) AS primeiro_ano,
  MAX(ano) AS ultimo_ano,
  COUNT(DISTINCT ano) AS quantidade_anos,
  COUNT(DISTINCT id_municipio) AS quantidade_municipios
FROM
  `basedosdados.br_inep_censo_escolar.escola`;


-- ============================================================
-- QUERY 04
-- Cobertura específica dos anos que interessam
-- ============================================================

SELECT
  ano,
  COUNT(*) AS quantidade_registros,
  COUNT(DISTINCT id_escola) AS quantidade_escolas,
  COUNT(DISTINCT id_municipio) AS quantidade_municipios
FROM
  `basedosdados.br_inep_censo_escolar.escola`
WHERE
  ano IN (2022, 2023)
GROUP BY
  ano
ORDER BY
  ano;


-- ============================================================
-- QUERY 05
-- Amostra das escolas de 2022
--
-- ============================================================

SELECT
  *
FROM
  `basedosdados.br_inep_censo_escolar.escola`
WHERE
  ano = 2022
LIMIT 10;


-- ============================================================
-- QUERY 06
-- Schema da tabela docente
--
-- ============================================================

SELECT
  ordinal_position,
  column_name,
  data_type
FROM
  `basedosdados.br_inep_censo_escolar.INFORMATION_SCHEMA.COLUMNS`
WHERE
  table_name = 'docente'
ORDER BY
  ordinal_position;


-- ============================================================
-- QUERY 07
-- Schema da tabela matricula
--
-- ============================================================

SELECT
  ordinal_position,
  column_name,
  data_type
FROM
  `basedosdados.br_inep_censo_escolar.INFORMATION_SCHEMA.COLUMNS`
WHERE
  table_name = 'matricula'
ORDER BY
  ordinal_position;


-- ============================================================
-- QUERY 08
-- Verificar se todos os municípios da Gold de ML possuem
-- correspondência no Censo Escolar 2022.
--
-- ============================================================

WITH municipios_gold AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

),

municipios_censo_2022 AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `basedosdados.br_inep_censo_escolar.escola`

  WHERE
    ano = 2022
    AND id_municipio IS NOT NULL

)

SELECT

  COUNT(*) AS municipios_gold,

  COUNTIF(
    c.id_municipio IS NOT NULL
  ) AS municipios_com_censo_2022,

  COUNTIF(
    c.id_municipio IS NULL
  ) AS municipios_sem_censo_2022

FROM
  municipios_gold AS g

LEFT JOIN
  municipios_censo_2022 AS c
USING (
  id_municipio
);


-- ============================================================
-- QUERY 09
-- Mesma auditoria para Censo Escolar 2023
-- ============================================================

WITH municipios_gold AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

),

municipios_censo_2023 AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `basedosdados.br_inep_censo_escolar.escola`

  WHERE
    ano = 2023
    AND id_municipio IS NOT NULL

)

SELECT

  COUNT(*) AS municipios_gold,

  COUNTIF(
    c.id_municipio IS NOT NULL
  ) AS municipios_com_censo_2023,

  COUNTIF(
    c.id_municipio IS NULL
  ) AS municipios_sem_censo_2023

FROM
  municipios_gold AS g

LEFT JOIN
  municipios_censo_2023 AS c
USING (
  id_municipio
);