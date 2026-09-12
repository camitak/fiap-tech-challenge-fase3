-- ============================================================
-- Tech Challenge Fase 3
-- Auditoria inicial da camada Gold construída na Fase 2
--
-- Objetivos:
-- 1. inventariar os produtos Gold existentes;
-- 2. verificar a granularidade da principal tabela de features;
-- 3. verificar se já existe microdado no nível de aluno/escola.
-- ============================================================


-- ============================================================
-- QUERY 1
-- Inventário das tabelas e views da Gold
-- ============================================================

SELECT
  table_name,
  table_type

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.INFORMATION_SCHEMA.TABLES`

ORDER BY
  table_name;



-- ============================================================
-- QUERY 1B
-- Contagem dos principais produtos físicos da Gold
-- ============================================================

SELECT
  'cobertura_integracao' AS tabela,
  COUNT(*) AS quantidade_linhas
FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.cobertura_integracao`

UNION ALL

SELECT
  'distribuicao_niveis_uf',
  COUNT(*)
FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.distribuicao_niveis_uf`

UNION ALL

SELECT
  'features_modelo_municipio',
  COUNT(*)
FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.features_modelo_municipio`

UNION ALL

SELECT
  'kpi_brasil',
  COUNT(*)
FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.kpi_brasil`

UNION ALL

SELECT
  'kpi_municipio',
  COUNT(*)
FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.kpi_municipio`

UNION ALL

SELECT
  'kpi_uf',
  COUNT(*)
FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.kpi_uf`

UNION ALL

SELECT
  'resumo_executivo',
  COUNT(*)
FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.resumo_executivo`

ORDER BY
  tabela;



-- ============================================================
-- QUERY 2
-- Schema da tabela features_modelo_municipio
--
-- Objetivo:
-- confirmar que sua granularidade é municipal e não individual.
-- ============================================================

SELECT
  ordinal_position,
  column_name,
  data_type,
  is_nullable

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.INFORMATION_SCHEMA.COLUMNS`

WHERE
  table_name = 'features_modelo_municipio'

ORDER BY
  ordinal_position;



-- ============================================================
-- QUERY 3
-- Verifica presença de identificadores individuais na Gold
--
-- Objetivo:
-- verificar se alguma tabela/view Gold da Fase 2 já contém
-- id_aluno ou id_escola.
-- ============================================================

SELECT
  table_name,

  LOGICAL_OR(column_name = 'id_aluno')
    AS tem_id_aluno,

  LOGICAL_OR(column_name = 'id_escola')
    AS tem_id_escola,

  LOGICAL_OR(column_name = 'id_municipio')
    AS tem_id_municipio,

  LOGICAL_OR(column_name = 'ano')
    AS tem_ano,

  LOGICAL_OR(
    column_name IN (
      'alfabetizado',
      'alfabetizado_codigo'
    )
  ) AS tem_alfabetizado,

  LOGICAL_OR(column_name = 'proficiencia')
    AS tem_proficiencia,

  STRING_AGG(
    IF(
      column_name IN (
        'ano',
        'id_aluno',
        'id_escola',
        'id_municipio',
        'alfabetizado',
        'alfabetizado_codigo',
        'proficiencia'
      ),
      column_name,
      NULL
    ),
    ', '
    IGNORE NULLS
    ORDER BY ordinal_position
  ) AS colunas_relevantes

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.INFORMATION_SCHEMA.COLUMNS`

GROUP BY
  table_name

ORDER BY
  table_name;


/*
Conclusão da auditoria:

A Gold da Fase 2 foi construída corretamente para análises agregadas
municipais, estaduais e nacionais.

Ela não contém granularidade individual suficiente para o objetivo
supervisionado da Fase 3.

Por isso, a Gold da Fase 2 foi preservada e uma nova tabela
gold_ml_aluno foi construída especificamente para Machine Learning,
a partir da Silver validada e de enriquecimentos externos.
*/