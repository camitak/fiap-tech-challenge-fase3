-- ============================================================
-- Tech Challenge Fase 3
-- Gold ML enriquecida com contexto educacional
--
-- Base principal:
-- alfabetizacao_gold.gold_ml_aluno
--
-- Enriquecimento:
-- alfabetizacao_gold.contexto_educacional_municipio_rede
--
-- Estratégia temporal:
--   aluno 2023 <- Censo Escolar 2022
--   aluno 2024 <- Censo Escolar 2023
--
-- Granularidade final:
-- 1 linha por aluno_key
--
-- IMPORTANTE:
-- - nenhum aluno é removido;
-- - LEFT JOIN preserva integralmente a população original;
-- - os poucos contextos ausentes permanecem NULL;
-- - imputação ocorrerá posteriormente dentro da pipeline ML.
-- ============================================================


-- ============================================================
-- CRIAÇÃO DA GOLD ENRIQUECIDA
-- ============================================================

CREATE OR REPLACE TABLE
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

PARTITION BY
  RANGE_BUCKET(
    ano,
    GENERATE_ARRAY(2023, 2025, 1)
  )

CLUSTER BY
  id_municipio,
  sigla_uf,
  rede_nome

AS

SELECT

  -- ==========================================================
  -- Base original
  --
  -- Preservamos todas as colunas da Gold ML já validada.
  -- ==========================================================

  g.*,


  -- ==========================================================
  -- Disponibilidade do enriquecimento
  --
  -- Coluna de auditoria.
  -- Não será automaticamente usada como feature.
  -- ==========================================================

  CASE
    WHEN c.id_municipio IS NOT NULL
      THEN 1
    ELSE 0
  END AS contexto_educacional_disponivel,


  -- ==========================================================
  -- Referência temporal do Censo Escolar
  -- ==========================================================

  c.ano_contexto_educacional,


  -- ==========================================================
  -- Escala da oferta educacional
  -- ==========================================================

  c.quantidade_escolas_anos_iniciais,

  c.total_matriculas_anos_iniciais,

  c.total_docentes_anos_iniciais,

  c.total_turmas_anos_iniciais,


  -- ==========================================================
  -- Perfil territorial da rede escolar
  -- ==========================================================

  c.proporcao_escolas_rurais,


  -- ==========================================================
  -- Infraestrutura escolar
  -- ==========================================================

  c.proporcao_internet_aprendizagem,

  c.proporcao_biblioteca_sala_leitura,

  c.proporcao_laboratorio_informatica,


  -- ==========================================================
  -- Indicadores educacionais derivados
  -- ==========================================================

  c.alunos_por_turma_anos_iniciais,

  c.razao_matriculas_docentes_anos_iniciais,

  c.proporcao_matriculas_integral_anos_iniciais,


  -- ==========================================================
  -- Rastreabilidade
  -- ==========================================================

  c.fonte_contexto_educacional,

  CURRENT_TIMESTAMP()
    AS enrichment_processed_at


FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`
    AS g

LEFT JOIN
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`
    AS c

  ON
    g.ano = c.ano_alvo

    AND g.id_municipio =
        c.id_municipio

    AND g.rede_nome =
        c.rede_nome
;


-- ============================================================
-- QUERY 24
-- Reconciliação da população original x enriquecida
--
-- Esperado:
--
-- 2023 = 1.502.809
-- 2024 = 1.851.852
--
-- Nenhum aluno ou target pode ser alterado.
-- ============================================================

WITH original AS (

  SELECT

    ano,

    COUNT(*) AS total_original,

    COUNT(DISTINCT aluno_key)
      AS alunos_original,

    COUNTIF(
      alfabetizado = 0
    ) AS nao_alfabetizados_original,

    COUNTIF(
      alfabetizado = 1
    ) AS alfabetizados_original

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    ano
),

enriquecida AS (

  SELECT

    ano,

    COUNT(*) AS total_enriquecido,

    COUNT(DISTINCT aluno_key)
      AS alunos_enriquecido,

    COUNTIF(
      alfabetizado = 0
    ) AS nao_alfabetizados_enriquecido,

    COUNTIF(
      alfabetizado = 1
    ) AS alfabetizados_enriquecido,

    COUNTIF(
      contexto_educacional_disponivel = 0
    ) AS alunos_sem_contexto_educacional

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

  GROUP BY
    ano
)

SELECT

  o.ano,

  o.total_original,

  e.total_enriquecido,

  o.alunos_original,

  e.alunos_enriquecido,

  o.nao_alfabetizados_original,

  e.nao_alfabetizados_enriquecido,

  o.alfabetizados_original,

  e.alfabetizados_enriquecido,

  e.alunos_sem_contexto_educacional,

  e.total_enriquecido
    - o.total_original
      AS diferenca_linhas

FROM
  original AS o

INNER JOIN
  enriquecida AS e
USING (
  ano
)

ORDER BY
  ano;


-- ============================================================
-- QUERY 25
-- Unicidade do aluno após enriquecimento
--
-- Esperado:
-- nenhuma linha.
-- ============================================================

SELECT

  ano,

  aluno_key,

  COUNT(*) AS quantidade_linhas

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

GROUP BY

  ano,

  aluno_key

HAVING
  COUNT(*) > 1

ORDER BY
  quantidade_linhas DESC

LIMIT 100;


-- ============================================================
-- QUERY 26
-- Auditoria dos valores faltantes das novas features
-- no nível do aluno.
-- ============================================================

SELECT

  ano,

  COUNT(*) AS total_alunos,


  COUNTIF(
    contexto_educacional_disponivel = 0
  ) AS sem_contexto_educacional,


  COUNTIF(
    quantidade_escolas_anos_iniciais IS NULL
  ) AS null_quantidade_escolas,


  COUNTIF(
    total_matriculas_anos_iniciais IS NULL
  ) AS null_total_matriculas,


  COUNTIF(
    total_docentes_anos_iniciais IS NULL
  ) AS null_total_docentes,


  COUNTIF(
    total_turmas_anos_iniciais IS NULL
  ) AS null_total_turmas,


  COUNTIF(
    proporcao_escolas_rurais IS NULL
  ) AS null_proporcao_rural,


  COUNTIF(
    proporcao_internet_aprendizagem IS NULL
  ) AS null_internet,


  COUNTIF(
    proporcao_biblioteca_sala_leitura IS NULL
  ) AS null_biblioteca,


  COUNTIF(
    proporcao_laboratorio_informatica IS NULL
  ) AS null_laboratorio,


  COUNTIF(
    alunos_por_turma_anos_iniciais IS NULL
  ) AS null_alunos_por_turma,


  COUNTIF(
    razao_matriculas_docentes_anos_iniciais IS NULL
  ) AS null_matriculas_docentes,


  COUNTIF(
    proporcao_matriculas_integral_anos_iniciais
      IS NULL
  ) AS null_proporcao_integral

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

GROUP BY
  ano

ORDER BY
  ano;


-- ============================================================
-- QUERY 27
-- Validação da regra temporal
--
-- O contexto educacional deve ser sempre anterior
-- ao target:
--
-- 2023 -> 2022
-- 2024 -> 2023
--
-- Contextos ausentes são ignorados nesta checagem.
-- Esperado:
-- problemas_temporais = 0.
-- ============================================================

SELECT

  ano,

  COUNT(*) AS alunos_com_contexto,

  COUNTIF(
    ano_contexto_educacional != ano - 1
  ) AS problemas_temporais,

  MIN(
    ano_contexto_educacional
  ) AS menor_ano_contexto,

  MAX(
    ano_contexto_educacional
  ) AS maior_ano_contexto

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

WHERE
  contexto_educacional_disponivel = 1

GROUP BY
  ano

ORDER BY
  ano;


-- ============================================================
-- QUERY 28
-- Validação semântica das novas features
--
-- Esperado:
-- todos os indicadores de problema iguais a zero.
--
-- Atenção:
-- razões altas NÃO são consideradas inválidas automaticamente.
-- Aqui verificamos apenas impossibilidades matemáticas.
-- ============================================================

SELECT

  ano,

  COUNT(*) AS total_alunos,


  -- ----------------------------------------------------------
  -- Quantidades negativas
  -- ----------------------------------------------------------

  COUNTIF(
    quantidade_escolas_anos_iniciais < 0
  ) AS escolas_negativas,

  COUNTIF(
    total_matriculas_anos_iniciais < 0
  ) AS matriculas_negativas,

  COUNTIF(
    total_docentes_anos_iniciais < 0
  ) AS docentes_negativos,

  COUNTIF(
    total_turmas_anos_iniciais < 0
  ) AS turmas_negativas,


  -- ----------------------------------------------------------
  -- Proporções fora de [0, 1]
  -- ----------------------------------------------------------

  COUNTIF(
    proporcao_escolas_rurais < 0
    OR proporcao_escolas_rurais > 1
  ) AS proporcao_rural_invalida,


  COUNTIF(
    proporcao_internet_aprendizagem < 0
    OR proporcao_internet_aprendizagem > 1
  ) AS proporcao_internet_invalida,


  COUNTIF(
    proporcao_biblioteca_sala_leitura < 0
    OR proporcao_biblioteca_sala_leitura > 1
  ) AS proporcao_biblioteca_invalida,


  COUNTIF(
    proporcao_laboratorio_informatica < 0
    OR proporcao_laboratorio_informatica > 1
  ) AS proporcao_laboratorio_invalida,


  COUNTIF(
    proporcao_matriculas_integral_anos_iniciais < 0
    OR proporcao_matriculas_integral_anos_iniciais > 1
  ) AS proporcao_integral_invalida,


  -- ----------------------------------------------------------
  -- Razões negativas
  -- ----------------------------------------------------------

  COUNTIF(
    alunos_por_turma_anos_iniciais < 0
  ) AS alunos_turma_negativo,


  COUNTIF(
    razao_matriculas_docentes_anos_iniciais < 0
  ) AS matriculas_docentes_negativo


FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

GROUP BY
  ano

ORDER BY
  ano;