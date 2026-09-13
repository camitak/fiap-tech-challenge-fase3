-- ============================================================
-- Tech Challenge Fase 3
-- Auditoria final do contexto educacional antes do join Gold ML
--
-- Objetivos:
--   1. investigar os 75 registros de 2023 sem correspondência;
--   2. validar unicidade da chave do contexto educacional;
--   3. medir nulls efetivamente recebidos pelos alunos;
--   4. investigar valores extremos das razões educacionais.
--
-- Nenhuma tabela é alterada nesta etapa.
-- ============================================================


-- ============================================================
-- QUERY 19
-- Registros de alunos sem contexto educacional correspondente
-- ============================================================

SELECT

  g.ano,
  g.id_municipio,
  g.municipio,
  g.sigla_uf,
  g.rede_nome,

  COUNT(*) AS quantidade_alunos

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

WHERE
  c.id_municipio IS NULL

GROUP BY

  g.ano,
  g.id_municipio,
  g.municipio,
  g.sigla_uf,
  g.rede_nome

ORDER BY

  g.ano,
  quantidade_alunos DESC,
  g.sigla_uf,
  g.municipio;


-- ============================================================
-- QUERY 20
-- Unicidade da chave da tabela de contexto.
--
-- O join deve ser no máximo 1:1 no nível:
--
-- ano_alvo + id_municipio + rede_nome
--
-- Se esta query retornar zero linhas, a chave está íntegra.
-- ============================================================

SELECT

  ano_alvo,
  id_municipio,
  rede_nome,

  COUNT(*) AS quantidade_linhas

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`

GROUP BY

  ano_alvo,
  id_municipio,
  rede_nome

HAVING
  COUNT(*) > 1

ORDER BY
  quantidade_linhas DESC;


-- ============================================================
-- QUERY 21
-- Nulls das features educacionais ponderados pelos alunos.
--
-- Essa é a auditoria mais importante para o modelo.
-- Não interessa apenas quantos contextos municipais têm null:
-- precisamos saber quantos ALUNOS seriam afetados.
-- ============================================================

SELECT

  g.ano,

  COUNT(*) AS total_alunos,


  -- ----------------------------------------------------------
  -- Sem qualquer contexto educacional
  -- ----------------------------------------------------------

  COUNTIF(
    c.id_municipio IS NULL
  ) AS alunos_sem_contexto,


  -- ----------------------------------------------------------
  -- Infraestrutura
  -- ----------------------------------------------------------

  COUNTIF(
    c.proporcao_escolas_rurais IS NULL
  ) AS null_proporcao_escolas_rurais,

  COUNTIF(
    c.proporcao_internet_aprendizagem IS NULL
  ) AS null_internet_aprendizagem,

  COUNTIF(
    c.proporcao_biblioteca_sala_leitura IS NULL
  ) AS null_biblioteca_sala_leitura,

  COUNTIF(
    c.proporcao_laboratorio_informatica IS NULL
  ) AS null_laboratorio_informatica,


  -- ----------------------------------------------------------
  -- Escala educacional
  -- ----------------------------------------------------------

  COUNTIF(
    c.quantidade_escolas_anos_iniciais IS NULL
  ) AS null_quantidade_escolas,

  COUNTIF(
    c.total_matriculas_anos_iniciais IS NULL
  ) AS null_total_matriculas,

  COUNTIF(
    c.total_docentes_anos_iniciais IS NULL
  ) AS null_total_docentes,

  COUNTIF(
    c.total_turmas_anos_iniciais IS NULL
  ) AS null_total_turmas,


  -- ----------------------------------------------------------
  -- Razões derivadas
  -- ----------------------------------------------------------

  COUNTIF(
    c.alunos_por_turma_anos_iniciais IS NULL
  ) AS null_alunos_por_turma,

  COUNTIF(
    c.razao_matriculas_docentes_anos_iniciais IS NULL
  ) AS null_matriculas_docentes,

  COUNTIF(
    c.proporcao_matriculas_integral_anos_iniciais IS NULL
  ) AS null_proporcao_integral,


  -- ----------------------------------------------------------
  -- Percentuais
  -- ----------------------------------------------------------

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        c.alunos_por_turma_anos_iniciais IS NULL
      ),
      COUNT(*)
    ),
    4
  ) AS pct_null_alunos_por_turma,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        c.razao_matriculas_docentes_anos_iniciais IS NULL
      ),
      COUNT(*)
    ),
    4
  ) AS pct_null_matriculas_docentes

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

GROUP BY
  g.ano

ORDER BY
  g.ano;


-- ============================================================
-- QUERY 22
-- Distribuição das razões SOMENTE nos contextos realmente
-- utilizados pelos alunos da avaliação.
--
-- DISTINCT evita que municípios com muitos alunos dominem
-- artificialmente os percentis desta análise de contexto.
-- ============================================================

WITH contextos_utilizados AS (

  SELECT DISTINCT

    g.ano,

    c.id_municipio,

    c.rede_nome,

    c.quantidade_escolas_anos_iniciais,

    c.total_matriculas_anos_iniciais,

    c.total_docentes_anos_iniciais,

    c.total_turmas_anos_iniciais,

    c.alunos_por_turma_anos_iniciais,

    c.razao_matriculas_docentes_anos_iniciais,

    c.proporcao_escolas_rurais,

    c.proporcao_internet_aprendizagem,

    c.proporcao_biblioteca_sala_leitura,

    c.proporcao_laboratorio_informatica,

    c.proporcao_matriculas_integral_anos_iniciais

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`
      AS g

  INNER JOIN
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`
      AS c

    ON
      g.ano = c.ano_alvo

      AND g.id_municipio =
          c.id_municipio

      AND g.rede_nome =
          c.rede_nome

)

SELECT

  ano,

  COUNT(*) AS contextos_utilizados,


  APPROX_QUANTILES(
    alunos_por_turma_anos_iniciais,
    100
  )[SAFE_OFFSET(0)] AS alunos_turma_min,

  APPROX_QUANTILES(
    alunos_por_turma_anos_iniciais,
    100
  )[SAFE_OFFSET(50)] AS alunos_turma_p50,

  APPROX_QUANTILES(
    alunos_por_turma_anos_iniciais,
    100
  )[SAFE_OFFSET(95)] AS alunos_turma_p95,

  APPROX_QUANTILES(
    alunos_por_turma_anos_iniciais,
    100
  )[SAFE_OFFSET(99)] AS alunos_turma_p99,

  APPROX_QUANTILES(
    alunos_por_turma_anos_iniciais,
    100
  )[SAFE_OFFSET(100)] AS alunos_turma_max,


  APPROX_QUANTILES(
    razao_matriculas_docentes_anos_iniciais,
    100
  )[SAFE_OFFSET(0)] AS matriculas_docentes_min,

  APPROX_QUANTILES(
    razao_matriculas_docentes_anos_iniciais,
    100
  )[SAFE_OFFSET(50)] AS matriculas_docentes_p50,

  APPROX_QUANTILES(
    razao_matriculas_docentes_anos_iniciais,
    100
  )[SAFE_OFFSET(95)] AS matriculas_docentes_p95,

  APPROX_QUANTILES(
    razao_matriculas_docentes_anos_iniciais,
    100
  )[SAFE_OFFSET(99)] AS matriculas_docentes_p99,

  APPROX_QUANTILES(
    razao_matriculas_docentes_anos_iniciais,
    100
  )[SAFE_OFFSET(100)] AS matriculas_docentes_max

FROM
  contextos_utilizados

GROUP BY
  ano

ORDER BY
  ano;


-- ============================================================
-- QUERY 23
-- Inspeção dos valores extremos.
--
-- Queremos entender se os extremos ocorrem em contextos muito
-- pequenos, por exemplo uma rede municipal com apenas uma escola,
-- uma turma ou um docente registrado.
-- ============================================================

WITH contextos_utilizados AS (

  SELECT DISTINCT

    g.ano,

    g.sigla_uf,

    g.municipio,

    c.id_municipio,

    c.rede_nome,

    c.quantidade_escolas_anos_iniciais,

    c.total_matriculas_anos_iniciais,

    c.total_docentes_anos_iniciais,

    c.total_turmas_anos_iniciais,

    c.alunos_por_turma_anos_iniciais,

    c.razao_matriculas_docentes_anos_iniciais

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`
      AS g

  INNER JOIN
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`
      AS c

    ON
      g.ano = c.ano_alvo

      AND g.id_municipio =
          c.id_municipio

      AND g.rede_nome =
          c.rede_nome

)

SELECT
  *
FROM
  contextos_utilizados

WHERE

  alunos_por_turma_anos_iniciais >= 50

  OR

  razao_matriculas_docentes_anos_iniciais >= 50

ORDER BY

  GREATEST(
    COALESCE(
      alunos_por_turma_anos_iniciais,
      0
    ),
    COALESCE(
      razao_matriculas_docentes_anos_iniciais,
      0
    )
  ) DESC

LIMIT 100;