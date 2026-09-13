-- ============================================================
-- Tech Challenge Fase 3
-- Contexto educacional municipal por rede
--
-- Fonte externa:
-- basedosdados.br_inep_censo_escolar.escola
--
-- Granularidade:
-- ano_alvo + id_municipio + rede_nome
--
-- Estratégia temporal:
--   target 2023 <- Censo Escolar 2022
--   target 2024 <- Censo Escolar 2023
--
-- Objetivo:
-- criar features educacionais anteriores ao ano alvo,
-- evitando o uso de contexto futuro.
-- ============================================================


-- ============================================================
-- CRIAÇÃO DA TABELA GOLD DE CONTEXTO EDUCACIONAL
-- ============================================================

CREATE OR REPLACE TABLE
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`

PARTITION BY
  RANGE_BUCKET(
    ano_alvo,
    GENERATE_ARRAY(2023, 2025, 1)
  )

CLUSTER BY
  id_municipio,
  rede_nome

AS

WITH escolas_base AS (

  SELECT

    -- --------------------------------------------------------
    -- Referência temporal
    -- --------------------------------------------------------

    ano AS ano_contexto_educacional,

    ano + 1 AS ano_alvo,


    -- --------------------------------------------------------
    -- Chaves
    -- --------------------------------------------------------

    id_municipio,

    rede AS rede_codigo_censo,

    CASE rede
      WHEN '1' THEN 'Federal'
      WHEN '2' THEN 'Estadual'
      WHEN '3' THEN 'Municipal'
      WHEN '4' THEN 'Privada'
      ELSE NULL
    END AS rede_nome,


    -- --------------------------------------------------------
    -- Localização
    -- --------------------------------------------------------

    tipo_localizacao,


    -- --------------------------------------------------------
    -- Infraestrutura educacional
    -- --------------------------------------------------------

    internet_aprendizagem,

    biblioteca_sala_leitura,

    laboratorio_informatica,


    -- --------------------------------------------------------
    -- Escala educacional
    -- --------------------------------------------------------

    quantidade_matricula_fundamental_anos_iniciais,

    quantidade_docente_fundamental_anos_iniciais,

    quantidade_turma_fundamental_anos_iniciais,

    quantidade_matricula_fundamental_anos_iniciais_integral


  FROM
    `basedosdados.br_inep_censo_escolar.escola`

  WHERE

    -- Somente os contextos anteriores aos targets
    ano IN (2022, 2023)

    -- Escola em atividade
    AND tipo_situacao_funcionamento = '1'

    -- Escola com oferta dos anos iniciais
    AND etapa_ensino_fundamental_anos_iniciais = 1

    -- Redes relevantes para a modelagem.
    -- Federal também é preservada para rastreabilidade,
    -- ainda que não apareça atualmente na população alvo.
    AND rede IN ('1', '2', '3', '4')

    AND id_municipio IS NOT NULL

),


agregado AS (

  SELECT

    ano_contexto_educacional,

    ano_alvo,

    id_municipio,

    rede_codigo_censo,

    rede_nome,


    -- ========================================================
    -- Quantidade de escolas
    -- ========================================================

    COUNT(*) AS
      quantidade_escolas_anos_iniciais,


    -- ========================================================
    -- Localização
    --
    -- tipo_localizacao:
    -- 1 = Urbana
    -- 2 = Rural
    -- ========================================================

    SAFE_DIVIDE(
      COUNTIF(
        tipo_localizacao = '2'
      ),
      COUNT(*)
    ) AS
      proporcao_escolas_rurais,


    -- ========================================================
    -- Infraestrutura
    --
    -- As variáveis possuem codificação binária 0/1.
    -- A média equivale à proporção de escolas que possuem
    -- o recurso.
    -- ========================================================

    AVG(
      CAST(
        internet_aprendizagem
        AS FLOAT64
      )
    ) AS
      proporcao_internet_aprendizagem,


    AVG(
      CAST(
        biblioteca_sala_leitura
        AS FLOAT64
      )
    ) AS
      proporcao_biblioteca_sala_leitura,


    AVG(
      CAST(
        laboratorio_informatica
        AS FLOAT64
      )
    ) AS
      proporcao_laboratorio_informatica,


    -- ========================================================
    -- Totais dos anos iniciais
    -- ========================================================

    SUM(
      quantidade_matricula_fundamental_anos_iniciais
    ) AS
      total_matriculas_anos_iniciais,


    SUM(
      COALESCE(
        quantidade_docente_fundamental_anos_iniciais,
        0
      )
    ) AS
      total_docentes_anos_iniciais,


    SUM(
      quantidade_turma_fundamental_anos_iniciais
    ) AS
      total_turmas_anos_iniciais,


    SUM(
      quantidade_matricula_fundamental_anos_iniciais_integral
    ) AS
      total_matriculas_integral_anos_iniciais


  FROM
    escolas_base

  GROUP BY

    ano_contexto_educacional,

    ano_alvo,

    id_municipio,

    rede_codigo_censo,

    rede_nome

)


SELECT

  ano_contexto_educacional,

  ano_alvo,

  id_municipio,

  rede_codigo_censo,

  rede_nome,


  -- ==========================================================
  -- Escala da rede educacional
  -- ==========================================================

  quantidade_escolas_anos_iniciais,

  total_matriculas_anos_iniciais,

  total_docentes_anos_iniciais,

  total_turmas_anos_iniciais,


  -- ==========================================================
  -- Perfil territorial da oferta
  -- ==========================================================

  proporcao_escolas_rurais,


  -- ==========================================================
  -- Infraestrutura escolar
  -- ==========================================================

  proporcao_internet_aprendizagem,

  proporcao_biblioteca_sala_leitura,

  proporcao_laboratorio_informatica,


  -- ==========================================================
  -- Engenharia de atributos
  -- ==========================================================

  SAFE_DIVIDE(
    total_matriculas_anos_iniciais,
    total_turmas_anos_iniciais
  ) AS
    alunos_por_turma_anos_iniciais,


  SAFE_DIVIDE(
    total_matriculas_anos_iniciais,
    total_docentes_anos_iniciais
  ) AS
    razao_matriculas_docentes_anos_iniciais,


  SAFE_DIVIDE(
    total_matriculas_integral_anos_iniciais,
    total_matriculas_anos_iniciais
  ) AS
    proporcao_matriculas_integral_anos_iniciais,


  -- ==========================================================
  -- Metadado de origem
  -- ==========================================================

  'basedosdados.br_inep_censo_escolar.escola'
    AS fonte_contexto_educacional,

  CURRENT_TIMESTAMP()
    AS gold_processed_at


FROM
  agregado;

  -- ============================================================
-- QUERY 15
-- Volume da nova Gold educacional
-- ============================================================

SELECT

  ano_alvo,

  ano_contexto_educacional,

  rede_nome,

  COUNT(*) AS
    municipio_rede,

  COUNT(DISTINCT id_municipio) AS
    municipios

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`

GROUP BY

  ano_alvo,

  ano_contexto_educacional,

  rede_nome

ORDER BY

  ano_alvo,

  rede_nome;


-- ============================================================
-- QUERY 16
-- Auditoria de cobertura no nível do aluno.
--
-- Verifica quantos registros da Gold ML encontram contexto
-- educacional do ano anterior para o mesmo município e rede.
-- ============================================================

SELECT

  g.ano,

  g.rede_nome,

  COUNT(*) AS
    alunos,

  COUNTIF(
    c.id_municipio IS NOT NULL
  ) AS
    alunos_com_contexto_educacional,

  COUNTIF(
    c.id_municipio IS NULL
  ) AS
    alunos_sem_contexto_educacional,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        c.id_municipio IS NOT NULL
      ),
      COUNT(*)
    ),
    4
  ) AS
    percentual_cobertura

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

  g.ano,

  g.rede_nome

ORDER BY

  g.ano,

  g.rede_nome;


-- ============================================================
-- QUERY 17
-- Nulls das features derivadas.
-- ============================================================

SELECT

  ano_alvo,

  COUNT(*) AS
    total_linhas,

  COUNTIF(
    proporcao_escolas_rurais IS NULL
  ) AS
    null_proporcao_escolas_rurais,

  COUNTIF(
    proporcao_internet_aprendizagem IS NULL
  ) AS
    null_internet_aprendizagem,

  COUNTIF(
    proporcao_biblioteca_sala_leitura IS NULL
  ) AS
    null_biblioteca_sala_leitura,

  COUNTIF(
    proporcao_laboratorio_informatica IS NULL
  ) AS
    null_laboratorio_informatica,

  COUNTIF(
    alunos_por_turma_anos_iniciais IS NULL
  ) AS
    null_alunos_por_turma,

  COUNTIF(
    razao_matriculas_docentes_anos_iniciais IS NULL
  ) AS
    null_razao_matriculas_docentes,

  COUNTIF(
    proporcao_matriculas_integral_anos_iniciais IS NULL
  ) AS
    null_proporcao_integral

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`

GROUP BY
  ano_alvo

ORDER BY
  ano_alvo;


-- ============================================================
-- QUERY 18
-- Distribuição das principais features.
--
-- Detecta valores extremos antes de incorporá-las ao modelo.
-- ============================================================

SELECT

  ano_alvo,

  APPROX_QUANTILES(
    quantidade_escolas_anos_iniciais,
    4
  ) AS
    quartis_quantidade_escolas,

  APPROX_QUANTILES(
    total_matriculas_anos_iniciais,
    4
  ) AS
    quartis_matriculas,

  APPROX_QUANTILES(
    alunos_por_turma_anos_iniciais,
    4
  ) AS
    quartis_alunos_por_turma,

  APPROX_QUANTILES(
    razao_matriculas_docentes_anos_iniciais,
    4
  ) AS
    quartis_matriculas_docentes,

  APPROX_QUANTILES(
    proporcao_escolas_rurais,
    4
  ) AS
    quartis_proporcao_rural,

  APPROX_QUANTILES(
    proporcao_internet_aprendizagem,
    4
  ) AS
    quartis_internet,

  APPROX_QUANTILES(
    proporcao_biblioteca_sala_leitura,
    4
  ) AS
    quartis_biblioteca,

  APPROX_QUANTILES(
    proporcao_laboratorio_informatica,
    4
  ) AS
    quartis_laboratorio,

  APPROX_QUANTILES(
    proporcao_matriculas_integral_anos_iniciais,
    4
  ) AS
    quartis_integral

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.contexto_educacional_municipio_rede`

GROUP BY
  ano_alvo

ORDER BY
  ano_alvo;