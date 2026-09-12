-- ============================================================
-- Tech Challenge Fase 3
-- Criação da base analítica para Machine Learning
--
-- Tabela destino:
-- alfabetizacao_gold.gold_ml_aluno
--
-- Granularidade:
-- 1 registro por aluno elegível por ano
--
-- População elegível:
-- - aluno presente
-- - prova preenchida
-- - quality_status = VALID
--
-- Cuidados com leakage:
-- - proficiência individual não é carregada
-- - resultados municipais do mesmo ano não são carregados
-- - metas não são utilizadas como features do aluno
--
-- Contexto socioeconômico:
-- referência fixa em 2020
-- ============================================================


CREATE OR REPLACE TABLE
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

PARTITION BY
  RANGE_BUCKET(
    ano,
    GENERATE_ARRAY(2023, 2031, 1)
  )

CLUSTER BY
  id_municipio,
  sigla_uf,
  rede_nome

OPTIONS (
  description = '''
  Base analítica em granularidade aluno-ano criada para o
  Tech Challenge Fase 3.

  Contém apenas alunos presentes, com prova preenchida e
  registros VALID da camada Silver.

  A proficiência individual e indicadores educacionais
  agregados do mesmo ano foram excluídos das features por
  risco de data leakage.

  O contexto territorial e socioeconômico utiliza
  informações municipais anteriores à avaliação,
  com referência fixa em 2020.
  '''
)

AS


-- ============================================================
-- 1. Diretório municipal
-- ============================================================

WITH diretorio AS (

  SELECT
    id_municipio,

    ANY_VALUE(nome) AS municipio,
    ANY_VALUE(sigla_uf) AS sigla_uf,
    ANY_VALUE(nome_uf) AS nome_uf,
    ANY_VALUE(nome_regiao) AS regiao,
    ANY_VALUE(capital_uf) AS capital_uf,
    ANY_VALUE(amazonia_legal) AS amazonia_legal

  FROM
    `basedosdados.br_bd_diretorios_brasil.municipio`

  GROUP BY
    id_municipio
),


-- ============================================================
-- 2. População municipal
-- Referência fixa: 2020
-- ============================================================

populacao AS (

  SELECT
    id_municipio,
    populacao

  FROM
    `basedosdados.br_ibge_populacao.municipio`

  WHERE
    ano = 2020
),


-- ============================================================
-- 3. PIB municipal
-- Referência fixa: 2020
-- ============================================================

pib AS (

  SELECT
    id_municipio,

    pib,
    va,
    va_agropecuaria,
    va_industria,
    va_servicos,
    va_adespss

  FROM
    `basedosdados.br_ibge_pib.municipio`

  WHERE
    ano = 2020
),


-- ============================================================
-- 4. Contexto territorial e socioeconômico
-- ============================================================

contexto_municipal AS (

  SELECT
    d.id_municipio,

    -- Território
    d.municipio,
    d.sigla_uf,
    d.nome_uf,
    d.regiao,
    d.capital_uf,
    d.amazonia_legal,

    -- População
    pop.populacao AS populacao_2020,

    -- PIB
    pib.pib AS pib_2020,

    -- PIB por habitante
    SAFE_DIVIDE(
      CAST(pib.pib AS FLOAT64),
      CAST(pop.populacao AS FLOAT64)
    ) AS pib_por_habitante_2020,

    -- Estrutura econômica municipal
    SAFE_DIVIDE(
      CAST(pib.va_agropecuaria AS FLOAT64),
      CAST(pib.va AS FLOAT64)
    ) AS participacao_agropecuaria_2020,

    SAFE_DIVIDE(
      CAST(pib.va_industria AS FLOAT64),
      CAST(pib.va AS FLOAT64)
    ) AS participacao_industria_2020,

    SAFE_DIVIDE(
      CAST(pib.va_servicos AS FLOAT64),
      CAST(pib.va AS FLOAT64)
    ) AS participacao_servicos_2020,

    SAFE_DIVIDE(
      CAST(pib.va_adespss AS FLOAT64),
      CAST(pib.va AS FLOAT64)
    ) AS participacao_administracao_publica_2020

  FROM
    diretorio d

  LEFT JOIN
    populacao pop
    USING (id_municipio)

  LEFT JOIN
    pib
    USING (id_municipio)
)


-- ============================================================
-- 5. Gold ML
-- ============================================================

SELECT

  -- ----------------------------------------------------------
  -- Controle temporal
  -- ----------------------------------------------------------

  a.ano,
  a.ano_referencia,


  -- ----------------------------------------------------------
  -- Identificadores para rastreabilidade
  --
  -- Estes campos NÃO devem ser utilizados diretamente
  -- como features do modelo.
  -- ----------------------------------------------------------

  TO_HEX(
    SHA256(
      CONCAT(
        CAST(a.ano AS STRING),
        '|',
        a.id_aluno
      )
    )
  ) AS aluno_key,

  a.id_escola AS id_escola_mascarado,

  a.id_municipio,


  -- ----------------------------------------------------------
  -- Target
  --
  -- 0 = não alfabetizado
  -- 1 = alfabetizado
  -- ----------------------------------------------------------

  SAFE_CAST(
    a.alfabetizado_codigo AS INT64
  ) AS alfabetizado,


  -- ----------------------------------------------------------
  -- Contexto educacional
  -- ----------------------------------------------------------

  a.rede_codigo,
  a.rede_nome,
  a.rede_agrupada,


  -- ----------------------------------------------------------
  -- Contexto territorial
  -- ----------------------------------------------------------

  c.municipio,
  c.sigla_uf,
  c.nome_uf,
  c.regiao,
  c.capital_uf,
  c.amazonia_legal,


  -- ----------------------------------------------------------
  -- Contexto socioeconômico
  -- ----------------------------------------------------------

  2020 AS ano_contexto_socioeconomico,

  c.populacao_2020,
  c.pib_2020,
  c.pib_por_habitante_2020,

  c.participacao_agropecuaria_2020,
  c.participacao_industria_2020,
  c.participacao_servicos_2020,
  c.participacao_administracao_publica_2020,


  -- ----------------------------------------------------------
  -- Governança e rastreabilidade
  -- ----------------------------------------------------------

  a.source_batch_id,

  a.quality_status AS source_quality_status,

  CURRENT_TIMESTAMP() AS gold_processed_at


FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos` a

LEFT JOIN
  contexto_municipal c
  USING (id_municipio)

WHERE
  a.presenca_codigo = '1'

  AND a.preenchimento_caderno_codigo = '1'

  AND a.quality_status = 'VALID';