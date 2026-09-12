-- ============================================================
-- Tech Challenge Fase 3
-- Auditoria de fontes complementares e enriquecimento
--
-- Objetivos:
-- - inventariar a Silver;
-- - identificar dados já disponíveis;
-- - avaliar fontes territoriais e socioeconômicas;
-- - validar as chaves de integração;
-- - validar cobertura temporal;
-- - validar cobertura dos municípios;
-- - simular a futura Gold de ML.
-- ============================================================


-- ============================================================
-- QUERY 16
-- Inventário da Silver
-- ============================================================

SELECT
  table_name,
  table_type

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.INFORMATION_SCHEMA.TABLES`

ORDER BY
  table_name;



-- ============================================================
-- QUERY 17
-- Schema das tabelas Silver, exceto alunos
-- ============================================================

SELECT
  table_name,
  ordinal_position,
  column_name,
  data_type

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.INFORMATION_SCHEMA.COLUMNS`

WHERE
  table_name != 'alunos'

ORDER BY
  table_name,
  ordinal_position;



-- ============================================================
-- QUERY 18
-- Visão compacta das colunas da Silver
-- ============================================================

SELECT
  table_name,

  STRING_AGG(
    column_name,
    ', '
    ORDER BY ordinal_position
  ) AS colunas

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.INFORMATION_SCHEMA.COLUMNS`

GROUP BY
  table_name

ORDER BY
  table_name;



-- ============================================================
-- QUERY 19
-- Schema das fontes externas candidatas
-- ============================================================

SELECT
  'diretorio_municipio' AS fonte,
  ordinal_position,
  column_name,
  data_type

FROM
  `basedosdados.br_bd_diretorios_brasil.INFORMATION_SCHEMA.COLUMNS`

WHERE
  table_name = 'municipio'


UNION ALL


SELECT
  'populacao_municipio' AS fonte,
  ordinal_position,
  column_name,
  data_type

FROM
  `basedosdados.br_ibge_populacao.INFORMATION_SCHEMA.COLUMNS`

WHERE
  table_name = 'municipio'


UNION ALL


SELECT
  'pib_municipio' AS fonte,
  ordinal_position,
  column_name,
  data_type

FROM
  `basedosdados.br_ibge_pib.INFORMATION_SCHEMA.COLUMNS`

WHERE
  table_name = 'municipio'


ORDER BY
  fonte,
  ordinal_position;



-- ============================================================
-- QUERY 20
-- Validação do diretório municipal e da chave IBGE
-- ============================================================

SELECT
  id_municipio,

  nome AS municipio,

  id_uf,
  sigla_uf,
  nome_regiao

FROM
  `basedosdados.br_bd_diretorios_brasil.municipio`

ORDER BY
  id_municipio

LIMIT 30;



-- ============================================================
-- QUERY 21
-- Cobertura temporal das fontes populacional e econômica
-- ============================================================

SELECT
  'populacao' AS fonte,

  MIN(ano) AS primeiro_ano,
  MAX(ano) AS ultimo_ano,

  COUNT(DISTINCT ano)
    AS quantidade_anos,

  COUNT(DISTINCT id_municipio)
    AS municipios

FROM
  `basedosdados.br_ibge_populacao.municipio`


UNION ALL


SELECT
  'pib' AS fonte,

  MIN(ano) AS primeiro_ano,
  MAX(ano) AS ultimo_ano,

  COUNT(DISTINCT ano)
    AS quantidade_anos,

  COUNT(DISTINCT id_municipio)
    AS municipios

FROM
  `basedosdados.br_ibge_pib.municipio`;



-- ============================================================
-- QUERY 22
-- Teste exploratório de cobertura utilizando 2022
--
-- Observação:
-- esta consulta foi utilizada durante a auditoria.
-- Posteriormente, 2020 foi escolhido como referência final
-- por questões de disponibilidade temporal/publicação.
-- ============================================================

WITH municipios_alunos AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

  WHERE
    presenca_codigo = '1'

    AND preenchimento_caderno_codigo = '1'

    AND quality_status = 'VALID'
),

diretorio AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `basedosdados.br_bd_diretorios_brasil.municipio`
),

populacao_2022 AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `basedosdados.br_ibge_populacao.municipio`

  WHERE
    ano = 2022
),

pib_2022 AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `basedosdados.br_ibge_pib.municipio`

  WHERE
    ano = 2022
)

SELECT
  COUNT(*) AS municipios_alunos,

  COUNTIF(
    d.id_municipio IS NOT NULL
  ) AS municipios_com_diretorio,

  COUNTIF(
    p.id_municipio IS NOT NULL
  ) AS municipios_com_populacao_2022,

  COUNTIF(
    g.id_municipio IS NOT NULL
  ) AS municipios_com_pib_2022,

  COUNTIF(
    d.id_municipio IS NOT NULL
    AND p.id_municipio IS NOT NULL
    AND g.id_municipio IS NOT NULL
  ) AS municipios_completos

FROM
  municipios_alunos a

LEFT JOIN
  diretorio d
  USING (id_municipio)

LEFT JOIN
  populacao_2022 p
  USING (id_municipio)

LEFT JOIN
  pib_2022 g
  USING (id_municipio);



-- ============================================================
-- QUERY 23
-- Cobertura e unicidade das fontes para o ano de 2020
--
-- 2020 é a referência socioeconômica adotada na Gold.
-- ============================================================

WITH municipios_alunos AS (

  SELECT DISTINCT
    id_municipio

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

  WHERE
    presenca_codigo = '1'

    AND preenchimento_caderno_codigo = '1'

    AND quality_status = 'VALID'
),

populacao AS (

  SELECT
    id_municipio,

    COUNT(*) AS quantidade_registros,

    ANY_VALUE(populacao)
      AS populacao

  FROM
    `basedosdados.br_ibge_populacao.municipio`

  WHERE
    ano = 2020

  GROUP BY
    id_municipio
),

pib AS (

  SELECT
    id_municipio,

    COUNT(*) AS quantidade_registros,

    ANY_VALUE(pib)
      AS pib,

    ANY_VALUE(va)
      AS va,

    ANY_VALUE(va_agropecuaria)
      AS va_agropecuaria,

    ANY_VALUE(va_industria)
      AS va_industria,

    ANY_VALUE(va_servicos)
      AS va_servicos,

    ANY_VALUE(va_adespss)
      AS va_adespss

  FROM
    `basedosdados.br_ibge_pib.municipio`

  WHERE
    ano = 2020

  GROUP BY
    id_municipio
)

SELECT
  COUNT(*) AS municipios_alunos,

  COUNTIF(
    pop.id_municipio IS NOT NULL
  ) AS municipios_com_populacao_2020,

  COUNTIF(
    pib.id_municipio IS NOT NULL
  ) AS municipios_com_pib_2020,

  COUNTIF(
    pop.id_municipio IS NOT NULL
    AND pib.id_municipio IS NOT NULL
  ) AS municipios_completos,

  COUNTIF(
    pop.quantidade_registros != 1
  ) AS problemas_unicidade_populacao,

  COUNTIF(
    pib.quantidade_registros != 1
  ) AS problemas_unicidade_pib,

  COUNTIF(
    pop.populacao IS NULL
  ) AS populacao_nula,

  COUNTIF(
    pop.populacao <= 0
  ) AS populacao_invalida,

  COUNTIF(
    pib.pib IS NULL
  ) AS pib_nulo,

  COUNTIF(
    pib.pib < 0
  ) AS pib_invalido,

  COUNTIF(
    pib.va IS NULL
    OR pib.va <= 0
  ) AS va_invalido

FROM
  municipios_alunos a

LEFT JOIN
  populacao pop
  USING (id_municipio)

LEFT JOIN
  pib
  USING (id_municipio);



-- ============================================================
-- QUERY 24
-- Simulação da futura gold_ml_aluno
--
-- Objetivo:
-- verificar volume e cobertura antes da materialização.
-- ============================================================

WITH contexto_municipal AS (

  SELECT
    d.id_municipio,

    d.nome AS municipio,

    d.sigla_uf,

    d.nome_regiao AS regiao,

    d.capital_uf,

    d.amazonia_legal,

    pop.populacao
      AS populacao_2020,

    pib.pib
      AS pib_2020,

    SAFE_DIVIDE(
      CAST(pib.pib AS FLOAT64),
      CAST(pop.populacao AS FLOAT64)
    ) AS pib_per_capita_2020,

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
    `basedosdados.br_bd_diretorios_brasil.municipio` d

  LEFT JOIN
    `basedosdados.br_ibge_populacao.municipio` pop

  ON
    d.id_municipio = pop.id_municipio

    AND pop.ano = 2020

  LEFT JOIN
    `basedosdados.br_ibge_pib.municipio` pib

  ON
    d.id_municipio = pib.id_municipio

    AND pib.ano = 2020
),

base_ml AS (

  SELECT
    a.ano,

    a.id_aluno,
    a.id_escola,
    a.id_municipio,

    a.rede_nome,
    a.rede_agrupada,

    SAFE_CAST(
      a.alfabetizado_codigo AS INT64
    ) AS alfabetizado,

    c.municipio,
    c.sigla_uf,
    c.regiao,
    c.capital_uf,
    c.amazonia_legal,

    c.populacao_2020,
    c.pib_2020,
    c.pib_per_capita_2020,

    c.participacao_agropecuaria_2020,
    c.participacao_industria_2020,
    c.participacao_servicos_2020,
    c.participacao_administracao_publica_2020

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos` a

  LEFT JOIN
    contexto_municipal c
    USING (id_municipio)

  WHERE
    a.presenca_codigo = '1'

    AND a.preenchimento_caderno_codigo = '1'

    AND a.quality_status = 'VALID'
)

SELECT
  ano,

  COUNT(*) AS total_linhas,

  COUNT(DISTINCT id_aluno)
    AS alunos_distintos,

  COUNTIF(
    alfabetizado = 0
  ) AS nao_alfabetizados,

  COUNTIF(
    alfabetizado = 1
  ) AS alfabetizados,

  COUNTIF(
    municipio IS NULL
  ) AS sem_contexto_territorial,

  COUNTIF(
    populacao_2020 IS NULL
  ) AS sem_populacao,

  COUNTIF(
    pib_per_capita_2020 IS NULL
  ) AS sem_pib_per_capita,

  COUNTIF(
    participacao_agropecuaria_2020 IS NULL
  ) AS sem_participacao_agro,

  COUNTIF(
    participacao_industria_2020 IS NULL
  ) AS sem_participacao_industria,

  COUNTIF(
    participacao_servicos_2020 IS NULL
  ) AS sem_participacao_servicos

FROM
  base_ml

GROUP BY
  ano

ORDER BY
  ano;


/*
Resultados finais da auditoria de cobertura:

Municípios presentes na população de modelagem:
5.547

Municípios com população 2020:
5.547

Municípios com PIB 2020:
5.547

Municípios completos:
5.547

Problemas de unicidade:
0

Missing nas fontes utilizadas:
0

A simulação preservou:

2023:
1.502.809 registros

2024:
1.851.852 registros
*/