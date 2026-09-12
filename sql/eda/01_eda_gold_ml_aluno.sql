-- ============================================================
-- Tech Challenge Fase 3
-- Análise Exploratória de Dados
--
-- Base:
-- alfabetizacao_gold.gold_ml_aluno
--
-- Objetivos:
-- - compreender a distribuição do target;
-- - analisar variáveis categóricas;
-- - analisar contexto territorial;
-- - avaliar distribuições socioeconômicas;
-- - investigar associações com alfabetização;
-- - apoiar decisões posteriores de modelagem.
-- ============================================================


-- ============================================================
-- QUERY 28
-- Cardinalidade das variáveis categóricas
--
-- Objetivo:
-- identificar categorias constantes, redundantes ou
-- potencialmente problemáticas antes da modelagem.
-- ============================================================

SELECT
  COUNT(DISTINCT ano) AS n_anos,
  COUNT(DISTINCT rede_codigo) AS n_rede_codigo,
  COUNT(DISTINCT rede_nome) AS n_rede_nome,
  COUNT(DISTINCT rede_agrupada) AS n_rede_agrupada,
  COUNT(DISTINCT sigla_uf) AS n_uf,
  COUNT(DISTINCT regiao) AS n_regioes,
  COUNT(DISTINCT capital_uf) AS n_capital_uf,
  COUNT(DISTINCT amazonia_legal) AS n_amazonia_legal,
  COUNT(DISTINCT ano_contexto_socioeconomico)
    AS n_ano_contexto

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`;



-- ============================================================
-- QUERY 29
-- Distribuição das categorias de rede
--
-- Objetivo:
-- entender relação e possível redundância entre
-- rede_codigo, rede_nome e rede_agrupada.
-- ============================================================

SELECT
  ano,
  rede_codigo,
  rede_nome,
  rede_agrupada,

  COUNT(*) AS quantidade,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNT(*),
      SUM(COUNT(*)) OVER (
        PARTITION BY ano
      )
    ),
    2
  ) AS percentual

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  rede_codigo,
  rede_nome,
  rede_agrupada

ORDER BY
  ano,
  quantidade DESC;



-- ============================================================
-- QUERY 30
-- Distribuição do target por ano
--
-- Objetivo:
-- confirmar o balanceamento das classes.
-- ============================================================

SELECT
  ano,
  alfabetizado,

  COUNT(*) AS quantidade,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNT(*),
      SUM(COUNT(*)) OVER (
        PARTITION BY ano
      )
    ),
    2
  ) AS percentual

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  alfabetizado

ORDER BY
  ano,
  alfabetizado;



-- ============================================================
-- QUERY 31
-- Taxa de alfabetização por região e ano
--
-- Observação:
-- esta é uma associação descritiva e não deve ser
-- interpretada como causalidade.
-- ============================================================

SELECT
  ano,
  regiao,

  COUNT(*) AS quantidade_alunos,

  SUM(alfabetizado)
    AS quantidade_alfabetizados,

  ROUND(
    100 * AVG(alfabetizado),
    2
  ) AS taxa_alfabetizacao

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  regiao

ORDER BY
  ano,
  taxa_alfabetizacao DESC;



-- ============================================================
-- QUERY 32
-- Taxa de alfabetização por rede e ano
-- ============================================================

SELECT
  ano,
  rede_nome,

  COUNT(*) AS quantidade_alunos,

  SUM(alfabetizado)
    AS quantidade_alfabetizados,

  ROUND(
    100 * AVG(alfabetizado),
    2
  ) AS taxa_alfabetizacao

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  rede_nome

ORDER BY
  ano,
  taxa_alfabetizacao DESC;



-- ============================================================
-- QUERY 33
-- Região x rede
--
-- Objetivo:
-- verificar se diferenças entre redes permanecem
-- semelhantes em diferentes regiões.
-- ============================================================

SELECT
  ano,
  regiao,
  rede_nome,

  COUNT(*) AS quantidade_alunos,

  ROUND(
    100 * AVG(alfabetizado),
    2
  ) AS taxa_alfabetizacao

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  regiao,
  rede_nome

ORDER BY
  ano,
  regiao,
  taxa_alfabetizacao DESC;



-- ============================================================
-- QUERY 34
-- Distribuição das variáveis socioeconômicas
--
-- IMPORTANTE:
-- usamos uma linha por município para não repetir milhares
-- de vezes o mesmo contexto municipal na descrição estatística.
-- ============================================================

WITH municipios AS (

  SELECT DISTINCT
    id_municipio,

    populacao_2020,
    pib_2020,
    pib_por_habitante_2020,

    participacao_agropecuaria_2020,
    participacao_industria_2020,
    participacao_servicos_2020,
    participacao_administracao_publica_2020

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`
)

SELECT
  COUNT(*) AS quantidade_municipios,

  -- População
  MIN(populacao_2020)
    AS populacao_min,

  APPROX_QUANTILES(
    populacao_2020,
    100
  )[OFFSET(25)] AS populacao_q1,

  APPROX_QUANTILES(
    populacao_2020,
    100
  )[OFFSET(50)] AS populacao_mediana,

  APPROX_QUANTILES(
    populacao_2020,
    100
  )[OFFSET(75)] AS populacao_q3,

  MAX(populacao_2020)
    AS populacao_max,

  -- PIB por habitante
  MIN(pib_por_habitante_2020)
    AS pib_per_capita_min,

  APPROX_QUANTILES(
    pib_por_habitante_2020,
    100
  )[OFFSET(25)] AS pib_per_capita_q1,

  APPROX_QUANTILES(
    pib_por_habitante_2020,
    100
  )[OFFSET(50)] AS pib_per_capita_mediana,

  APPROX_QUANTILES(
    pib_por_habitante_2020,
    100
  )[OFFSET(75)] AS pib_per_capita_q3,

  MAX(pib_por_habitante_2020)
    AS pib_per_capita_max,

  -- Participações econômicas
  AVG(participacao_agropecuaria_2020)
    AS media_participacao_agro,

  AVG(participacao_industria_2020)
    AS media_participacao_industria,

  AVG(participacao_servicos_2020)
    AS media_participacao_servicos,

  AVG(participacao_administracao_publica_2020)
    AS media_participacao_administracao

FROM
  municipios;



-- ============================================================
-- QUERY 35
-- Distribuição municipal da alfabetização
--
-- Uma linha por município e ano.
--
-- Objetivo:
-- entender a heterogeneidade territorial sem deixar
-- municípios maiores dominarem a estatística descritiva.
-- ============================================================

WITH resultado_municipio AS (

  SELECT
    ano,
    id_municipio,

    COUNT(*) AS quantidade_alunos,

    AVG(alfabetizado)
      AS taxa_alfabetizacao

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    ano,
    id_municipio
)

SELECT
  ano,

  COUNT(*) AS quantidade_municipios,

  MIN(taxa_alfabetizacao)
    AS taxa_min,

  APPROX_QUANTILES(
    taxa_alfabetizacao,
    100
  )[OFFSET(25)] AS taxa_q1,

  APPROX_QUANTILES(
    taxa_alfabetizacao,
    100
  )[OFFSET(50)] AS taxa_mediana,

  APPROX_QUANTILES(
    taxa_alfabetizacao,
    100
  )[OFFSET(75)] AS taxa_q3,

  MAX(taxa_alfabetizacao)
    AS taxa_max,

  AVG(taxa_alfabetizacao)
    AS media_municipal_nao_ponderada

FROM
  resultado_municipio

GROUP BY
  ano

ORDER BY
  ano;



-- ============================================================
-- QUERY 36
-- Correlações no nível municipal
--
-- IMPORTANTE:
-- correlação representa associação estatística,
-- não relação causal.
--
-- A taxa de alfabetização é agregada por município.
-- O contexto socioeconômico é municipal.
-- ============================================================

WITH resultado_municipio AS (

  SELECT
    ano,
    id_municipio,

    AVG(alfabetizado)
      AS taxa_alfabetizacao,

    ANY_VALUE(populacao_2020)
      AS populacao_2020,

    ANY_VALUE(pib_por_habitante_2020)
      AS pib_por_habitante_2020,

    ANY_VALUE(participacao_agropecuaria_2020)
      AS participacao_agropecuaria_2020,

    ANY_VALUE(participacao_industria_2020)
      AS participacao_industria_2020,

    ANY_VALUE(participacao_servicos_2020)
      AS participacao_servicos_2020,

    ANY_VALUE(
      participacao_administracao_publica_2020
    ) AS participacao_administracao_publica_2020

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    ano,
    id_municipio
)

SELECT
  ano,

  CORR(
    taxa_alfabetizacao,
    populacao_2020
  ) AS corr_populacao,

  CORR(
    taxa_alfabetizacao,
    pib_por_habitante_2020
  ) AS corr_pib_por_habitante,

  CORR(
    taxa_alfabetizacao,
    participacao_agropecuaria_2020
  ) AS corr_agropecuaria,

  CORR(
    taxa_alfabetizacao,
    participacao_industria_2020
  ) AS corr_industria,

  CORR(
    taxa_alfabetizacao,
    participacao_servicos_2020
  ) AS corr_servicos,

  CORR(
    taxa_alfabetizacao,
    participacao_administracao_publica_2020
  ) AS corr_administracao_publica

FROM
  resultado_municipio

GROUP BY
  ano

ORDER BY
  ano;



-- ============================================================
-- QUERY 37
-- Taxa por capital x não capital
-- ============================================================

SELECT
  ano,
  capital_uf,

  COUNT(*) AS quantidade_alunos,

  ROUND(
    100 * AVG(alfabetizado),
    2
  ) AS taxa_alfabetizacao

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  capital_uf

ORDER BY
  ano,
  capital_uf;



-- ============================================================
-- QUERY 38
-- Taxa por Amazônia Legal
-- ============================================================

SELECT
  ano,
  amazonia_legal,

  COUNT(*) AS quantidade_alunos,

  ROUND(
    100 * AVG(alfabetizado),
    2
  ) AS taxa_alfabetizacao

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

GROUP BY
  ano,
  amazonia_legal

ORDER BY
  ano,
  amazonia_legal;



-- ============================================================
-- QUERY 39
-- Municípios com menor taxa de alfabetização
--
-- Filtro mínimo de alunos reduz interpretações baseadas
-- em taxas muito instáveis de municípios com poucos registros.
-- ============================================================

WITH resultado_municipio AS (

  SELECT
    ano,
    id_municipio,

    ANY_VALUE(municipio)
      AS municipio,

    ANY_VALUE(sigla_uf)
      AS sigla_uf,

    ANY_VALUE(regiao)
      AS regiao,

    COUNT(*) AS quantidade_alunos,

    AVG(alfabetizado)
      AS taxa_alfabetizacao

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    ano,
    id_municipio
)

SELECT
  ano,
  id_municipio,
  municipio,
  sigla_uf,
  regiao,
  quantidade_alunos,

  ROUND(
    100 * taxa_alfabetizacao,
    2
  ) AS taxa_alfabetizacao

FROM
  resultado_municipio

WHERE
  quantidade_alunos >= 100

QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY ano
    ORDER BY taxa_alfabetizacao ASC
  ) <= 20

ORDER BY
  ano,
  taxa_alfabetizacao;



-- ============================================================
-- QUERY 40
-- Municípios com maior taxa de alfabetização
-- Mesma regra mínima de 100 alunos.
-- ============================================================

WITH resultado_municipio AS (

  SELECT
    ano,
    id_municipio,

    ANY_VALUE(municipio)
      AS municipio,

    ANY_VALUE(sigla_uf)
      AS sigla_uf,

    ANY_VALUE(regiao)
      AS regiao,

    COUNT(*) AS quantidade_alunos,

    AVG(alfabetizado)
      AS taxa_alfabetizacao

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    ano,
    id_municipio
)

SELECT
  ano,
  id_municipio,
  municipio,
  sigla_uf,
  regiao,
  quantidade_alunos,

  ROUND(
    100 * taxa_alfabetizacao,
    2
  ) AS taxa_alfabetizacao

FROM
  resultado_municipio

WHERE
  quantidade_alunos >= 100

QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY ano
    ORDER BY taxa_alfabetizacao DESC
  ) <= 20

ORDER BY
  ano,
  taxa_alfabetizacao DESC;

  -- ============================================================
-- QUERY 41
-- Distribuição da série na população elegível
--
-- Objetivo:
-- confirmar se série é constante e se sua exclusão
-- da Gold de ML foi adequada.
-- ============================================================

SELECT
  ano,
  serie_codigo,
  serie_nome,
  COUNT(*) AS quantidade,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNT(*),
      SUM(COUNT(*)) OVER (
        PARTITION BY ano
      )
    ),
    4
  ) AS percentual

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

WHERE
  presenca_codigo = '1'
  AND preenchimento_caderno_codigo = '1'
  AND quality_status = 'VALID'

GROUP BY
  ano,
  serie_codigo,
  serie_nome

ORDER BY
  ano,
  quantidade DESC;

  -- ============================================================
-- QUERY 42
-- Cobertura de UFs na Gold
--
-- Objetivo:
-- identificar qual UF não está representada.
-- ============================================================

WITH ufs_referencia AS (

  SELECT DISTINCT
    sigla_uf

  FROM
    `basedosdados.br_bd_diretorios_brasil.municipio`

  WHERE
    sigla_uf IS NOT NULL
),

ufs_gold AS (

  SELECT
    sigla_uf,

    COUNTIF(ano = 2023) AS registros_2023,
    COUNTIF(ano = 2024) AS registros_2024

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    sigla_uf
)

SELECT
  r.sigla_uf,

  COALESCE(g.registros_2023, 0)
    AS registros_2023,

  COALESCE(g.registros_2024, 0)
    AS registros_2024,

  CASE
    WHEN g.sigla_uf IS NULL
      THEN 'AUSENTE'
    ELSE 'PRESENTE'
  END AS status_gold

FROM
  ufs_referencia r

LEFT JOIN
  ufs_gold g
  USING (sigla_uf)

ORDER BY
  r.sigla_uf;

-- ============================================================
-- QUERY 43
-- Cobertura municipal entre 2023 e 2024
--
-- Objetivo:
-- quantificar municípios presentes em ambos os anos
-- e municípios exclusivos de cada avaliação.
-- ============================================================

WITH cobertura AS (

  SELECT
    id_municipio,

    COUNTIF(ano = 2023) > 0
      AS presente_2023,

    COUNTIF(ano = 2024) > 0
      AS presente_2024

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    id_municipio
)

SELECT
  COUNT(*) AS municipios_total,

  COUNTIF(
    presente_2023
    AND presente_2024
  ) AS municipios_nos_dois_anos,

  COUNTIF(
    presente_2023
    AND NOT presente_2024
  ) AS somente_2023,

  COUNTIF(
    NOT presente_2023
    AND presente_2024
  ) AS somente_2024

FROM
  cobertura;

-- ============================================================
-- QUERY 44
-- Registros da rede privada
--
-- Objetivo:
-- compreender a categoria que aparece somente em 2024
-- antes de decidir seu tratamento na modelagem.
-- ============================================================

SELECT
  ano,
  id_municipio,
  municipio,
  sigla_uf,
  regiao,
  id_escola_mascarado,

  COUNT(*) AS quantidade_alunos,

  SUM(alfabetizado)
    AS alfabetizados,

  ROUND(
    100 * AVG(alfabetizado),
    2
  ) AS taxa_alfabetizacao

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

WHERE
  rede_nome = 'Privada'

GROUP BY
  ano,
  id_municipio,
  municipio,
  sigla_uf,
  regiao,
  id_escola_mascarado

ORDER BY
  quantidade_alunos DESC;

-- ============================================================
-- QUERY 45
-- Estabilidade das taxas municipais 2023 x 2024
--
-- Objetivo:
-- medir quanto o resultado municipal de um ano se associa
-- ao resultado do ano seguinte.
--
-- Essa análise é apenas exploratória.
-- ============================================================

WITH taxas AS (

  SELECT
    ano,
    id_municipio,

    COUNT(*) AS quantidade_alunos,

    AVG(alfabetizado)
      AS taxa_alfabetizacao

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

  GROUP BY
    ano,
    id_municipio
),

comparacao AS (

  SELECT
    a.id_municipio,

    a.quantidade_alunos
      AS alunos_2023,

    b.quantidade_alunos
      AS alunos_2024,

    a.taxa_alfabetizacao
      AS taxa_2023,

    b.taxa_alfabetizacao
      AS taxa_2024,

    b.taxa_alfabetizacao
      - a.taxa_alfabetizacao
      AS variacao

  FROM
    taxas a

  INNER JOIN
    taxas b

  ON
    a.id_municipio = b.id_municipio

  WHERE
    a.ano = 2023
    AND b.ano = 2024
)

SELECT
  COUNT(*) AS municipios_comparaveis,

  CORR(
    taxa_2023,
    taxa_2024
  ) AS correlacao_taxas_entre_anos,

  AVG(
    ABS(variacao)
  ) AS media_variacao_absoluta,

  APPROX_QUANTILES(
    ABS(variacao),
    100
  )[OFFSET(50)]
    AS mediana_variacao_absoluta,

  APPROX_QUANTILES(
    variacao,
    100
  )[OFFSET(25)]
    AS variacao_q1,

  APPROX_QUANTILES(
    variacao,
    100
  )[OFFSET(50)]
    AS variacao_mediana,

  APPROX_QUANTILES(
    variacao,
    100
  )[OFFSET(75)]
    AS variacao_q3

FROM
  comparacao;

  