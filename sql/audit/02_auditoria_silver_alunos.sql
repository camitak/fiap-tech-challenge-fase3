-- ============================================================
-- Tech Challenge Fase 3
-- Auditoria da tabela Silver de alunos
--
-- Tabela:
-- alfabetizacao_silver.alunos
--
-- Objetivos:
-- - compreender schema e volume;
-- - definir população elegível;
-- - validar target;
-- - investigar leakage;
-- - verificar duplicidades;
-- - verificar missing;
-- - investigar qualidade;
-- - investigar semântica temporal dos identificadores.
-- ============================================================


-- ============================================================
-- QUERY 4
-- Schema da Silver alunos
-- ============================================================

SELECT
  ordinal_position,
  column_name,
  data_type,
  is_nullable

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.INFORMATION_SCHEMA.COLUMNS`

WHERE
  table_name = 'alunos'

ORDER BY
  ordinal_position;



-- ============================================================
-- QUERY 5
-- Volume por ano
-- ============================================================

SELECT
  ano,
  COUNT(*) AS quantidade_linhas

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

GROUP BY
  ano

ORDER BY
  ano;



-- ============================================================
-- QUERY 6
-- Relação entre presença, preenchimento e target
-- ============================================================

SELECT
  ano,

  presenca_codigo,
  presenca_nome,

  preenchimento_caderno_codigo,
  preenchimento_caderno_nome,

  alfabetizado_codigo,
  alfabetizado_nome,

  COUNT(*) AS quantidade

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

GROUP BY
  ano,
  presenca_codigo,
  presenca_nome,
  preenchimento_caderno_codigo,
  preenchimento_caderno_nome,
  alfabetizado_codigo,
  alfabetizado_nome

ORDER BY
  ano,
  presenca_codigo,
  preenchimento_caderno_codigo,
  alfabetizado_codigo;



-- ============================================================
-- QUERY 7
-- Validação do target contra o corte de 743 pontos
--
-- Objetivo:
-- verificar se proficiência reproduz diretamente o target.
-- ============================================================

SELECT
  ano,

  COUNTIF(
    proficiencia IS NOT NULL
  ) AS registros_com_proficiencia,

  COUNTIF(
    proficiencia IS NOT NULL
    AND alfabetizado_codigo = '1'
  ) AS alfabetizados,

  COUNTIF(
    proficiencia >= 743
  ) AS acima_ou_igual_743,

  COUNTIF(
    proficiencia IS NOT NULL
    AND SAFE_CAST(alfabetizado_codigo AS INT64)
      != IF(proficiencia >= 743, 1, 0)
  ) AS divergencias_regra_743,

  ROUND(
    100 * SAFE_DIVIDE(

      COUNTIF(
        proficiencia IS NOT NULL
        AND SAFE_CAST(alfabetizado_codigo AS INT64)
          != IF(proficiencia >= 743, 1, 0)
      ),

      COUNTIF(
        proficiencia IS NOT NULL
      )

    ),
    4
  ) AS percentual_divergencia

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

GROUP BY
  ano

ORDER BY
  ano;



-- ============================================================
-- QUERY 8
-- Duplicidade de id_aluno dentro do mesmo ano
-- ============================================================

WITH contagem AS (

  SELECT
    ano,
    id_aluno,
    COUNT(*) AS quantidade

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

  GROUP BY
    ano,
    id_aluno
)

SELECT
  ano,

  COUNT(*) AS alunos_distintos,

  COUNTIF(
    quantidade > 1
  ) AS alunos_com_duplicidade,

  MAX(
    quantidade
  ) AS max_registros_mesmo_aluno

FROM
  contagem

GROUP BY
  ano

ORDER BY
  ano;



-- ============================================================
-- QUERY 9
-- Quantidade de valores de id_aluno presentes em 2023 e 2024
-- ============================================================

SELECT
  COUNT(*) AS alunos_presentes_nos_dois_anos

FROM (

  SELECT
    id_aluno

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

  WHERE
    ano IN (2023, 2024)

  GROUP BY
    id_aluno

  HAVING
    COUNT(DISTINCT ano) = 2
);



-- ============================================================
-- QUERY 10
-- Missing values das principais colunas
-- ============================================================

SELECT
  ano,

  COUNT(*) AS total,

  COUNTIF(id_aluno IS NULL)
    AS nulos_id_aluno,

  COUNTIF(id_escola IS NULL)
    AS nulos_id_escola,

  COUNTIF(id_municipio IS NULL)
    AS nulos_id_municipio,

  COUNTIF(rede_codigo IS NULL)
    AS nulos_rede,

  COUNTIF(presenca_codigo IS NULL)
    AS nulos_presenca,

  COUNTIF(preenchimento_caderno_codigo IS NULL)
    AS nulos_preenchimento,

  COUNTIF(alfabetizado_codigo IS NULL)
    AS nulos_alfabetizado,

  COUNTIF(proficiencia IS NULL)
    AS nulos_proficiencia,

  COUNTIF(peso_aluno IS NULL)
    AS nulos_peso_aluno

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

GROUP BY
  ano

ORDER BY
  ano;



-- ============================================================
-- QUERY 11
-- Distribuição de quality_status
-- ============================================================

SELECT
  ano,
  quality_status,

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
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

GROUP BY
  ano,
  quality_status

ORDER BY
  ano,
  quality_status;



-- ============================================================
-- QUERY 12
-- Confirmação da população final elegível
-- ============================================================

SELECT
  ano,

  COUNT(*) AS total_elegiveis,

  COUNTIF(
    alfabetizado_codigo = '0'
  ) AS nao_alfabetizados,

  COUNTIF(
    alfabetizado_codigo = '1'
  ) AS alfabetizados,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(alfabetizado_codigo = '1'),
      COUNT(*)
    ),
    2
  ) AS percentual_alfabetizados,

  COUNTIF(
    proficiencia IS NULL
  ) AS nulos_proficiencia,

  COUNTIF(
    peso_aluno IS NULL
  ) AS nulos_peso_aluno,

  COUNTIF(
    quality_status != 'VALID'
  ) AS registros_nao_validos

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

WHERE
  presenca_codigo = '1'

  AND preenchimento_caderno_codigo = '1'

GROUP BY
  ano

ORDER BY
  ano;



-- ============================================================
-- QUERY 13
-- Identificação dos alertas de qualidade
-- ============================================================

SELECT
  ano,
  quality_status,
  alerta,

  COUNT(*) AS quantidade

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`,

  UNNEST(
    quality_alerts
  ) AS alerta

GROUP BY
  ano,
  quality_status,
  alerta

ORDER BY
  ano,
  quantidade DESC;



-- ============================================================
-- QUERY 14
-- Consistência de id_aluno entre 2023 e 2024
--
-- Objetivo:
-- investigar se id_aluno pode ser interpretado como
-- identificador longitudinal de pessoa.
-- ============================================================

WITH alunos_2023 AS (

  SELECT
    id_aluno,
    id_municipio,
    id_escola,
    rede_codigo,
    serie_codigo,
    alfabetizado_codigo

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

  WHERE
    ano = 2023
),

alunos_2024 AS (

  SELECT
    id_aluno,
    id_municipio,
    id_escola,
    rede_codigo,
    serie_codigo,
    alfabetizado_codigo

  FROM
    `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos`

  WHERE
    ano = 2024
)

SELECT
  COUNT(*) AS ids_presentes_nos_dois_anos,

  COUNTIF(
    a23.id_municipio = a24.id_municipio
  ) AS mesmo_municipio,

  COUNTIF(
    a23.id_municipio != a24.id_municipio
  ) AS municipio_diferente,

  COUNTIF(
    a23.id_escola = a24.id_escola
  ) AS mesma_escola,

  COUNTIF(
    a23.id_escola != a24.id_escola
  ) AS escola_diferente,

  COUNTIF(
    a23.rede_codigo = a24.rede_codigo
  ) AS mesma_rede,

  COUNTIF(
    a23.alfabetizado_codigo
      = a24.alfabetizado_codigo
  ) AS mesmo_resultado_alfabetizacao

FROM
  alunos_2023 a23

INNER JOIN
  alunos_2024 a24

USING (id_aluno);



-- ============================================================
-- QUERY 15
-- Amostra de identificadores presentes nos dois anos
-- ============================================================

SELECT
  a23.id_aluno,

  a23.id_municipio AS municipio_2023,
  a24.id_municipio AS municipio_2024,

  a23.id_escola AS escola_2023,
  a24.id_escola AS escola_2024,

  a23.rede_nome AS rede_2023,
  a24.rede_nome AS rede_2024,

  a23.alfabetizado_nome
    AS alfabetizado_2023,

  a24.alfabetizado_nome
    AS alfabetizado_2024,

  a23.proficiencia
    AS proficiencia_2023,

  a24.proficiencia
    AS proficiencia_2024

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos` a23

INNER JOIN
  `fiap-tc-f2-camila-takemoto.alfabetizacao_silver.alunos` a24

ON
  a23.id_aluno = a24.id_aluno

WHERE
  a23.ano = 2023

  AND a24.ano = 2024

ORDER BY
  a23.id_aluno

LIMIT 30;


/*
Conclusões principais:

1. A granularidade lógica utilizada será aluno-ano.

2. A população de modelagem contém somente alunos:
   - presentes;
   - com prova preenchida;
   - com registro válido.

3. Proficiência individual reproduz diretamente o target
   e, portanto, é leakage direto.

4. Ausência ou não preenchimento da prova não deve ser
   interpretado automaticamente como evidência de
   não alfabetização.

5. id_aluno não será utilizado como feature nem tratado
   como identificador longitudinal confiável entre anos.

6. id_escola também não é feature. A documentação da
   fonte informa que se trata de um código mascarado/fictício.
*/