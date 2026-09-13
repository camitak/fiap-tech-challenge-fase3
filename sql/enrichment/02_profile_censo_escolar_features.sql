-- ============================================================
-- Tech Challenge Fase 3
-- Perfil das features educacionais candidatas
--
-- Fonte:
-- basedosdados.br_inep_censo_escolar.escola
--
-- Objetivo:
-- - decodificar categorias relevantes;
-- - analisar situação de funcionamento;
-- - medir cobertura e valores faltantes;
-- - validar possíveis indicadores educacionais;
--
-- Importante:
-- nenhuma Gold é alterada nesta etapa.
--
-- Estratégia temporal pretendida:
-- target 2023 -> Censo Escolar 2022
-- target 2024 -> Censo Escolar 2023
-- ============================================================


-- ============================================================
-- QUERY 10
-- Schema da tabela de dicionário.
--
-- Precisamos confirmar os nomes das colunas antes de
-- utilizá-la diretamente nas próximas transformações.
-- ============================================================

SELECT
  ordinal_position,
  column_name,
  data_type
FROM
  `basedosdados.br_inep_censo_escolar.INFORMATION_SCHEMA.COLUMNS`
WHERE
  table_name = 'dicionario'
ORDER BY
  ordinal_position;


-- ============================================================
-- QUERY 11
-- Registros do dicionário relacionados às categorias
-- que serão necessárias no enriquecimento.
--
-- TO_JSON_STRING permite realizar a descoberta sem assumir
-- previamente o schema interno do dicionário.
-- ============================================================

SELECT
  TO_JSON_STRING(d) AS registro
FROM
  `basedosdados.br_inep_censo_escolar.dicionario` AS d
WHERE
  REGEXP_CONTAINS(
    LOWER(TO_JSON_STRING(d)),
    r'rede|tipo_localizacao|tipo_situacao_funcionamento'
  )
ORDER BY
  registro;


-- ============================================================
-- QUERY 12
-- Distribuição dos códigos de rede, situação de funcionamento
-- e localização.
--
-- Também contamos quantas escolas declaram oferta dos
-- anos iniciais do Ensino Fundamental.
--
-- Ainda NÃO filtramos situação de funcionamento porque
-- primeiro queremos descobrir/validar sua codificação.
-- ============================================================

SELECT
  ano,
  rede,
  tipo_situacao_funcionamento,
  tipo_localizacao,

  COUNT(*) AS quantidade_escolas,

  COUNTIF(
    etapa_ensino_fundamental_anos_iniciais = 1
  ) AS escolas_com_anos_iniciais

FROM
  `basedosdados.br_inep_censo_escolar.escola`

WHERE
  ano IN (2022, 2023)

GROUP BY
  ano,
  rede,
  tipo_situacao_funcionamento,
  tipo_localizacao

ORDER BY
  ano,
  rede,
  tipo_situacao_funcionamento,
  tipo_localizacao;


-- ============================================================
-- QUERY 13
-- Perfil de cobertura das principais features candidatas.
--
-- Universo:
-- escolas que declaram oferta dos anos iniciais do
-- Ensino Fundamental.
--
-- Não filtramos ainda situação de funcionamento.
-- ============================================================

SELECT

  ano,

  COUNT(*) AS escolas_anos_iniciais,

  -- ----------------------------------------------------------
  -- Internet para aprendizagem
  -- ----------------------------------------------------------

  COUNTIF(
    internet_aprendizagem IS NULL
  ) AS internet_aprendizagem_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        internet_aprendizagem IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS internet_aprendizagem_pct_nulos,

  AVG(
    CAST(
      internet_aprendizagem AS FLOAT64
    )
  ) AS media_internet_aprendizagem,


  -- ----------------------------------------------------------
  -- Biblioteca / sala de leitura
  -- ----------------------------------------------------------

  COUNTIF(
    biblioteca_sala_leitura IS NULL
  ) AS biblioteca_sala_leitura_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        biblioteca_sala_leitura IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS biblioteca_sala_leitura_pct_nulos,

  AVG(
    CAST(
      biblioteca_sala_leitura AS FLOAT64
    )
  ) AS media_biblioteca_sala_leitura,


  -- ----------------------------------------------------------
  -- Laboratório de informática
  -- ----------------------------------------------------------

  COUNTIF(
    laboratorio_informatica IS NULL
  ) AS laboratorio_informatica_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        laboratorio_informatica IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS laboratorio_informatica_pct_nulos,

  AVG(
    CAST(
      laboratorio_informatica AS FLOAT64
    )
  ) AS media_laboratorio_informatica,


  -- ----------------------------------------------------------
  -- Computadores disponíveis aos alunos
  -- ----------------------------------------------------------

  COUNTIF(
    quantidade_computador_aluno IS NULL
  ) AS computador_aluno_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        quantidade_computador_aluno IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS computador_aluno_pct_nulos,


  -- ----------------------------------------------------------
  -- Matrículas nos anos iniciais
  -- ----------------------------------------------------------

  COUNTIF(
    quantidade_matricula_fundamental_anos_iniciais IS NULL
  ) AS matricula_anos_iniciais_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        quantidade_matricula_fundamental_anos_iniciais IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS matricula_anos_iniciais_pct_nulos,


  -- ----------------------------------------------------------
  -- Docentes nos anos iniciais
  -- ----------------------------------------------------------

  COUNTIF(
    quantidade_docente_fundamental_anos_iniciais IS NULL
  ) AS docente_anos_iniciais_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        quantidade_docente_fundamental_anos_iniciais IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS docente_anos_iniciais_pct_nulos,


  -- ----------------------------------------------------------
  -- Turmas nos anos iniciais
  -- ----------------------------------------------------------

  COUNTIF(
    quantidade_turma_fundamental_anos_iniciais IS NULL
  ) AS turma_anos_iniciais_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        quantidade_turma_fundamental_anos_iniciais IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS turma_anos_iniciais_pct_nulos,


  -- ----------------------------------------------------------
  -- Matrículas em tempo integral
  -- ----------------------------------------------------------

  COUNTIF(
    quantidade_matricula_fundamental_anos_iniciais_integral
      IS NULL
  ) AS matricula_integral_anos_iniciais_nulos,

  ROUND(
    100 * SAFE_DIVIDE(
      COUNTIF(
        quantidade_matricula_fundamental_anos_iniciais_integral
          IS NULL
      ),
      COUNT(*)
    ),
    2
  ) AS matricula_integral_anos_iniciais_pct_nulos

FROM
  `basedosdados.br_inep_censo_escolar.escola`

WHERE
  ano IN (2022, 2023)

  AND etapa_ensino_fundamental_anos_iniciais = 1

GROUP BY
  ano

ORDER BY
  ano;


-- ============================================================
-- QUERY 14
-- Validade dos denominadores.
--
-- Precisamos saber se matrículas, docentes e turmas são
-- utilizáveis na engenharia de atributos.
--
-- Exemplos futuros:
--
-- alunos por turma
-- matrículas por docente
-- proporção de matrículas em tempo integral
-- computadores por 100 matrículas
-- ============================================================

SELECT

  ano,

  COUNT(*) AS escolas_anos_iniciais,

  COUNTIF(
    quantidade_matricula_fundamental_anos_iniciais IS NULL
  ) AS matriculas_nulas,

  COUNTIF(
    quantidade_matricula_fundamental_anos_iniciais = 0
  ) AS matriculas_zero,

  COUNTIF(
    quantidade_docente_fundamental_anos_iniciais IS NULL
  ) AS docentes_nulos,

  COUNTIF(
    quantidade_docente_fundamental_anos_iniciais = 0
  ) AS docentes_zero,

  COUNTIF(
    quantidade_turma_fundamental_anos_iniciais IS NULL
  ) AS turmas_nulas,

  COUNTIF(
    quantidade_turma_fundamental_anos_iniciais = 0
  ) AS turmas_zero,

  MIN(
    quantidade_matricula_fundamental_anos_iniciais
  ) AS matriculas_min,

  MAX(
    quantidade_matricula_fundamental_anos_iniciais
  ) AS matriculas_max,

  MIN(
    quantidade_docente_fundamental_anos_iniciais
  ) AS docentes_min,

  MAX(
    quantidade_docente_fundamental_anos_iniciais
  ) AS docentes_max,

  MIN(
    quantidade_turma_fundamental_anos_iniciais
  ) AS turmas_min,

  MAX(
    quantidade_turma_fundamental_anos_iniciais
  ) AS turmas_max

FROM
  `basedosdados.br_inep_censo_escolar.escola`

WHERE
  ano IN (2022, 2023)

  AND etapa_ensino_fundamental_anos_iniciais = 1

GROUP BY
  ano

ORDER BY
  ano;