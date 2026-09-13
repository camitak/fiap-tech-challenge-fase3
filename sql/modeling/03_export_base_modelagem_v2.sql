-- ============================================================
-- Tech Challenge Fase 3
-- Exportação da base reproduzível de modelagem - versão 2
--
-- Origem:
-- alfabetizacao_gold.gold_ml_aluno_enriquecido
--
-- Destino:
-- GCS dedicado aos artefatos de ML da Fase 3
--
-- Estratégia:
--   v2/ano=2023 -> desenvolvimento
--   v2/ano=2024 -> teste temporal final
--
-- IMPORTANTE:
-- 2024 é exportado para garantir reprodutibilidade, mas NÃO
-- deve ser utilizado para seleção de features, algoritmos,
-- hiperparâmetros ou thresholds.
-- ============================================================


-- ============================================================
-- QUERY 01
-- Auditoria da base antes da exportação
-- ============================================================

SELECT

  ano,

  COUNT(*) AS total_linhas,

  COUNT(DISTINCT aluno_key) AS alunos_distintos,

  COUNTIF(
    alfabetizado = 0
  ) AS nao_alfabetizados,

  COUNTIF(
    alfabetizado = 1
  ) AS alfabetizados,

  COUNTIF(
    contexto_educacional_disponivel = 0
  ) AS sem_contexto_educacional,

  COUNTIF(
    alunos_por_turma_anos_iniciais IS NULL
  ) AS null_alunos_por_turma,

  COUNTIF(
    razao_matriculas_docentes_anos_iniciais IS NULL
  ) AS null_matriculas_docentes

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

GROUP BY
  ano

ORDER BY
  ano;


-- ============================================================
-- EXPORTAÇÃO 2023
--
-- Desenvolvimento:
-- EDA complementar, feature selection, comparação de modelos,
-- tuning e threshold.
-- ============================================================

EXPORT DATA OPTIONS (

  uri = 'gs://fiap-tc-f2-camila-takemoto-fase3-ml/base_modelagem/v2/ano=2023/part-*.parquet',

  format = 'PARQUET',

  overwrite = TRUE

)

AS

SELECT

  -- ==========================================================
  -- Controle e validação
  -- ==========================================================

  ano,

  aluno_key,

  id_municipio,

  alfabetizado,


  -- ==========================================================
  -- Categóricas / territoriais
  -- ==========================================================

  rede_nome,

  regiao,

  sigla_uf,


  -- ==========================================================
  -- Binárias territoriais
  -- ==========================================================

  capital_uf,

  amazonia_legal,


  -- ==========================================================
  -- Contexto socioeconômico
  -- ==========================================================

  populacao_2020,

  pib_por_habitante_2020,

  participacao_agropecuaria_2020,

  participacao_industria_2020,

  participacao_administracao_publica_2020,


  -- ==========================================================
  -- Contexto educacional
  -- Censo Escolar do ano anterior ao target
  -- ==========================================================

  quantidade_escolas_anos_iniciais,

  total_matriculas_anos_iniciais,

  total_docentes_anos_iniciais,

  total_turmas_anos_iniciais,

  proporcao_escolas_rurais,

  proporcao_internet_aprendizagem,

  proporcao_biblioteca_sala_leitura,

  proporcao_laboratorio_informatica,

  alunos_por_turma_anos_iniciais,

  razao_matriculas_docentes_anos_iniciais,

  proporcao_matriculas_integral_anos_iniciais

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

WHERE
  ano = 2023;


-- ============================================================
-- EXPORTAÇÃO 2024
--
-- Teste temporal final.
--
-- O arquivo é exportado agora apenas para garantir
-- reprodutibilidade e simetria da base.
--
-- NÃO utilizar durante desenvolvimento.
-- ============================================================

EXPORT DATA OPTIONS (

  uri = 'gs://fiap-tc-f2-camila-takemoto-fase3-ml/base_modelagem/v2/ano=2024/part-*.parquet',

  format = 'PARQUET',

  overwrite = TRUE

)

AS

SELECT

  -- ==========================================================
  -- Controle e validação
  -- ==========================================================

  ano,

  aluno_key,

  id_municipio,

  alfabetizado,


  -- ==========================================================
  -- Categóricas / territoriais
  -- ==========================================================

  rede_nome,

  regiao,

  sigla_uf,


  -- ==========================================================
  -- Binárias territoriais
  -- ==========================================================

  capital_uf,

  amazonia_legal,


  -- ==========================================================
  -- Contexto socioeconômico
  -- ==========================================================

  populacao_2020,

  pib_por_habitante_2020,

  participacao_agropecuaria_2020,

  participacao_industria_2020,

  participacao_administracao_publica_2020,


  -- ==========================================================
  -- Contexto educacional
  -- Censo Escolar do ano anterior ao target
  -- ==========================================================

  quantidade_escolas_anos_iniciais,

  total_matriculas_anos_iniciais,

  total_docentes_anos_iniciais,

  total_turmas_anos_iniciais,

  proporcao_escolas_rurais,

  proporcao_internet_aprendizagem,

  proporcao_biblioteca_sala_leitura,

  proporcao_laboratorio_informatica,

  alunos_por_turma_anos_iniciais,

  razao_matriculas_docentes_anos_iniciais,

  proporcao_matriculas_integral_anos_iniciais

FROM
  `fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno_enriquecido`

WHERE
  ano = 2024;