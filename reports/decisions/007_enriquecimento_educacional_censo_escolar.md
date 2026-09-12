# Decisão 006 — Enriquecimento educacional com Censo Escolar

## Contexto

A primeira versão da base de modelagem continha variáveis territoriais e
socioeconômicas, porém apresentava baixa diversidade de atributos
propriamente educacionais.

Como o Tech Challenge propõe a predição da alfabetização utilizando
variáveis educacionais, territoriais e socioeconômicas, foi realizado um
enriquecimento adicional utilizando dados públicos do Censo Escolar.

## Fonte

Fonte externa:

`basedosdados.br_inep_censo_escolar.escola`

A tabela de escolas foi escolhida porque permite construir indicadores
educacionais em nível de município e rede sem depender do identificador de
escola presente na avaliação de alfabetização.

O `id_escola` da fonte de alfabetização não foi utilizado no join.

## Estratégia temporal

Para evitar data leakage, foram utilizadas informações do Censo Escolar do
ano anterior ao target:

- alfabetização 2023 <- Censo Escolar 2022;
- alfabetização 2024 <- Censo Escolar 2023.

Nenhum indicador educacional do próprio ano ou de anos posteriores ao target
foi utilizado.

## Granularidade do enriquecimento

O contexto educacional foi agregado por:

- ano de referência;
- município;
- rede de ensino.

A correspondência das redes foi realizada a partir do dicionário oficial da
fonte do Censo Escolar:

- 1 = Federal;
- 2 = Estadual;
- 3 = Municipal;
- 4 = Privada.

Somente escolas:

- em atividade;
- com oferta dos anos iniciais do Ensino Fundamental

foram consideradas.

## Features educacionais construídas

Foram criadas as seguintes variáveis candidatas:

- quantidade de escolas dos anos iniciais;
- total de matrículas dos anos iniciais;
- total de docentes dos anos iniciais;
- total de turmas dos anos iniciais;
- proporção de escolas rurais;
- proporção de escolas com internet para aprendizagem;
- proporção de escolas com biblioteca ou sala de leitura;
- proporção de escolas com laboratório de informática;
- alunos por turma nos anos iniciais;
- razão entre matrículas e docentes nos anos iniciais;
- proporção de matrículas em tempo integral nos anos iniciais.

A variável `quantidade_computador_aluno` foi descartada porque apresentou
100% de valores nulos nos anos analisados.

## Tratamento das razões

As razões foram calculadas somente após a agregação por município e rede.

Foi utilizado `SAFE_DIVIDE` para evitar divisões inválidas quando o total de
docentes ou turmas é igual a zero.

Os valores extremos foram preservados na camada Gold. Nenhum clipping ou
winsorization foi aplicado na engenharia de dados.

Qualquer transformação necessária será realizada posteriormente dentro da
pipeline de Machine Learning e definida somente com os dados de
desenvolvimento de 2023.

## Cobertura

A correspondência entre a Gold de alfabetização e o contexto educacional foi
praticamente completa.

Em 2023:

- rede estadual: 100% de cobertura;
- rede municipal: 99,9945% de cobertura;
- 75 alunos ficaram sem contexto educacional.

Em 2024:

- rede estadual: 100%;
- rede municipal: 100%;
- rede privada: 100%.

Os 75 alunos sem correspondência de 2023 foram preservados na Gold.

Suas features educacionais permanecem nulas e serão tratadas pelo mecanismo
de imputação da pipeline de Machine Learning.

Nenhum registro foi excluído por ausência de contexto educacional.

## Qualidade da Gold enriquecida

A tabela:

`alfabetizacao_gold.gold_ml_aluno_enriquecido`

preserva integralmente a população da Gold original.

### 2023

- registros originais: 1.502.809;
- registros enriquecidos: 1.502.809;
- alunos distintos: 1.502.809;
- não alfabetizados: 625.382;
- alfabetizados: 877.427;
- diferença de linhas: 0.

### 2024

- registros originais: 1.851.852;
- registros enriquecidos: 1.851.852;
- alunos distintos: 1.851.852;
- não alfabetizados: 744.733;
- alfabetizados: 1.107.119;
- diferença de linhas: 0.

Não foram encontradas duplicidades de `aluno_key`.

## Validação temporal

Todos os registros com contexto educacional respeitam:

`ano_contexto_educacional = ano_target - 1`

Resultados:

- 2023 utiliza exclusivamente 2022;
- 2024 utiliza exclusivamente 2023;
- problemas temporais: 0.

## Validação semântica

Não foram identificados:

- valores negativos em quantidades;
- proporções menores que 0;
- proporções maiores que 1;
- razões educacionais negativas.

## Valores faltantes

Os valores faltantes são residuais.

Em 2023:

- 75 alunos sem qualquer contexto educacional;
- 86 alunos com `alunos_por_turma` nulo;
- 86 alunos com razão matrículas/docentes nula.

Em 2024:

- nenhum aluno sem contexto educacional;
- 9 alunos com `alunos_por_turma` nulo;
- 9 alunos com razão matrículas/docentes nula.

Esses valores não serão preenchidos na Gold.

A imputação será realizada dentro do pipeline de Machine Learning para evitar
data leakage.

## Valores extremos

Foram observados valores extremos nas razões educacionais.

Entretanto, os percentis mostram que são casos raros.

Em 2023, por exemplo:

- mediana de alunos por turma: aproximadamente 20,1;
- P95: 27,5;
- P99: aproximadamente 33,8;
- máximo: 80.

Por esse motivo, os extremos não foram removidos da Gold.

Transformações robustas poderão ser avaliadas posteriormente usando somente
o conjunto de desenvolvimento de 2023.

## Resultado

O enriquecimento adiciona ao problema uma dimensão educacional que não estava
adequadamente representada na primeira base de modelagem.

A nova Gold combina:

- variáveis territoriais;
- variáveis socioeconômicas;
- variáveis educacionais.

Essa tabela passa a ser a fonte da próxima versão reproduzível da base de
modelagem.