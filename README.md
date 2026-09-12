# Tech Challenge Fase 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech FIAP em
Inteligência Artificial.

## Objetivo

Desenvolver uma solução de Machine Learning supervisionado capaz de estimar
se um aluno será classificado como alfabetizado ou não alfabetizado, utilizando
informações educacionais, territoriais e socioeconômicas.

Além da classificação individual, o projeto busca gerar inteligência aplicada
para identificar fatores associados à alfabetização e apoiar análises de risco
educacional no território brasileiro.

## Origem dos dados

O projeto dá continuidade ao pipeline de dados desenvolvido no Tech Challenge
da Fase 2.

A arquitetura original utiliza:

- Google Cloud Storage na camada Bronze;
- BigQuery nas camadas Silver e Gold;
- dados do Indicador Criança Alfabetizada;
- metas nacionais, estaduais e municipais.

Para a Fase 3 foi criada uma nova base analítica em granularidade aluno-ano:

`alfabetizacao_gold.gold_ml_aluno`

A base foi enriquecida com contexto territorial, populacional e econômico
municipal.

## Estrutura do projeto

```text
data/                 documentação das bases
notebooks/            análises exploratórias
sql/                  auditorias e construção da Gold
src/preprocessing/    pré-processamento
src/modeling/         treinamento e otimização
src/evaluation/       avaliação dos modelos
src/visualization/    visualizações
reports/              documentação e resultados
images/               gráficos utilizados no projeto