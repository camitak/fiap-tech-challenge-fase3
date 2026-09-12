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

## Reprodutibilidade da EDA

A fonte oficial dos dados é a tabela:

`fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

armazenada no Google BigQuery.

Para permitir a reprodução da análise sem necessidade de credenciais do Google
Cloud, o repositório contém um snapshot agregado derivado da Gold:

`data/eda/municipio_ano_rede.csv`

O arquivo possui granularidade de:

`ano + município + rede`

e não contém identificadores individuais de alunos ou escolas.

### Executar localmente

Instale as dependências:

```bash
python -m pip install -r requirements.txt