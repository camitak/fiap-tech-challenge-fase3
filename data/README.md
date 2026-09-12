# Dados

Os dados utilizados neste projeto são processados no Google BigQuery e não são
versionados diretamente neste repositório.

## Base principal para Machine Learning

Projeto GCP:

`fiap-tc-f2-camila-takemoto`

Dataset:

`alfabetizacao_gold`

Tabela:

`gold_ml_aluno`

## Granularidade

Uma linha por aluno elegível por ano de avaliação.

Critérios de elegibilidade:

- aluno presente na avaliação;
- prova preenchida;
- registro classificado como `VALID` na camada Silver.

## Volume

- 2023: 1.502.809 registros;
- 2024: 1.851.852 registros;
- Total: 3.354.661 registros.

## Target

`alfabetizado`

- `0`: não alfabetizado;
- `1`: alfabetizado.

## Observação sobre leakage

A variável individual `proficiencia` não faz parte da Gold de modelagem porque
o target de alfabetização é derivado diretamente do desempenho na avaliação.

Indicadores agregados de alfabetização do mesmo ano também não são utilizados
como features do modelo individual.

## Enriquecimento

A base incorpora contexto territorial, demográfico e econômico municipal a
partir de fontes públicas, utilizando `id_municipio` como chave de integração.