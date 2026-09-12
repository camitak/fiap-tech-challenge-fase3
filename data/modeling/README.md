# Base de modelagem supervisionada

Esta pasta contém a versão reproduzível da base utilizada na etapa de
Machine Learning supervisionado da Fase 3.

## Origem oficial

Os dados foram extraídos da tabela Gold:

`fiap-tc-f2-camila-takemoto.alfabetizacao_gold.gold_ml_aluno`

construída no Google BigQuery a partir da pipeline de dados desenvolvida
na Fase 2.

A consulta utilizada para selecionar as variáveis da modelagem está em:

`sql/modeling/01_base_modelagem.sql`

A exportação para Parquet está documentada em:

`sql/modeling/02_export_base_modelagem.sql`

## Formato

Os dados são armazenados em Parquet e separados fisicamente por ano:

- `base_modelagem/v1/ano=2023/`
- `base_modelagem/v1/ano=2024/`

A separação temporal foi preservada deliberadamente.

## Estratégia temporal

- **2023:** desenvolvimento, treinamento, validação e otimização;
- **2024:** teste temporal final.

O conjunto de 2024 não deve ser utilizado para seleção de features,
comparação de modelos, ajuste de hiperparâmetros ou escolha de limiar.

## Volume

| Ano | Registros | Alfabetizados | Não alfabetizados |
|---|---:|---:|---:|
| 2023 | 1.502.809 | 877.427 | 625.382 |
| 2024 | 1.851.852 | 1.107.119 | 744.733 |
| Total | 3.354.661 | 1.984.546 | 1.370.115 |

## Variáveis disponíveis

A base contém:

- `ano`
- `alfabetizado`
- `rede_nome`
- `regiao`
- `sigla_uf`
- `capital_uf`
- `amazonia_legal`
- `populacao_2020`
- `pib_por_habitante_2020`
- `participacao_agropecuaria_2020`
- `participacao_industria_2020`
- `participacao_servicos_2020`
- `participacao_administracao_publica_2020`
- `id_municipio`

`id_municipio` é mantido apenas para avaliação territorial e não deve ser
utilizado como feature do modelo.

A variável `ano` também é utilizada para separar os períodos e não deve ser
utilizada como feature do classificador.

## Controle de data leakage

Variáveis diretamente relacionadas ao resultado da avaliação, identificadores
individuais e metadados técnicos da pipeline foram excluídos da matriz de
modelagem.

Entre as variáveis deliberadamente não utilizadas estão:

- proficiência;
- identificador de aluno;
- identificador de escola;
- métricas observadas de alfabetização do próprio município no mesmo ano;
- metas futuras;
- metadados técnicos de ingestão.

## Reconciliação

Após a exportação para Parquet, as contagens foram reconciliadas com a Gold
oficial.

Resultado:

- 2023: reconciliação OK;
- 2024: reconciliação OK.

Os arquivos versionados correspondem à base completa utilizada na modelagem,
não a uma amostra.