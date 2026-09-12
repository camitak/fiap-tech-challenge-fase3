# Decisão 004 — Estratégia de validação supervisionada

## Objetivo

Definir a separação entre desenvolvimento, validação e teste da
classificação de alfabetização.

## Separação temporal

A estratégia adotada é:

- 2023: desenvolvimento do modelo;
- 2024: teste temporal final.

O conjunto de 2024 permanece isolado durante:

- seleção de features;
- comparação de algoritmos;
- ajuste de hiperparâmetros;
- escolha de limiar;
- decisões de pré-processamento.

O conjunto final somente será utilizado após a definição do modelo campeão.

## Validação dentro de 2023

Como grande parte das variáveis utilizadas possui granularidade municipal,
um split aleatório por aluno poderia colocar alunos do mesmo município
simultaneamente nos conjuntos de treino e validação.

Isso poderia produzir uma estimativa excessivamente otimista da capacidade
de generalização territorial.

Por esse motivo, o baseline utiliza `GroupShuffleSplit`, tendo
`id_municipio` exclusivamente como variável de agrupamento.

Parâmetros:

- `test_size=0.20`;
- `random_state=42`;
- grupos definidos por `id_municipio`.

`id_municipio` não é utilizado como feature.

Após o split, é verificado programaticamente que não existe sobreposição
de municípios entre treino e validação.

## Target

A codificação original é preservada:

- `0`: não alfabetizado;
- `1`: alfabetizado.

Não é realizada inversão silenciosa do target.

Como a identificação do risco educacional é um objetivo relevante do
projeto, além das métricas globais são calculados Precision, Recall,
F1 e PR-AUC especificamente para a classe `não alfabetizado`.

## Baseline

O primeiro algoritmo utilizado é a Regressão Logística.

O objetivo do baseline é estabelecer uma referência simples,
interpretável e computacionalmente eficiente antes da comparação com
modelos de maior complexidade.

O baseline utiliza:

- `C=1.0`;
- solver `lbfgs`;
- `max_iter=300`;
- nenhuma ponderação automática das classes.

## Próximas etapas

A otimização de hiperparâmetros e a comparação de modelos serão realizadas
sem utilizar o conjunto de 2024.

Estratégias de validação cruzada com agrupamento municipal serão avaliadas
na etapa de otimização.