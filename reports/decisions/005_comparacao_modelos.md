# Decisão 005 — Comparação inicial dos modelos supervisionados

## Objetivo

Comparar modelos supervisionados com diferentes níveis de complexidade antes
da etapa de otimização de hiperparâmetros.

Foram avaliados:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest.

Todos os modelos utilizaram o mesmo conjunto inicial de features definido pela
EDA e somente dados de 2023.

O conjunto de 2024 permaneceu isolado.

## Estratégia de validação

A comparação principal utilizou validação cruzada com 5 folds e agrupamento
por `id_municipio`.

Dessa forma, um município presente no conjunto de treinamento de um fold não
pode aparecer simultaneamente na validação daquele mesmo fold.

`id_municipio` é utilizado somente como variável de agrupamento e não como
feature do modelo.

## Resultados médios

| Modelo | Balanced Accuracy | ROC-AUC | PR-AUC não alfabetizado | Recall não alfabetizado | F1 não alfabetizado |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,5085 | 0,5637 | 0,4624 | 0,0564 | 0,1004 |
| Árvore de Decisão | 0,5283 | 0,5573 | 0,4539 | 0,3007 | 0,3652 |
| Random Forest | 0,5199 | 0,5728 | 0,4730 | 0,1678 | 0,2475 |

## Interpretação

A Regressão Logística apresentou baixo poder de classificação da população em
risco e não será priorizada na etapa de otimização.

A Árvore de Decisão apresentou os melhores resultados no threshold padrão de
0,5 para Balanced Accuracy, Recall e F1 da classe não alfabetizado.

A Random Forest apresentou os maiores valores médios de ROC-AUC e PR-AUC para
não alfabetizados, indicando melhor capacidade de discriminação e ordenação do
risco.

Como ROC-AUC e PR-AUC avaliam a qualidade das probabilidades em diferentes
limiares, a Random Forest permanece como forte candidata para otimização,
mesmo apresentando recall inferior à árvore no threshold padrão de 0,5.

## Decisão

A próxima etapa irá priorizar modelos baseados em árvores.

Serão otimizadas:

1. Random Forest, devido à melhor capacidade de discriminação;
2. Árvore de Decisão, utilizada como modelo mais simples e interpretável de
   comparação.

A escolha do modelo final não será baseada somente em accuracy.

Serão consideradas principalmente:

- ROC-AUC;
- PR-AUC da classe não alfabetizado;
- Balanced Accuracy;
- Recall da classe não alfabetizado;
- F1 da classe não alfabetizado;
- estabilidade entre folds.

A escolha e o ajuste do threshold serão realizados somente com dados de 2023,
antes da avaliação final em 2024.