# Decisão 011: Comparação de famílias de modelos

## Objetivo

Comparar uma nova família de modelo, HistGradientBoostingClassifier, com
Decision Tree e Random Forest utilizando o conjunto de features previamente
selecionado ENRICHED + UF.

O objetivo desta etapa não foi realizar tuning definitivo, mas verificar se
Gradient Boosting apresenta potencial suficiente para avançar para a etapa de
otimização.

Toda a análise utilizou exclusivamente dados de 2023.

Os dados de 2024 permaneceram reservados para avaliação temporal final.

## Configuração

Todos os modelos utilizaram:

- 18 features brutas;
- 45 features após preprocessing;
- os mesmos cinco folds de GroupKFold por município;
- ausência de sobreposição de municípios entre treino e validação;
- target `alfabetizado`;
- risco definido como `1 - P(alfabetizado)`.

PR-AUC de risco e ROC-AUC foram utilizadas como métricas principais por serem
independentes do threshold operacional.

O threshold 0,50 foi utilizado apenas para métricas diagnósticas.

## Resultados OOF

### HistGradientBoosting

- Accuracy: 0,6225
- Balanced Accuracy: 0,5812
- ROC-AUC: 0,6422
- PR-AUC risco: 0,5422
- Precision risco @0,50: 0,5805
- Recall risco @0,50: 0,3348
- F1 risco @0,50: 0,4247
- Tempo total dos cinco folds: aproximadamente 69 segundos

### Random Forest

- Accuracy: 0,6192
- Balanced Accuracy: 0,5689
- ROC-AUC: 0,6417
- PR-AUC risco: 0,5404
- Precision risco @0,50: 0,5939
- Recall risco @0,50: 0,2688
- F1 risco @0,50: 0,3701
- Tempo total dos cinco folds: aproximadamente 564 segundos

### Decision Tree

- Accuracy: 0,5635
- Balanced Accuracy: 0,5695
- ROC-AUC: 0,5978
- PR-AUC risco: 0,4952
- Precision risco @0,50: 0,4806
- Recall risco @0,50: 0,6053
- F1 risco @0,50: 0,5358

## HistGradientBoosting versus Random Forest

Nas métricas independentes de threshold, os dois modelos apresentaram resultados
muito próximos.

Diferenças a favor do HistGradientBoosting:

- PR-AUC risco: aproximadamente +0,0017;
- ROC-AUC: aproximadamente +0,0005.

Essas diferenças são pequenas e não justificam, isoladamente, declarar um modelo
campeão.

No threshold diagnóstico de 0,50, o HistGradientBoosting apresentou:

- maior Balanced Accuracy;
- maior recall da classe de risco;
- maior F1 da classe de risco;
- pequena redução de precision de risco em relação ao Random Forest.

O HistGradientBoosting também apresentou custo computacional significativamente
menor nesta configuração experimental.

## Estabilidade entre folds

O HistGradientBoosting apresentou resultados consistentes nos cinco folds.

PR-AUC risco aproximado:

- Fold 1: 0,5306
- Fold 2: 0,5398
- Fold 3: 0,5516
- Fold 4: 0,5523
- Fold 5: 0,5386

O desvio-padrão entre folds permaneceu baixo e semelhante ao observado no
Random Forest.

## Decisão

Decision Tree não avança para a etapa principal de tuning devido à menor
capacidade de ranking observada em PR-AUC e ROC-AUC.

Random Forest e HistGradientBoosting avançam para otimização.

Ainda não existe modelo campeão.

A diferença entre Random Forest e HistGradientBoosting antes do tuning é pequena
nas métricas principais, portanto ambos deverão ser avaliados com
hiperparâmetros ajustados utilizando exatamente o conjunto ENRICHED + UF.

## Próxima etapa

Realizar tuning de:

- Random Forest;
- HistGradientBoosting.

A otimização utilizará:

- somente dados de 2023;
- GroupKFold por município;
- PR-AUC de risco como métrica principal;
- ROC-AUC e Balanced Accuracy como métricas complementares.

Após o tuning, serão geradas predições OOF para os modelos finalistas.

Somente depois da seleção de modelo, hiperparâmetros e threshold será realizada
a avaliação temporal final com dados de 2024.