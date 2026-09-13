# Decisão 012: Tuning do HistGradientBoosting

## Objetivo

Otimizar os hiperparâmetros do HistGradientBoostingClassifier utilizando o
conjunto de features ENRICHED + UF previamente selecionado.

A otimização utilizou exclusivamente dados de 2023.

Os dados de 2024 permaneceram reservados para avaliação temporal final e não
foram utilizados para escolha de hiperparâmetros ou threshold.

## Estratégia

Foi utilizada busca aleatória de hiperparâmetros com 15 configurações:

- 1 configuração correspondente ao HGB utilizado na comparação de famílias;
- 14 configurações amostradas aleatoriamente.

Cada configuração foi avaliada utilizando cinco folds de GroupKFold por
município.

Foram realizados 75 fits no total.

O PR-AUC da classe de risco foi utilizado como métrica principal de seleção.

ROC-AUC foi utilizada como métrica complementar.

Métricas calculadas com threshold 0,50 foram mantidas apenas como diagnóstico e
não representam o threshold operacional final.

Também foi acompanhado o gap de PR-AUC entre treino e validação para auxiliar
na avaliação de overfitting.

## Configuração anterior

A configuração anterior do HistGradientBoosting utilizava:

- learning_rate: 0,08
- max_iter: 150
- max_leaf_nodes: 31
- max_depth: None
- min_samples_leaf: 50
- l2_regularization: 1,0
- max_bins: 255

Resultados médios nos cinco folds:

- PR-AUC risco de validação: 0,54258
- ROC-AUC de validação: 0,64168
- Balanced Accuracy @0,50: 0,58127
- Precision risco @0,50: 0,58071
- Recall risco @0,50: 0,33497
- F1 risco @0,50: 0,42453
- gap médio de PR-AUC treino-validação: 0,04239

## Melhor configuração

O candidato 6 apresentou o maior PR-AUC médio de validação.

Hiperparâmetros:

- learning_rate: 0,06198
- max_iter: 230
- max_leaf_nodes: 15
- max_depth: None
- min_samples_leaf: 70
- l2_regularization: 0,71640
- max_bins: 255

Resultados médios:

- PR-AUC risco de validação: 0,54463
- desvio-padrão PR-AUC: 0,00981
- ROC-AUC de validação: 0,64403
- Balanced Accuracy @0,50: 0,58323
- Precision risco @0,50: 0,58504
- Recall risco @0,50: 0,33770
- F1 risco @0,50: 0,42765
- gap médio de PR-AUC treino-validação: 0,03223

## Ganho do tuning

Em comparação com a configuração anterior avaliada dentro do mesmo processo de
cross-validation:

- PR-AUC risco: aproximadamente +0,00206
- ROC-AUC: aproximadamente +0,00235
- Balanced Accuracy: aproximadamente +0,00196
- Precision risco: aproximadamente +0,00433
- Recall risco: aproximadamente +0,00273
- F1 risco: aproximadamente +0,00312

O gap médio de PR-AUC entre treino e validação caiu aproximadamente 0,0102.

## Interpretação

O ganho de performance foi incremental, porém acompanhado por redução do gap
entre treino e validação.

A configuração selecionada possui menor número máximo de folhas e maior
min_samples_leaf que a configuração anterior.

Isso indica que a busca encontrou um modelo mais conservador que apresentou
melhor capacidade de generalização na validação por municípios.

O resultado não justifica continuar aumentando indefinidamente o orçamento de
busca sobre os mesmos folds, pois sucessivas otimizações também podem provocar
overfitting de hiperparâmetros.

## Decisão

O candidato 6 passa a ser a configuração finalista do
HistGradientBoostingClassifier.

Essa decisão ainda não torna o HGB o modelo campeão.

O Random Forest deverá receber tuning utilizando o mesmo conjunto ENRICHED + UF
e a mesma estratégia de validação por municípios.

Somente após a comparação dos modelos otimizados será definido o modelo
finalista.

O threshold operacional também permanece indefinido.

2024 continua reservado exclusivamente para a avaliação temporal final.