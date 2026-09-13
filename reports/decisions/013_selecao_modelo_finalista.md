# Decisão 013: Seleção do modelo finalista

## Objetivo

Comparar os modelos Random Forest e HistGradientBoosting após tuning de
hiperparâmetros e selecionar a família de modelo que avançará para a definição
do threshold operacional.

A decisão utiliza exclusivamente dados de desenvolvimento de 2023.

Os dados de 2024 permaneceram fechados durante todas as decisões de seleção de
features, algoritmo e hiperparâmetros.

## Condições da comparação

Os dois modelos utilizaram:

- conjunto ENRICHED + UF;
- 18 features brutas;
- preprocessing integrado;
- cinco folds de GroupKFold por município;
- ausência de sobreposição de municípios entre treino e validação;
- PR-AUC da classe de risco como métrica principal;
- ROC-AUC como métrica complementar.

O threshold 0,50 foi utilizado somente para métricas diagnósticas.

O threshold operacional ainda não foi selecionado.

## Random Forest otimizado

Melhor configuração:

- n_estimators: 250
- max_depth: 12
- min_samples_split: 50
- min_samples_leaf: 200
- max_features: 0.5
- criterion: gini
- class_weight: None

Resultados médios de validação:

- PR-AUC risco: 0,54361
- desvio-padrão PR-AUC: 0,01197
- ROC-AUC: 0,64290
- Balanced Accuracy @0,50: 0,57674
- Precision risco @0,50: 0,59620
- Recall risco @0,50: 0,29695
- F1 risco @0,50: 0,39515
- gap estimado de PR-AUC treino-validação: 0,03861

O tuning melhorou o Random Forest em relação à configuração anterior,
confirmando que a comparação final não estava limitada pelos hiperparâmetros
utilizados anteriormente.

## HistGradientBoosting otimizado

Melhor configuração:

- learning_rate: 0,06198
- max_iter: 230
- max_leaf_nodes: 15
- max_depth: None
- min_samples_leaf: 70
- l2_regularization: 0,71640
- max_bins: 255

Resultados médios de validação:

- PR-AUC risco: 0,54463
- desvio-padrão PR-AUC: 0,00981
- ROC-AUC: 0,64403
- Balanced Accuracy @0,50: 0,58323
- Precision risco @0,50: 0,58504
- Recall risco @0,50: 0,33770
- F1 risco @0,50: 0,42765
- gap médio de PR-AUC treino-validação: 0,03223

## Comparação

A diferença de PR-AUC entre os dois modelos é pequena:

- HGB: 0,54463
- Random Forest: 0,54361
- diferença aproximada: +0,00102 para o HGB

A diferença de ROC-AUC também é pequena:

- HGB: 0,64403
- Random Forest: 0,64290
- diferença aproximada: +0,00113 para o HGB

Portanto, a seleção não se baseia apenas na diferença absoluta de uma única
métrica.

O HistGradientBoosting também apresentou:

- menor variabilidade de PR-AUC entre folds;
- menor gap entre treino e validação;
- maior Balanced Accuracy no threshold diagnóstico;
- maior recall da classe de risco;
- maior F1 da classe de risco;
- custo computacional substancialmente menor nos experimentos realizados.

O Random Forest apresentou maior precision da classe de risco no threshold
0,50, porém esse threshold ainda não é o threshold operacional final.

## Consideração sobre custo computacional

Nos experimentos de tuning, a avaliação dos cinco folds da configuração
selecionada do HistGradientBoosting consumiu aproximadamente 113 segundos de
tempo de treinamento e avaliação.

A configuração selecionada do Random Forest consumiu aproximadamente 1.693
segundos nos cinco folds.

Os tempos dependem do hardware e da execução, portanto não são tratados como
benchmark universal.

Ainda assim, dentro do mesmo ambiente experimental, o HGB apresentou vantagem
operacional relevante sem sacrificar performance preditiva.

## Decisão

O HistGradientBoostingClassifier é selecionado como modelo finalista.

A decisão considera conjuntamente:

- PR-AUC;
- ROC-AUC;
- estabilidade entre folds;
- generalização;
- desempenho da classe de risco;
- custo computacional.

O conjunto final de features permanece ENRICHED + UF.

Os hiperparâmetros do modelo ficam congelados em:

- learning_rate: 0,06197763962309192
- max_iter: 230
- max_leaf_nodes: 15
- max_depth: None
- min_samples_leaf: 70
- l2_regularization: 0,716404042819101
- max_bins: 255
- early_stopping: False
- random_state: 42

Nenhuma nova alteração de algoritmo, features ou hiperparâmetros deverá ser
realizada utilizando informações de 2024.

## Próxima etapa

Gerar probabilidades out-of-fold de 2023 com a configuração finalista e
selecionar o threshold operacional.

O critério principal para escolha do threshold será Balanced Accuracy, com
desempate por maior recall da classe de risco e posteriormente maior F1 da
classe de risco.

Somente após o congelamento do threshold o modelo será treinado com todo o
conjunto de 2023 e avaliado uma única vez em 2024.