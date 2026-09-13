# Decisão 009: Comparação BASELINE × ENRICHED

## Objetivo

Avaliar se as variáveis educacionais provenientes do Censo Escolar adicionam
capacidade preditiva ao conjunto baseline.

O experimento foi realizado exclusivamente com dados de 2023.

Os dados de 2024 permaneceram isolados e não foram utilizados para seleção de
features, modelos, hiperparâmetros ou thresholds.

## Metodologia

Foram comparados dois conjuntos:

### BASELINE

9 features originais.

Após preprocessing: 14 features.

### ENRICHED

9 features baseline + 8 features educacionais selecionadas após EDA.

Total: 17 features brutas.

Após preprocessing: 22 features.

Foram utilizados os mesmos 5 folds de GroupKFold por município para todos os
experimentos, garantindo ausência de sobreposição de municípios entre treino e
validação.

Foram avaliados:

- Decision Tree;
- Random Forest.

Os hiperparâmetros foram mantidos fixos para que a única diferença entre os
experimentos fosse o conjunto de features.

O threshold foi mantido em 0,50 apenas para métricas diagnósticas.

PR-AUC de risco e ROC-AUC foram priorizadas nesta comparação por serem
independentes de threshold.

## Random Forest

### Baseline

- PR-AUC risco OOF: 0,4732
- ROC-AUC OOF: 0,5751
- Balanced Accuracy @0,50: 0,5225
- Precision risco @0,50: 0,4899
- Recall risco @0,50: 0,1747
- F1 risco @0,50: 0,2575

### Enriched

- PR-AUC risco OOF: 0,4994
- ROC-AUC OOF: 0,6015
- Balanced Accuracy @0,50: 0,5425
- Precision risco @0,50: 0,5386
- Recall risco @0,50: 0,2181
- F1 risco @0,50: 0,3105

### Ganhos

- PR-AUC risco: +0,0262
- ROC-AUC: +0,0263
- Balanced Accuracy: +0,0199
- Precision risco: +0,0486
- Recall risco: +0,0435
- F1 risco: +0,0530

O enriquecimento também apresentou menor variabilidade entre folds.

O desvio-padrão do PR-AUC caiu aproximadamente de 0,0265 para 0,0095 e o
desvio-padrão do ROC-AUC caiu aproximadamente de 0,0178 para 0,0042.

## Decision Tree

A Decision Tree apresentou ganhos menores:

- PR-AUC risco: +0,0036;
- ROC-AUC: +0,0037;
- Balanced Accuracy: +0,0063.

Recall e F1 utilizando threshold 0,50 diminuíram.

Como o threshold ainda não foi otimizado para o novo conjunto de features,
essas métricas não são utilizadas isoladamente para rejeitar o enriquecimento.

## Decisão

O conjunto ENRICHED avança como conjunto principal de features.

O BASELINE será preservado como referência experimental.

As evidências mostram que o contexto educacional proveniente do Censo Escolar
adiciona sinal preditivo ao conjunto territorial e socioeconômico anterior,
principalmente no Random Forest.

Não há evidência de que o modelo tenha alcançado alta capacidade discriminativa.
Os valores de ROC-AUC e PR-AUC permanecem moderados, coerentes com a limitação
de utilizar predominantemente variáveis contextuais para prever um target no
nível individual.

## Próxima etapa

Avaliar separadamente a inclusão de `sigla_uf`.

O experimento deverá comparar:

- ENRICHED;
- ENRICHED + UF.

A comparação continuará utilizando somente 2023 e os mesmos grupos por
município.

Somente após essa decisão serão avaliados novos algoritmos e realizada nova
otimização de hiperparâmetros.

2024 permanece reservado para avaliação temporal final.