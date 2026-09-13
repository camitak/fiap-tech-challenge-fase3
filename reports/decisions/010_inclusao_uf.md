# Decisão 010: Inclusão de UF no conjunto de features

## Objetivo

Avaliar se a inclusão de `sigla_uf` adiciona capacidade preditiva ao conjunto
ENRICHED previamente selecionado.

O experimento foi realizado exclusivamente com dados de 2023.

Os dados de 2024 permaneceram reservados para a avaliação temporal final e não
foram utilizados para seleção de features, modelos, hiperparâmetros ou threshold.

## Metodologia

Foram comparados:

- ENRICHED: 17 features brutas e 22 features após preprocessing;
- ENRICHED + UF: 18 features brutas e 45 features após preprocessing.

A UF foi tratada como variável categórica utilizando One-Hot Encoding com
`handle_unknown="ignore"`.

Foram mantidos:

- os mesmos 5 folds de GroupKFold por município;
- os mesmos algoritmos;
- os mesmos hiperparâmetros;
- o mesmo preprocessing;
- threshold 0,50 apenas para métricas diagnósticas.

PR-AUC de risco e ROC-AUC foram priorizadas na decisão por serem independentes
do threshold de classificação.

## Random Forest

### ENRICHED

- PR-AUC risco OOF: 0,4994
- ROC-AUC OOF: 0,6015
- Balanced Accuracy @0,50: 0,5425
- Precision risco @0,50: 0,5386
- Recall risco @0,50: 0,2181
- F1 risco @0,50: 0,3105

### ENRICHED + UF

- PR-AUC risco OOF: 0,5404
- ROC-AUC OOF: 0,6417
- Balanced Accuracy @0,50: 0,5689
- Precision risco @0,50: 0,5939
- Recall risco @0,50: 0,2688
- F1 risco @0,50: 0,3701

### Ganhos

- PR-AUC risco: +0,0410
- ROC-AUC: +0,0402
- Balanced Accuracy: +0,0264
- Precision risco: +0,0553
- Recall risco: +0,0507
- F1 risco: +0,0596

O ganho também foi consistente entre os cinco folds.

Os PR-AUC de risco do Random Forest com UF foram aproximadamente:

- Fold 1: 0,5258
- Fold 2: 0,5418
- Fold 3: 0,5493
- Fold 4: 0,5524
- Fold 5: 0,5348

## Decision Tree

A inclusão de UF também melhorou a Decision Tree:

- PR-AUC risco: +0,0321
- ROC-AUC: +0,0380
- Balanced Accuracy: +0,0250
- Precision risco: +0,0201
- Recall risco: +0,0651
- F1 risco: +0,0386

A melhoria em mais de uma família de modelos reforça a evidência de que UF
contém sinal preditivo adicional ao conjunto enriquecido.

## Interpretação

A UF representa contexto territorial conhecido no momento da previsão e,
portanto, pode ser utilizada como feature.

Entretanto, sua contribuição não deve ser interpretada causalmente. A variável
pode funcionar como proxy para diferenças institucionais, socioeconômicas,
educacionais e de políticas públicas existentes entre estados.

Também existe uma limitação de generalização para UFs não observadas no
desenvolvimento de 2023.

O pipeline utiliza One-Hot Encoding com `handle_unknown="ignore"`, permitindo
processar categorias não vistas no treinamento. O desempenho nesses casos deverá
ser analisado separadamente na avaliação temporal final de 2024.

## Decisão

O conjunto ENRICHED + UF, com 18 features brutas, passa a ser o conjunto
principal para os próximos experimentos supervisionados.

Os conjuntos BASELINE e ENRICHED permanecem registrados como referências
experimentais.

A decisão ainda não define o modelo campeão, os hiperparâmetros finais ou o
threshold operacional.

## Próxima etapa

Comparar o desempenho das famílias já avaliadas com uma nova família de modelo
capaz de capturar relações não lineares e interações entre as variáveis.

A próxima família a ser avaliada será HistGradientBoostingClassifier.

A comparação continuará utilizando somente dados de 2023.

Somente após a definição de:

- conjunto de features;
- algoritmo;
- hiperparâmetros;
- threshold;

os dados de 2024 serão utilizados para avaliação temporal final.