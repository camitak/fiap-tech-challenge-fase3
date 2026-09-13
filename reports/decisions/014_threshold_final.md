# Decisão 014: Seleção do threshold operacional final

## Objetivo

Selecionar o threshold operacional do modelo finalista utilizando exclusivamente
predições out-of-fold do conjunto de desenvolvimento de 2023.

Esta etapa foi realizada após o congelamento de:

- conjunto de features;
- algoritmo;
- hiperparâmetros.

Os dados de 2024 permaneceram completamente fechados durante a seleção do
threshold.

## Modelo congelado

Modelo:

HistGradientBoostingClassifier

Feature set:

ENRICHED + UF

Hiperparâmetros:

- learning_rate: 0,06197763962309192
- max_iter: 230
- max_leaf_nodes: 15
- max_depth: None
- min_samples_leaf: 70
- l2_regularization: 0,716404042819101
- max_bins: 255
- early_stopping: False
- random_state: 42

## Estratégia de validação

Foram utilizadas predições out-of-fold com cinco folds de GroupKFold por
município.

Cada aluno recebeu uma probabilidade gerada por um modelo que não foi treinado
com seu município.

Não houve sobreposição de municípios entre treino e validação em nenhum fold.

A definição de risco utilizada foi:

risco = 1 - P(alfabetizado)

## Performance OOF independente de threshold

- ROC-AUC: 0,64455
- PR-AUC risco: 0,54465
- média PR-AUC entre folds: 0,54463
- desvio-padrão PR-AUC entre folds: 0,00981
- média ROC-AUC entre folds: 0,64403
- desvio-padrão ROC-AUC entre folds: 0,00956

## Regra de seleção do threshold

Antes da avaliação do grid, foi estabelecida a seguinte regra:

1. maximizar Balanced Accuracy;
2. em caso de empate, selecionar maior recall da classe de risco;
3. persistindo o empate, selecionar maior F1 da classe de risco.

Foi avaliado um grid de thresholds de risco entre 0,10 e 0,90, com incremento de
0,01.

## Threshold selecionado

O threshold selecionado foi:

**0,41**

Resultados OOF:

- Accuracy: 0,59437
- Balanced Accuracy: 0,59928
- Precision risco: 0,51025
- Recall risco: 0,62859
- F1 risco: 0,56327
- Recall alfabetizado: 0,56998
- taxa real de risco: 0,41614
- taxa prevista de risco: 0,51265

Matriz de confusão considerando risco como classe positiva:

- True Negative: 500.114
- False Positive: 377.313
- False Negative: 232.273
- True Positive: 393.109

## Comparação com threshold 0,50

No threshold 0,50:

- Balanced Accuracy: 0,58325
- Precision risco: 0,58439
- Recall risco: 0,33767
- F1 risco: 0,42802
- taxa prevista de risco: 0,24045

A redução do threshold para 0,41 aumenta substancialmente a capacidade do modelo
de identificar alunos da classe de risco.

Há redução de precision e aumento do número de falsos positivos.

Esse trade-off é consistente com o objetivo de utilizar o modelo como mecanismo
de priorização e triagem, em que deixar de identificar alunos potencialmente em
risco pode ser mais prejudicial que encaminhar casos adicionais para avaliação.

## Interpretação operacional

O modelo não deve ser interpretado como diagnóstico individual de alfabetização.

Com precision da classe de risco próxima de 51%, a previsão deve funcionar como
sinal de priorização para investigação ou direcionamento de políticas, e não como
substituição da avaliação educacional.

O threshold 0,41 sinaliza aproximadamente 51,3% dos alunos como pertencentes ao
grupo de risco no desenvolvimento OOF de 2023.

Esse percentual não deve ser interpretado como estimativa da prevalência real de
não alfabetização. Ele é consequência da regra de decisão adotada para equilibrar
sensibilidade e especificidade.

## Comparação com thresholds próximos

Threshold 0,43 apresentou Balanced Accuracy muito próxima à obtida em 0,41, mas
menor recall da classe de risco.

Como a regra de seleção foi definida antes da análise do grid, não foi realizada
alteração pós-hoc do threshold em função de conveniência operacional.

## Decisão

O threshold operacional final da probabilidade de risco fica congelado em:

**0,41**

A partir deste ponto ficam congelados:

- conjunto de features ENRICHED + UF;
- HistGradientBoostingClassifier;
- hiperparâmetros selecionados;
- threshold de risco 0,41.

Nenhuma dessas escolhas deverá ser alterada com base nos resultados de 2024.

## Próxima etapa

Treinar o pipeline congelado utilizando todo o conjunto de desenvolvimento de
2023.

Em seguida, realizar uma única avaliação temporal no conjunto de 2024.

A avaliação de 2024 deverá utilizar exatamente:

- o mesmo preprocessing;
- as mesmas features;
- o mesmo algoritmo;
- os mesmos hiperparâmetros;
- threshold de risco 0,41.

Os resultados de 2024 serão tratados como teste final de generalização temporal
e não poderão ser utilizados para retuning do modelo.