# Decisão 015: Teste temporal final em 2024

## Objetivo

Avaliar a capacidade de generalização temporal do pipeline final utilizando
2024 como conjunto de teste completamente separado do processo de
desenvolvimento.

Antes da abertura de 2024 já estavam congelados:

- conjunto de features;
- algoritmo;
- hiperparâmetros;
- threshold operacional.

Nenhuma dessas decisões foi alterada após a observação dos resultados de 2024.

## Configuração final

Modelo:

HistGradientBoostingClassifier

Feature set:

ENRICHED + UF

Número de features:

- 18 features brutas;
- 45 features após preprocessing.

Threshold de risco:

- 0,41

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

## Estratégia

O pipeline foi ajustado uma única vez utilizando todos os registros elegíveis
de 2023.

Em seguida, o modelo congelado foi utilizado para gerar probabilidades para
todos os registros elegíveis de 2024.

2024 não foi utilizado para:

- seleção de features;
- seleção de algoritmo;
- tuning de hiperparâmetros;
- escolha de threshold.

## Base de teste

2024 contém:

- 1.851.852 alunos;
- 5.517 municípios;
- 4.841 municípios também presentes em 2023;
- 676 municípios novos em relação ao desenvolvimento.

As UFs AC, DF e SP não estavam presentes no conjunto de desenvolvimento de
2023.

## Resultado global de 2024

Métricas independentes de threshold:

- PR-AUC risco: 0,52566
- ROC-AUC: 0,63218

Métricas utilizando o threshold de risco 0,41:

- Accuracy: 0,58878
- Balanced Accuracy: 0,59110
- Precision risco: 0,49083
- Recall risco: 0,60294
- F1 risco: 0,54114
- Recall alfabetizado: 0,57926
- taxa real de risco: 0,40216
- taxa sinalizada como risco: 0,49401

Matriz de confusão considerando risco como classe positiva:

- True Negative: 641.307
- False Positive: 465.812
- False Negative: 295.706
- True Positive: 449.027

## Comparação temporal

Referência OOF de 2023:

- PR-AUC risco: 0,54465
- ROC-AUC: 0,64455
- Balanced Accuracy: 0,59928
- Precision risco: 0,51025
- Recall risco: 0,62859
- F1 risco: 0,56327

Variação de 2024 em relação ao desenvolvimento:

- PR-AUC risco: -0,01900
- ROC-AUC: -0,01237
- Balanced Accuracy: -0,00819
- Precision risco: -0,01943
- Recall risco: -0,02565
- F1 risco: -0,02214

A avaliação temporal apresentou degradação moderada em relação à validação OOF
de 2023, sem colapso global do poder discriminativo.

## Municípios representados no desenvolvimento

Nos 4.841 municípios também presentes em 2023:

- PR-AUC risco: 0,54012
- ROC-AUC: 0,65151
- Balanced Accuracy: 0,60569
- Precision risco: 0,49982
- Recall risco: 0,62560
- F1 risco: 0,55568

O desempenho nesse segmento permaneceu próximo ou superior à avaliação global.

## Territórios novos em 2024

Nos 676 municípios que não estavam presentes em 2023:

- PR-AUC risco: 0,44865
- ROC-AUC: 0,55261
- Balanced Accuracy: 0,54379
- Recall risco: 0,53067
- F1 risco: 0,49266

Entretanto, 665 desses municípios pertencem às UFs AC, DF e SP, que também
estavam completamente ausentes do conjunto de desenvolvimento.

Por esse motivo, não é possível separar adequadamente o efeito de um município
novo do efeito de uma UF não representada durante o treinamento.

A conclusão correta é que existe perda de generalização em territórios fora do
domínio geográfico observado em 2023.

## UFs não vistas em 2023

### Acre

- 10.234 alunos
- 22 municípios
- Balanced Accuracy: 0,50062
- ROC-AUC: 0,50032
- PR-AUC risco: 0,49579
- aproximadamente 98,3% dos registros foram sinalizados como risco

O modelo apresentou comportamento praticamente sem capacidade discriminativa
nesse segmento.

### Distrito Federal

- 22.111 alunos
- 1 município
- Balanced Accuracy: 0,50000
- ROC-AUC: 0,50000
- 100% dos registros foram sinalizados como risco

O modelo não apresentou capacidade discriminativa nesse segmento.

### São Paulo

- 395.444 alunos
- 642 municípios
- Balanced Accuracy: 0,54599
- ROC-AUC: 0,55901
- PR-AUC risco: 0,46231
- Recall risco: 0,49161

São Paulo apresentou alguma capacidade discriminativa, porém substancialmente
inferior àquela observada nas UFs representadas no desenvolvimento.

## Desempenho por rede

### Municipal

- 1.610.754 alunos
- PR-AUC risco: 0,53535
- ROC-AUC: 0,64010
- Balanced Accuracy: 0,59587
- Recall risco: 0,63500
- F1 risco: 0,55635

### Estadual

- 241.074 alunos
- PR-AUC risco: 0,43268
- ROC-AUC: 0,56221
- Balanced Accuracy: 0,54634
- Recall risco: 0,37056
- F1 risco: 0,40401

A diferença de desempenho entre redes deve ser interpretada como resultado
segmentado do modelo, e não como evidência causal sobre qualidade ou efeito da
dependência administrativa.

### Privada

A rede privada possui apenas 24 registros, um município e uma UF em 2024.

As métricas desse grupo são exclusivamente descritivas e não devem ser
utilizadas para conclusões de generalização.

## Interpretação

O resultado confirma que o modelo possui capacidade moderada de ranking e
triagem de alunos em risco, inclusive em um teste temporal completamente
separado.

O modelo não deve ser utilizado como diagnóstico individual de alfabetização.

Sua aplicação mais defensável é como ferramenta de priorização, permitindo
direcionar atenção, investigação e recursos para alunos ou territórios com maior
probabilidade estimada de risco.

O desempenho é substancialmente mais confiável no domínio territorial
representado no treinamento.

Novas UFs ou territórios muito diferentes devem passar por validação local antes
de utilização operacional.

## Limitações

As principais limitações identificadas são:

- preditores predominantemente contextuais, enquanto o alvo é individual;
- ausência de variáveis socioeconômicas individuais;
- identificação de escola mascarada, impedindo enriquecimento escolar direto;
- contexto socioeconômico defasado em relação ao ano da avaliação;
- ausência completa de AC, DF e SP no desenvolvimento de 2023;
- desempenho heterogêneo entre redes e territórios;
- rede privada com amostra insuficiente para avaliação;
- resultados associativos, não causais.

## Decisão

O teste temporal final é considerado concluído.

O resultado de 2024 não será utilizado para retuning do modelo.

A configuração permanece congelada:

- ENRICHED + UF;
- HistGradientBoostingClassifier;
- hiperparâmetros previamente selecionados;
- threshold de risco 0,41.

O modelo será apresentado como mecanismo de inteligência educacional para
priorização de risco, e não como substituto da avaliação educacional.

## Próxima etapa

Com a modelagem preditiva encerrada, a próxima etapa será dedicada à
interpretabilidade do modelo e à transformação das previsões em inteligência
educacional.

Serão produzidas análises de:

- importância das variáveis;
- SHAP;
- fatores associados ao risco;
- ranking territorial de risco;
- apoio à identificação de municípios prioritários.

Essas análises não alterarão a configuração do modelo final.