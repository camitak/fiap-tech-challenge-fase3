# Decisão 006 — Seleção do modelo campeão e threshold

## Objetivo

Definir e congelar o modelo supervisionado que será utilizado na avaliação
temporal final de 2024.

Todas as decisões desta etapa foram realizadas utilizando exclusivamente
dados de 2023.

O conjunto de 2024 permaneceu isolado durante:

- análise exploratória;
- seleção de features;
- comparação de algoritmos;
- otimização de hiperparâmetros;
- validação cruzada;
- escolha do modelo;
- escolha do threshold.

## Modelos finalistas

Após a comparação inicial e a otimização de hiperparâmetros, dois modelos
foram mantidos como finalistas:

- Decision Tree;
- Random Forest.

As probabilidades utilizadas para seleção do threshold foram geradas por
validação cruzada out-of-fold com `GroupKFold`, utilizando `id_municipio`
como variável de agrupamento.

Dessa forma, cada aluno de 2023 recebeu uma probabilidade produzida por um
modelo que não havia sido treinado com registros daquele município.

## Resultados out-of-fold

| Modelo | PR-AUC risco | ROC-AUC | Threshold | Balanced Accuracy | Precision risco | Recall risco | Recall alfabetizado | F1 risco |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0,4594 | 0,5562 | 0,49 | 0,5421 | 0,4494 | 0,6640 | 0,4202 | 0,5360 |
| Random Forest | 0,4726 | 0,5738 | 0,42 | 0,5545 | 0,4646 | 0,6095 | 0,4994 | 0,5273 |

## Diagnóstico do critério baseado somente em F1

Inicialmente foi avaliada a maximização isolada do F1 da classe
`não alfabetizado`.

Esse critério produziu thresholds inadequados operacionalmente.

Na Decision Tree:

- threshold: 0,12;
- recall de risco: 99,77%;
- recall de alfabetizados: 0,15%;
- taxa predita como risco: 99,82%.

Na Random Forest:

- threshold: 0,30;
- recall de risco: 94,99%;
- recall de alfabetizados: 9,91%;
- taxa predita como risco: 92,13%.

Apesar do F1 elevado, esses thresholds classificavam praticamente toda a
população como pertencente à classe de risco.

Por esse motivo, a maximização isolada do F1 foi rejeitada como critério
oficial.

## Critério oficial do threshold

O threshold foi selecionado utilizando:

1. maior Balanced Accuracy;
2. maior Recall da classe não alfabetizado em caso de empate;
3. maior F1 da classe não alfabetizado como segundo critério de desempate.

Essa estratégia busca preservar a capacidade de identificação da população
de risco sem produzir uma classificação degenerada de praticamente toda a
população na mesma classe.

## Modelo campeão

O modelo selecionado é a **Random Forest**.

Configuração congelada:

- `n_estimators=150`;
- `max_depth=12`;
- `max_features="sqrt"`;
- `min_samples_leaf=50`;
- `min_samples_split=100`;
- `class_weight=None`;
- `random_state=42`;
- `n_jobs=-1`.

O pré-processamento utiliza o conjunto baseline de 9 features definido
durante a EDA e não utiliza `sigla_uf`.

## Threshold congelado

O threshold oficial para probabilidade de risco é:

`0.42`

A regra de classificação será:

- probabilidade de não alfabetização >= 0,42:
  classificar como `não alfabetizado`;
- probabilidade de não alfabetização < 0,42:
  classificar como `alfabetizado`.

Na validação out-of-fold de 2023 esse threshold apresentou:

- Balanced Accuracy: 0,5545;
- Precision da classe não alfabetizado: 0,4646;
- Recall da classe não alfabetizado: 0,6095;
- Recall da classe alfabetizado: 0,4994;
- F1 da classe não alfabetizado: 0,5273;
- taxa predita como risco: 54,60%.

## Justificativa da escolha

A Decision Tree apresentou Recall e F1 ligeiramente superiores para a classe
de risco.

Entretanto, a Random Forest apresentou:

- maior Balanced Accuracy;
- maior PR-AUC da classe não alfabetizado;
- maior ROC-AUC;
- maior Precision para a classe de risco;
- melhor equilíbrio entre Recall de não alfabetizados e alfabetizados.

Por esse motivo, a Random Forest foi selecionada como modelo campeão.

## Congelamento antes do teste

A partir desta decisão não serão alterados com base nos resultados de 2024:

- conjunto de features;
- pré-processamento;
- algoritmo;
- hiperparâmetros;
- threshold;
- regra de classificação.

O conjunto de 2024 será utilizado uma única vez como teste temporal final.

Qualquer desempenho inferior ao esperado em 2024 será reportado como
resultado de generalização temporal e não será utilizado para retroativamente
ajustar o modelo.