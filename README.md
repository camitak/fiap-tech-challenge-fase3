# Tech Challenge Fase 3
## Inteligência educacional aplicada à alfabetização infantil no Brasil

Projeto desenvolvido para o **Tech Challenge da Fase 3 da Pós-Tech FIAP**, com
foco na aplicação de Machine Learning para apoiar a análise da alfabetização
infantil no Brasil.

A solução parte da arquitetura de dados construída na Fase 2 e transforma dados
educacionais, territoriais e socioeconômicos em uma camada de inteligência
analítica capaz de:

- estimar risco de não alfabetização;
- identificar fatores associados às previsões do modelo;
- priorizar municípios por risco educacional;
- identificar perfis municipais semelhantes;
- apoiar o acompanhamento de metas futuras de alfabetização.

> O objetivo não é substituir avaliações educacionais oficiais nem produzir
> diagnósticos individuais. A proposta é oferecer uma ferramenta analítica de
> triagem, priorização e apoio à tomada de decisão.

---

## Sumário

- [1. Contexto](#1-contexto)
- [2. Objetivo analítico](#2-objetivo-analítico)
- [3. Perguntas estratégicas](#3-perguntas-estratégicas)
- [4. Fontes de dados](#4-fontes-de-dados)
- [5. Construção da base analítica](#5-construção-da-base-analítica)
- [6. Prevenção de data leakage](#6-prevenção-de-data-leakage)
- [7. Análise exploratória](#7-análise-exploratória)
- [8. Features utilizadas](#8-features-utilizadas)
- [9. Estratégia de validação](#9-estratégia-de-validação)
- [10. Modelos avaliados](#10-modelos-avaliados)
- [11. Modelo final](#11-modelo-final)
- [12. Threshold operacional](#12-threshold-operacional)
- [13. Teste temporal final em 2024](#13-teste-temporal-final-em-2024)
- [14. Interpretabilidade](#14-interpretabilidade)
- [15. Inteligência municipal de risco](#15-inteligência-municipal-de-risco)
- [16. Clustering de perfis municipais](#16-clustering-de-perfis-municipais)
- [17. Risco de não atingir metas futuras](#17-risco-de-não-atingir-metas-futuras)
- [18. Principais insights](#18-principais-insights)
- [19. Aplicação em políticas públicas](#19-aplicação-em-políticas-públicas)
- [20. Limitações](#20-limitações)
- [21. Evoluções futuras](#21-evoluções-futuras)
- [22. Estrutura do repositório](#22-estrutura-do-repositório)
- [23. Reprodutibilidade](#23-reprodutibilidade)
- [24. Governança e versionamento](#24-governança-e-versionamento)
- [25. Documentação técnica](#25-documentação-técnica)

---

# 1. Contexto

A alfabetização na infância influencia toda a trajetória educacional do
estudante e constitui um dos principais indicadores de qualidade e equidade da
educação básica.

O desafio analítico não consiste apenas em calcular uma taxa de alfabetização.
É necessário integrar resultados educacionais, informações territoriais,
contexto socioeconômico, características da oferta educacional e metas públicas,
mantendo rastreabilidade e evitando vazamento de informação.

Na Fase 2 deste Tech Challenge foi construída uma arquitetura de dados na Google
Cloud para ingestão, tratamento e disponibilização dos dados nas camadas
Bronze, Silver e Gold.

A Fase 3 utiliza essa base como ponto de partida para construir modelos de
Machine Learning e produtos de inteligência educacional.

Repositório da Fase 2:

https://github.com/camitak/fiap-tech-challenge-fase2

---

# 2. Objetivo analítico

O objetivo principal é desenvolver um modelo supervisionado capaz de estimar a
probabilidade de um aluno ser classificado como:

- `1` — alfabetizado;
- `0` — não alfabetizado.

A análise utiliza variáveis educacionais, territoriais e socioeconômicas
disponíveis antes da avaliação.

Na perspectiva operacional, foi definida a variável:

```text
risco = 1 - P(alfabetizado)
```

Assim, quanto maior o `risk_score`, maior a prioridade potencial para
monitoramento.

Além da classificação individual, o projeto utiliza as previsões e os dados
contextuais para produzir inteligência no nível municipal.

---

# 3. Perguntas estratégicas

O projeto busca responder cinco perguntas principais:

1. Quais fatores e variáveis possuem maior influência nas previsões de
   alfabetização?
2. Quais municípios apresentam maior risco educacional?
3. Quais municípios e regiões apresentam perfis estruturais semelhantes?
4. Quais municípios apresentam trajetória de maior risco em relação às metas
   futuras de alfabetização?
5. Como esses resultados podem apoiar a priorização de políticas públicas?

Cada pergunta foi tratada por uma etapa analítica específica.

---

# 4. Fontes de dados

## 4.1 Avaliação de alfabetização

Fonte principal:

```text
basedosdados.br_inep_avaliacao_alfabetizacao
```

A base contém resultados individuais da avaliação de alfabetização.

Principais campos utilizados na construção analítica:

- ano;
- identificador do aluno;
- município;
- rede;
- indicador de alfabetização;
- presença;
- preenchimento do caderno;
- controles de qualidade.

A variável de proficiência não foi utilizada como predictor.

## 4.2 Dados territoriais

Foram utilizadas informações territoriais como:

- UF;
- região;
- capital;
- Amazônia Legal.

## 4.3 Contexto socioeconômico

Foram incorporados dados municipais de fontes públicas, incluindo:

- população;
- PIB;
- PIB por habitante;
- participação da agropecuária;
- participação da indústria;
- participação da administração pública.

O contexto socioeconômico utilizado para a modelagem é anterior ao período das
avaliações, preservando a lógica temporal.

## 4.4 Censo Escolar

O projeto foi enriquecido com informações do Censo Escolar.

Como o identificador de escola da avaliação é mascarado, não foi realizado join
direto escola a escola.

O enriquecimento foi construído no nível:

```text
município + rede
```

Foram utilizadas informações referentes ao período anterior ao ano-alvo:

```text
avaliação 2023 <- Censo Escolar 2022
avaliação 2024 <- Censo Escolar 2023
```

Essa estratégia reduz risco de leakage temporal.

## 4.5 Metas de alfabetização

Foram utilizadas metas municipais de alfabetização para os anos:

```text
2025
2026
2027
2028
2029
2030
```

As metas são usadas exclusivamente nas análises estratégicas posteriores à
modelagem.

Elas **não são predictors do modelo supervisionado**.

---

# 5. Construção da base analítica

Após os filtros de elegibilidade e qualidade, foram obtidos:

| Ano | Alunos elegíveis | Não alfabetizados | Alfabetizados | Taxa de alfabetização |
|---|---:|---:|---:|---:|
| 2023 | 1.502.809 | 625.382 | 877.427 | 58,39% |
| 2024 | 1.851.852 | 744.733 | 1.107.119 | 59,78% |
| **Total** | **3.354.661** | **1.370.115** | **1.984.546** | — |

Os filtros principais preservaram apenas registros considerados válidos na
camada analítica.

A base final utilizada na modelagem possui granularidade de aluno, enriquecida
com contexto territorial, socioeconômico e educacional.

---

# 6. Prevenção de data leakage

O controle de leakage foi uma das principais decisões metodológicas do projeto.

## Variáveis excluídas

### Proficiência

A proficiência possui relação direta com a definição do indicador de
alfabetização.

Utilizá-la como predictor permitiria ao modelo reconstruir praticamente a regra
que define o target.

Por esse motivo:

```text
proficiencia -> EXCLUÍDA
```

### Identificador do aluno

O identificador individual é utilizado apenas como controle de granularidade.

```text
id_aluno / aluno_key -> NÃO É PREDICTOR
```

### Município

O identificador municipal é utilizado para:

- agrupamento;
- validação;
- análises territoriais;
- joins.

Ele não é utilizado diretamente como predictor.

```text
id_municipio -> NÃO É PREDICTOR
```

### Resultado municipal do mesmo ano

Taxas agregadas observadas do município no mesmo ano da avaliação não são
utilizadas como features do aluno.

### Metas futuras

Metas de alfabetização posteriores ao ano analisado são utilizadas apenas
depois do encerramento da modelagem.

## Separação temporal

Todo o desenvolvimento do modelo foi realizado com dados de:

```text
2023
```

O conjunto de:

```text
2024
```

permaneceu fechado até o congelamento de:

- features;
- algoritmo;
- hiperparâmetros;
- threshold.

Depois da abertura de 2024 não houve retuning.

---

# 7. Análise exploratória

A análise exploratória foi dividida em duas etapas.

## EDA principal

Notebook:

```text
notebooks/01_eda.ipynb
```

Foram avaliados:

- distribuição do target;
- diferenças por ano;
- diferenças por rede;
- comportamento regional;
- distribuição municipal;
- variáveis econômicas;
- presença da Amazônia Legal;
- cobertura territorial;
- mudança entre 2023 e 2024.

## EDA do enriquecimento educacional

Notebook:

```text
notebooks/02_eda_enriquecimento_educacional.ipynb
```

Foram avaliadas as variáveis provenientes do Censo Escolar e sua redundância.

A análise identificou um bloco de variáveis de escala altamente correlacionadas,
levando à seleção de um conjunto mais compacto e interpretável de indicadores
educacionais.

---

# 8. Features utilizadas

O conjunto final foi denominado:

```text
ENRICHED + UF
```

São **18 features brutas**, transformadas em **45 features** após o
pré-processamento.

## Territoriais / administrativas

- `rede_nome`
- `regiao`
- `sigla_uf`
- `capital_uf`
- `amazonia_legal`

## Socioeconômicas

- `populacao_2020`
- `pib_por_habitante_2020`
- `participacao_agropecuaria_2020`
- `participacao_industria_2020`
- `participacao_administracao_publica_2020`

## Educacionais

- `quantidade_escolas_anos_iniciais`
- `proporcao_escolas_rurais`
- `proporcao_internet_aprendizagem`
- `proporcao_biblioteca_sala_leitura`
- `proporcao_laboratorio_informatica`
- `alunos_por_turma_anos_iniciais`
- `razao_matriculas_docentes_anos_iniciais`
- `proporcao_matriculas_integral_anos_iniciais`

---

# 9. Estratégia de validação

## Desenvolvimento

O desenvolvimento utiliza somente dados de 2023.

Foi utilizada validação cruzada:

```text
GroupKFold
```

com:

```text
group = id_municipio
```

Essa decisão impede que registros de um mesmo município apareçam
simultaneamente nos folds de treino e validação.

Foram utilizados cinco folds.

## Teste final

Depois do congelamento da configuração:

```text
treino final = todo 2023
teste final  = todo 2024
```

Isso permite avaliar generalização temporal em um conjunto não utilizado durante
a construção do modelo.

---

# 10. Modelos avaliados

Foram avaliadas diferentes famílias de modelos.

## Logistic Regression

Utilizada como baseline linear.

O modelo apresentou baixa capacidade de capturar relações não lineares entre os
atributos contextuais.

## Decision Tree

Apresentou melhoria em relação ao baseline, especialmente em recall da classe de
risco, mas menor capacidade de ranking e maior sensibilidade à estrutura da
árvore.

## Random Forest

Apresentou ganho relevante em PR-AUC e ROC-AUC e foi levado à etapa de tuning.

## HistGradientBoostingClassifier

O HistGradientBoosting apresentou desempenho ligeiramente superior ao Random
Forest após tuning, com:

- maior PR-AUC;
- maior ROC-AUC;
- maior Balanced Accuracy;
- maior recall da classe de risco;
- menor gap de generalização;
- custo computacional significativamente menor.

Por esse conjunto de fatores, foi selecionado como modelo final.

---

# 11. Modelo final

Modelo:

```text
HistGradientBoostingClassifier
```

Hiperparâmetros finais:

```python
learning_rate = 0.06197763962309192
max_iter = 230
max_leaf_nodes = 15
max_depth = None
min_samples_leaf = 70
l2_regularization = 0.716404042819101
max_bins = 255
early_stopping = False
random_state = 42
```

O tuning foi realizado somente utilizando 2023.

A métrica principal de seleção foi:

```text
PR-AUC da classe de risco
```

O ROC-AUC foi utilizado como métrica complementar.

---

# 12. Threshold operacional

A probabilidade produzida pelo modelo foi convertida para a perspectiva de risco:

```text
p_risco = 1 - p_alfabetizado
```

O threshold foi selecionado exclusivamente utilizando previsões
**out-of-fold de 2023**.

Threshold final:

```text
0,41
```

A seleção priorizou Balanced Accuracy, observando também:

- recall de risco;
- precision;
- F1;
- proporção de alunos sinalizados.

O threshold foi congelado antes da abertura do teste de 2024.

---

# 13. Teste temporal final em 2024

O pipeline final foi treinado com todo o conjunto de 2023 e aplicado uma única
vez ao conjunto de 2024.

## Métricas globais

| Métrica | Resultado 2024 |
|---|---:|
| PR-AUC — risco | **0,5257** |
| ROC-AUC | **0,6322** |
| Balanced Accuracy | **0,5911** |
| Accuracy | 0,5888 |
| Precision — risco | 0,4908 |
| Recall — risco | **0,6029** |
| F1 — risco | **0,5411** |
| Recall — alfabetizado | 0,5793 |

A prevalência real da classe de risco em 2024 foi aproximadamente:

```text
40,22%
```

A taxa de alunos sinalizados pelo threshold foi aproximadamente:

```text
49,40%
```

## Matriz de confusão

Considerando risco como classe positiva:

```text
True Negative     641.307
False Positive    465.812
False Negative    295.706
True Positive     449.027
```

Dos 744.733 alunos realmente classificados como não alfabetizados em 2024, o
modelo identificou aproximadamente 60,3%.

## Comparação com desenvolvimento

O desempenho em 2024 apresentou queda moderada em relação à validação OOF de
2023.

| Métrica | 2023 OOF | 2024 |
|---|---:|---:|
| PR-AUC risco | 0,5447 | 0,5257 |
| ROC-AUC | 0,6446 | 0,6322 |
| Balanced Accuracy | 0,5993 | 0,5911 |
| Recall risco | 0,6286 | 0,6029 |
| F1 risco | 0,5633 | 0,5411 |

A degradação temporal existe, mas não representa colapso global do modelo.

## Generalização territorial

Nos municípios já representados durante o desenvolvimento, o desempenho foi
superior ao observado em territórios novos.

As UFs:

```text
AC
DF
SP
```

não estavam presentes no conjunto de desenvolvimento de 2023.

O modelo apresentou queda relevante de capacidade de generalização nesses
territórios.

Por esse motivo, previsões em novas UFs devem ser submetidas a validação local
antes de utilização operacional.

---

# 14. Interpretabilidade

Foram utilizadas duas técnicas complementares:

```text
Permutation Importance
SHAP Values
```

## 14.1 Permutation Importance

A Permutation Importance mede quanto o desempenho piora quando a informação de
uma variável é destruída.

A métrica principal foi:

```text
queda no PR-AUC da classe de risco
```

Principais variáveis:

| Rank | Feature | Queda média no PR-AUC |
|---:|---|---:|
| 1 | `sigla_uf` | **0,0735** |
| 2 | `participacao_administracao_publica_2020` | 0,0071 |
| 3 | `participacao_agropecuaria_2020` | 0,0053 |
| 4 | `quantidade_escolas_anos_iniciais` | 0,0051 |
| 5 | `populacao_2020` | 0,0042 |

A UF apresentou importância muito superior às demais features, evidenciando forte
dependência territorial.

Visualização:

```text
images/hgb_permutation_importance_2024.png
```

## 14.2 SHAP Values

O SHAP foi utilizado para analisar magnitude e direção das contribuições.

A interpretação foi convertida para a perspectiva de risco:

```text
SHAP risco > 0
    -> empurra a previsão para maior risco

SHAP risco < 0
    -> empurra a previsão para maior alfabetização
```

Principais variáveis por magnitude média de SHAP:

| Rank | Feature | Mean \|SHAP risco\| |
|---:|---|---:|
| 1 | `sigla_uf` | **0,2324** |
| 2 | `regiao` | 0,0799 |
| 3 | `proporcao_internet_aprendizagem` | 0,0722 |
| 4 | `participacao_administracao_publica_2020` | 0,0529 |
| 5 | `participacao_agropecuaria_2020` | 0,0481 |

## Padrões educacionais encontrados pelo SHAP

Dentro do comportamento aprendido pelo modelo:

- maior disponibilidade de internet para aprendizagem aparece associada a menor
  risco previsto;
- maior disponibilidade de biblioteca ou sala de leitura aparece associada a
  menor risco;
- turmas maiores aparecem associadas a maior risco;
- maior razão matrículas/docentes aparece associada a maior risco;
- altos níveis de matrículas em tempo integral aparecem associados a menor
  risco.

Esses padrões são **associativos e preditivos**.

Não representam evidência causal.

Visualizações:

```text
images/hgb_shap_global_importance_2024.png
images/hgb_shap_beeswarm_2024.png
```

---

# 15. Inteligência municipal de risco

Depois do congelamento do modelo, as previsões foram agregadas no nível
municipal.

Foram construídas duas dimensões de priorização.

## 15.1 Severidade

Indicador:

```text
probabilidade média prevista de risco
```

Pergunta:

> Em quais municípios a concentração prevista de risco é maior?

O ranking executivo considera:

- municípios dentro do domínio territorial mais confiável;
- mínimo de 100 alunos avaliados.

Entre os maiores riscos previstos aparecem:

1. Nossa Senhora do Socorro / SE
2. Pacatuba / SE
3. Canindé de São Francisco / SE
4. Riachão do Dantas / SE
5. Porto da Folha / SE

A concentração do ranking em determinadas UFs é coerente com a forte dependência
territorial identificada na interpretabilidade.

## 15.2 Carga estimada

Indicador:

```text
carga_risco_estimada =
n_alunos * probabilidade_media_risco
```

A carga representa a soma esperada das probabilidades individuais e não uma
contagem observada de alunos não alfabetizados.

Pergunta:

> Em quais municípios pode estar o maior volume absoluto de alunos em risco?

Principais prioridades por carga:

1. Rio de Janeiro / RJ
2. Manaus / AM
3. Salvador / BA
4. Belo Horizonte / MG
5. Campo Grande / MS

Severidade e carga respondem a perguntas diferentes e, portanto, foram mantidas
como indicadores separados.

Visualização:

```text
images/municipal_risk_top20_2024.png
```

---

# 16. Clustering de perfis municipais

Para identificar municípios estruturalmente semelhantes foi utilizada
aprendizagem não supervisionada.

Algoritmo:

```text
K-Means
```

O clustering **não utilizou**:

- alfabetização;
- risco previsto;
- metas;
- UF;
- região;
- identificador municipal.

Foram utilizadas somente características socioeconômicas e educacionais.

## Escolha de K

Foram avaliados valores de:

```text
K = 2 ... 10
```

utilizando:

- Silhouette Score;
- Davies-Bouldin Index;
- inertia;
- Calinski-Harabasz;
- tamanho dos grupos;
- interpretabilidade.

Configuração final:

```text
K = 3
```

Métricas:

| Métrica | Resultado |
|---|---:|
| Silhouette | **0,2034** |
| Davies-Bouldin | **1,6429** |
| Calinski-Harabasz | **1413,89** |
| Inertia | **47.408,33** |

A reprodução final apresentou:

```text
Adjusted Rand Index = 1,0
```

em relação à partição candidata de K=3.

## Perfil 0 — Urbano-industrial de maior escala

Principais características:

- maior população;
- maior PIB por habitante;
- forte participação industrial;
- menor participação agropecuária;
- menor proporção de escolas rurais;
- infraestrutura educacional acima da média.

Quantidade:

```text
1.374 municípios
```

## Perfil 1 — Pequeno porte agropecuário com maior disponibilidade de infraestrutura educacional

Principais características:

- municípios menores;
- maior participação agropecuária;
- maior disponibilidade relativa de internet;
- maior presença de biblioteca/sala de leitura;
- maior presença de laboratório de informática;
- menos alunos por turma;
- menor razão matrículas/docentes.

Quantidade:

```text
1.985 municípios
```

## Perfil 2 — Rural, menor renda e maior presença da administração pública

Principais características:

- PIB por habitante inferior;
- maior presença da administração pública na economia;
- maior proporção de escolas rurais;
- menor disponibilidade de internet;
- menor disponibilidade de biblioteca;
- menor disponibilidade de laboratório de informática;
- maior razão matrículas/docentes.

Quantidade:

```text
2.158 municípios
```

## Padrões regionais

Região e UF não participaram da formação dos clusters.

Mesmo assim, depois da clusterização surgiram padrões territoriais:

- Nordeste -> forte predominância do Perfil 2;
- Norte -> predominância do Perfil 2;
- Sul -> predominância do Perfil 1;
- Centro-Oeste -> predominância do Perfil 1;
- Sudeste -> distribuição mais heterogênea entre Perfis 0 e 1.

O resultado demonstra que políticas podem ser pensadas também por perfil
estrutural, e não apenas por fronteira administrativa.

Visualizações:

```text
images/kmeans_final_pca_2024.png
images/kmeans_final_profile_heatmap_2024.png
```

---

# 17. Risco de não atingir metas futuras

A última análise estratégica avalia quais municípios apresentam trajetória de
maior risco em relação às metas futuras.

## Por que não foi utilizado forecasting tradicional?

Existem somente dois resultados municipais observados:

```text
2023
2024
```

Dois pontos não são suficientes para estimar de forma robusta:

- tendência de longo prazo;
- sazonalidade;
- autocorrelação;
- ARIMA;
- SARIMA;
- Prophet;
- forecasting multianual confiável.

Por esse motivo, foi construída uma:

```text
triagem de trajetória
```

e não um modelo de série temporal.

## Método

Para cada município:

```text
ganho recente =
taxa_2024 - taxa_2023
```

É calculado o ganho anual necessário para atingir cada meta.

Também é construído um cenário de continuidade:

```text
cenario_bruto =
taxa_2024 + ganho_recente * horizonte
```

O cenário bruto não é interpretado como taxa literal.

O indicador principal é:

```text
deficit_trajetoria =
meta - cenario_bruto
```

Valores positivos indicam trajetória recente insuficiente para atingir a meta.

## Resultado para 2025

Universo:

```text
5.352 municípios
```

Com histórico completo:

```text
5.232
```

Sem resultado 2023:

```text
120
```

Entre os municípios com histórico:

| Situação | Municípios |
|---|---:|
| Abaixo da meta e em risco | 2.450 |
| Meta atingida, mas trajetória em risco | 82 |
| Abaixo da meta, mas em trajetória suficiente | 474 |
| Meta atingida e trajetória sustentável | 2.226 |

Total sinalizado:

```text
2.532 municípios
```

ou aproximadamente:

```text
48,4%
```

dos municípios com histórico.

## Distribuição regional dos alertas de 2025

| Região | Proporção sinalizada |
|---|---:|
| Sul | **68,9%** |
| Norte | 56,6% |
| Nordeste | 51,3% |
| Sudeste | 35,3% |
| Centro-Oeste | **28,9%** |

Essa análise responde a uma pergunta diferente da classificação individual.

Um território pode possuir risco contextual relativamente baixo, mas apresentar
deterioração recente incompatível com sua meta planejada.

## Horizonte 2025–2030

A proporção de municípios sinalizados permanece próxima de 50%.

Entretanto, o déficit médio aumenta fortemente conforme a extrapolação é
estendida.

Por esse motivo:

```text
2025
-> horizonte operacional principal

2026–2030
-> cenários de planejamento / stress de trajetória
```

Os resultados futuros não são apresentados como forecasts determinísticos.

Visualizações:

```text
images/future_target_risk_top20_2025.png
images/future_target_risk_horizon_2025_2030.png
```

---

# 18. Principais insights

## 1. O território é extremamente relevante para o modelo

UF foi a variável dominante tanto em Permutation Importance quanto em SHAP.

Ao mesmo tempo, a perda de desempenho em UFs ausentes no treinamento demonstra
que essa dependência territorial limita a generalização.

## 2. Recursos educacionais aparecem associados ao risco previsto

O modelo encontrou padrões relacionados a:

- internet para aprendizagem;
- biblioteca ou sala de leitura;
- laboratório de informática;
- alunos por turma;
- razão matrículas/docentes;
- matrícula em tempo integral.

Essas relações representam associação preditiva, não causalidade.

## 3. Severidade e escala são problemas diferentes

Municípios com maior concentração prevista de risco não são necessariamente os
que concentram o maior volume absoluto de alunos potencialmente em risco.

Por isso são mantidas duas visões:

```text
severidade
+
carga
```

## 4. Municípios estruturalmente semelhantes atravessam fronteiras regionais

O clustering mostrou que municípios de regiões diferentes podem compartilhar
características socioeconômicas e educacionais semelhantes.

Isso possibilita pensar políticas por perfil estrutural.

## 5. Risco educacional e risco de meta são conceitos distintos

O modelo supervisionado responde:

> Qual a probabilidade contextual de um aluno estar em risco de não
> alfabetização?

A análise de metas responde:

> A trajetória recente do município é compatível com o benchmark futuro?

Os resultados são complementares.

---

# 19. Aplicação em políticas públicas

A solução pode apoiar gestores em diferentes níveis.

## Priorização de territórios

O ranking municipal pode ajudar a identificar localidades para:

- diagnóstico adicional;
- acompanhamento pedagógico;
- priorização de visitas técnicas;
- investigação de condições locais.

## Priorização por escala

A carga estimada permite identificar municípios onde uma política pode alcançar
maior volume potencial de alunos em risco.

## Segmentação por perfil estrutural

Os clusters permitem desenhar estratégias diferentes para:

- municípios rurais;
- municípios pequenos agropecuários;
- municípios urbanos e industriais.

Uma política única para todos os municípios pode não ser adequada a estruturas
muito distintas.

## Monitoramento de metas

A triagem de trajetória pode ser utilizada para:

- sinalizar deteriorações;
- acompanhar municípios abaixo da trajetória necessária;
- direcionar investigação para mudanças abruptas;
- priorizar acompanhamento anual.

## Uso responsável

O modelo não deve ser utilizado para:

- negar recursos;
- rotular individualmente estudantes;
- substituir avaliações oficiais;
- produzir decisões automáticas sem validação humana.

O uso recomendado é como ferramenta complementar para:

```text
triagem
priorização
investigação
monitoramento
```

---

# 20. Limitações

## Natureza contextual das features

O target é individual, mas grande parte dos predictors representa contexto
municipal ou município-rede.

Isso impõe um limite natural à capacidade de discriminação individual.

## Ausência de variáveis socioeconômicas individuais

Não estão disponíveis informações individuais como:

- renda familiar;
- escolaridade dos responsáveis;
- características domiciliares;
- acesso individual a recursos educacionais.

## Escola mascarada

O identificador de escola da avaliação é mascarado.

Isso impede enriquecimento direto no nível da escola com o Censo Escolar.

## Defasagem socioeconômica

Parte do contexto socioeconômico disponível é de 2020.

Essas variáveis representam características estruturais, mas não capturam
integralmente mudanças econômicas mais recentes.

## Dependência territorial

O modelo depende fortemente de UF e região.

Essa característica melhora a discriminação dentro do domínio representado, mas
reduz a segurança da generalização para territórios completamente novos.

## UFs ausentes no desenvolvimento

AC, DF e SP não estavam presentes em 2023.

O desempenho nesses territórios foi substancialmente inferior.

## Métricas moderadas

O modelo não possui poder discriminativo próximo de 90%.

Isso é coerente com a natureza contextual das features disponíveis.

Uma acurácia extremamente alta neste problema, utilizando somente essas
informações, exigiria auditoria cuidadosa de leakage.

## Clustering

O Silhouette em torno de 0,20 mostra que os clusters não são classes naturais
perfeitamente separadas.

São segmentos analíticos úteis, com sobreposição entre municípios
intermediários.

## Metas futuras

A análise de trajetória possui apenas dois anos observados.

Portanto:

- não constitui forecast robusto;
- é sensível a variações anuais;
- pode ser influenciada por coortes pequenas;
- horizontes longos possuem incerteza elevada.

## Causalidade

Nenhuma análise deste projeto permite concluir causalidade.

Expressões adequadas:

```text
"o modelo associa"
"há contribuição preditiva"
"o padrão observado é compatível com"
```

Expressões que devem ser evitadas:

```text
"essa variável causa"
"essa política produz"
"essa característica determina"
```

---

# 21. Evoluções futuras

## Validação leave-one-state-out

Uma evolução importante seria avaliar explicitamente a generalização para UFs
completamente ausentes do treinamento.

## Modelo menos dependente de UF

Comparar:

```text
modelo com UF explícita
versus
modelo sem UF explícita
```

utilizando maior riqueza de atributos estruturais.

## Dados socioeconômicos mais recentes

Atualizar população e contexto econômico com dados temporalmente mais próximos
das avaliações.

## Variáveis individuais adicionais

Caso se tornem disponíveis de forma ética e devidamente anonimizada, atributos
individuais poderiam melhorar a discriminação.

## Série temporal mais longa

Com vários anos adicionais de resultados municipais seria possível avaliar:

- tendências;
- modelos de séries temporais;
- backtesting;
- intervalos de previsão;
- forecasting de metas.

## Validação local

Antes de uso operacional, especialmente em novas UFs, recomenda-se:

- validação regional;
- análise de calibração;
- auditoria de performance;
- monitoramento de drift;
- revisão por especialistas em educação.

---

# 22. Estrutura do repositório

A estrutura principal do projeto é:

```text
fiap-tech-challenge-fase3/
│
├── data/
│   ├── modeling/
│   │   ├── README.md
│   │   └── base_modelagem/
│   └── reference/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_eda_enriquecimento_educacional.ipynb
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py
│   │
│   ├── modeling/
│   │   ├── compare_feature_sets.py
│   │   ├── compare_uf_feature.py
│   │   ├── compare_hist_gradient_boosting.py
│   │   ├── tune_hgb_enriched_uf.py
│   │   └── tune_random_forest_enriched_uf.py
│   │
│   ├── evaluation/
│   │   ├── permutation_importance_final.py
│   │   ├── shap_final.py
│   │   ├── municipal_risk_intelligence.py
│   │   └── future_target_risk.py
│   │
│   └── clustering/
│       ├── evaluate_kmeans_profiles.py
│       └── finalize_kmeans_profiles.py
│
├── sql/
│   └── modeling/
│
├── reports/
│   ├── metrics/
│   ├── business/
│   ├── clustering/
│   └── decisions/
│
├── images/
│
├── requirements.txt
├── README.md
└── .gitignore
```

Arquivos de dados de grande volume não precisam ser versionados no Git.

Os scripts e a documentação descrevem como reconstruir os artefatos.

---

# 23. Reprodutibilidade

## 23.1 Ambiente de referência

Os resultados finais deste projeto foram gerados no seguinte ambiente:

```text
Python 3.13.9
numpy 2.3.5
pandas 2.3.3
scipy 1.16.3
scikit-learn 1.7.2
matplotlib 3.10.6
pyarrow 21.0.0
shap 0.52.0
jupyter-core 5.8.1
```

As versões utilizadas estão registradas em:

```text
requirements.txt
```

---

## 23.2 Clonar o repositório

```bash
git clone git@github.com:camitak/fiap-tech-challenge-fase3.git
cd fiap-tech-challenge-fase3
```

---

## 23.3 Criar ambiente virtual

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 23.4 Instalar dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

O `requirements.txt` contém as dependências diretas utilizadas pelos scripts do
projeto.

Os notebooks podem ser abertos em ambientes como VS Code, Jupyter ou Google
Colab. A interface de notebook não é necessária para executar os módulos Python
do pipeline.

---

## 23.5 Disponibilizar a base de modelagem

A base de modelagem deve seguir a estrutura:

```text
data/modeling/base_modelagem/v2/
├── ano=2023/
│   └── *.parquet
└── ano=2024/
    └── *.parquet
```

A base é derivada da camada Gold construída a partir da arquitetura da Fase 2 e
do enriquecimento utilizado na Fase 3.

Os arquivos de grande volume não são versionados no Git.

---

## 23.6 Validar a base

Antes de executar a modelagem:

```bash
python -m src.preprocessing.validate_modeling_data_v2
```

Também pode ser executado o smoke test do pré-processamento:

```bash
python -m src.preprocessing.smoke_test_preprocessor_v2
```

---

## 23.7 Análise exploratória

Notebooks:

```text
notebooks/01_eda.ipynb
notebooks/02_eda_enriquecimento_educacional.ipynb
```

---

## 23.8 Comparação de features

```bash
python -m src.modeling.compare_feature_sets
python -m src.modeling.compare_uf_feature
```

---

## 23.9 Comparação de famílias de modelos

```bash
python -m src.modeling.compare_models
python -m src.modeling.compare_hist_gradient_boosting
```

---

## 23.10 Tuning

HistGradientBoosting:

```bash
python -m src.modeling.tune_hgb_enriched_uf
```

Random Forest:

```bash
python -m src.modeling.tune_random_forest_enriched_uf
```

Os experimentos utilizam validação agrupada por município e dados de 2023.

---

## 23.11 Seleção do threshold final

Depois do congelamento do modelo:

```bash
python -m src.modeling.select_final_threshold_hgb
```

O threshold é selecionado exclusivamente a partir de previsões out-of-fold de
2023.

Threshold final utilizado:

```text
risco >= 0,41
```

---

## 23.12 Avaliação temporal final

Somente após congelar features, modelo, hiperparâmetros e threshold:

```bash
python -m src.modeling.evaluate_final_2024
```

O conjunto de 2024 é utilizado como teste temporal final.

Depois dessa avaliação não é realizado retuning.

---

## 23.13 Interpretabilidade

Permutation Importance:

```bash
python -m src.evaluation.permutation_importance_final
```

SHAP:

```bash
python -m src.evaluation.shap_final
```

---

## 23.14 Inteligência municipal

```bash
python -m src.evaluation.municipal_risk_intelligence
```

Essa etapa gera análises de:

- severidade média prevista;
- carga estimada de risco;
- rankings municipais;
- agregações territoriais.

---

## 23.15 Clustering municipal

Avaliação de candidatos:

```bash
python -m src.clustering.evaluate_kmeans_profiles
```

Clustering final:

```bash
python -m src.clustering.finalize_kmeans_profiles
```

O resultado final utiliza `K=3`.

---

## 23.16 Triagem de metas futuras

Disponibilizar localmente:

```text
data/reference/meta_alfabetizacao_municipio.csv
```

Em seguida:

```bash
python -m src.evaluation.future_target_risk
```

A análise utiliza os resultados observados de 2023 e 2024 e as metas municipais
de 2025 a 2030.

Ela deve ser interpretada como triagem de trajetória, e não como forecast
determinístico.

---

## 23.17 Smoke test do ambiente

Uma verificação mínima do ambiente pode ser executada com:

```bash
python -c "import numpy,pandas,scipy,sklearn,matplotlib,pyarrow,shap; print('imports OK'); from src.preprocessing.preprocessor import get_model_features,build_preprocessor; f=get_model_features(feature_set='enriched',include_uf=True); assert len(f)==18; build_preprocessor(feature_set='enriched',include_uf=True); print('preprocessing OK | features=',len(f))"
```

Resultado esperado:

```text
imports OK
preprocessing OK | features= 18
```

Também é possível verificar a compilação dos módulos:

```bash
python -m compileall -q src
```

A ausência de mensagens de erro indica sucesso.

---
# 24. Governança e versionamento

O desenvolvimento utilizou fluxo baseado em Git com:

- commits incrementais;
- branches por objetivo;
- pull requests;
- revisão antes de merge;
- documentação das decisões analíticas.

Exemplos de blocos desenvolvidos separadamente:

```text
modelagem supervisionada
clustering municipal
risco de metas futuras
documentação final
```

As principais decisões metodológicas estão registradas em:

```text
reports/decisions/
```

Entre elas:

- seleção de features;
- inclusão de UF;
- comparação entre famílias de modelos;
- tuning;
- seleção do modelo final;
- threshold;
- teste temporal;
- interpretabilidade;
- risco municipal;
- clustering;
- triagem de metas.

Essa documentação permite rastrear por que cada decisão foi tomada.

---

# 25. Documentação técnica

Os artefatos estão organizados em três grandes grupos.

## Métricas de modelagem

```text
reports/metrics/
```

Contém:

- resultados de validação;
- tuning;
- threshold;
- teste final;
- Permutation Importance;
- SHAP.

## Inteligência educacional

```text
reports/business/
```

Contém:

- rankings municipais;
- severidade;
- carga estimada;
- análise regional;
- triagem de metas.

## Aprendizagem não supervisionada

```text
reports/clustering/
```

Contém:

- avaliação de K;
- perfis municipais;
- labels;
- composição regional;
- composição por UF;
- perfis finais dos clusters.

## Decisões analíticas

```text
reports/decisions/
```

Os arquivos registram:

- objetivo da decisão;
- evidências observadas;
- alternativas consideradas;
- limitações;
- escolha realizada;
- consequência para as etapas seguintes.

---

# Conclusão

O projeto construiu uma solução de inteligência educacional que vai além de uma
métrica única de classificação.

O pipeline combina:

```text
engenharia de dados
        ↓
EDA
        ↓
modelagem supervisionada
        ↓
validação temporal
        ↓
interpretabilidade
        ↓
priorização territorial
        ↓
clustering
        ↓
monitoramento de metas
```

O HistGradientBoosting final apresenta capacidade moderada de discriminação e
deve ser interpretado como ferramenta de triagem, especialmente dentro do
domínio territorial representado no treinamento.

A principal contribuição da solução está na integração entre previsão,
explicabilidade e inteligência territorial, permitindo transformar dados
públicos em informações úteis para priorização, monitoramento e investigação
educacional.

O projeto demonstra também uma preocupação central com o uso responsável de
Machine Learning: performance é analisada juntamente com generalização,
interpretabilidade, limitações e adequação ao contexto de negócio.
