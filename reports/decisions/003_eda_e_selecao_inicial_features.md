# Decisão 003 — EDA e seleção inicial de features

## Objetivo

Esta etapa consolida os principais resultados da Análise Exploratória de Dados
realizada sobre a `alfabetizacao_gold.gold_ml_aluno` e registra as decisões
iniciais de seleção de variáveis para a modelagem supervisionada.

A EDA foi utilizada para avaliar distribuições, diferenças territoriais,
comportamento temporal, relações entre variáveis e possíveis problemas para
generalização do modelo.

As associações encontradas nesta etapa são exploratórias e não devem ser
interpretadas como relações causais.

---

## 1. Distribuição do target

A população de modelagem apresentou a seguinte distribuição:

### 2023

- Não alfabetizados: 625.382 (41,61%)
- Alfabetizados: 877.427 (58,39%)

### 2024

- Não alfabetizados: 744.733 (40,22%)
- Alfabetizados: 1.107.119 (59,78%)

A diferença entre as classes não foi considerada suficientemente extrema para
justificar técnicas de oversampling ou SMOTE no início da modelagem.

A estratégia inicial será treinar os modelos sem reamostragem e avaliar
separadamente as métricas das duas classes.

---

## 2. Série escolar

Todos os registros elegíveis pertencem ao 2º ano do Ensino Fundamental.

### 2023

- 1.502.809 registros
- 100% no 2º ano

### 2024

- 1.851.852 registros
- 100% no 2º ano

Por apresentar variância zero, `serie_codigo` e `serie_nome` não serão
utilizados como features.

---

## 3. Rede de ensino

Em 2023, a população contém somente alunos das redes Municipal e Estadual.

Distribuição de 2023:

- Municipal: 91,25%
- Estadual: 8,75%

Em 2024:

- Municipal: 86,98%
- Estadual: 13,02%
- Privada: 24 registros

Os 24 registros da rede privada pertencem a uma única escola mascarada do
município de Gurupi, Tocantins.

Essa categoria não será removida do conjunto de teste, pois faz parte da
população válida observada em 2024.

Como a categoria `Privada` não existe no conjunto de treino de 2023, o
pré-processamento categórico deverá aceitar categorias desconhecidas no teste.

Será utilizado:

`OneHotEncoder(handle_unknown="ignore")`

A performance desses 24 registros não será interpretada isoladamente como
evidência sobre a rede privada devido ao tamanho extremamente reduzido da
amostra.

Entre `rede_codigo`, `rede_nome` e `rede_agrupada`, será utilizada inicialmente
apenas `rede_nome`.

Motivos:

- `rede_codigo` representa a mesma informação de forma codificada;
- `rede_agrupada` reduz Estadual e Municipal à mesma categoria Pública;
- `rede_nome` preserva maior informação sem utilizar códigos numéricos
  arbitrários.

---

## 4. Diferenças regionais

Foram observadas diferenças importantes entre as taxas de alfabetização das
regiões.

### 2023

- Sul: 67,73%
- Sudeste: 59,12%
- Centro-Oeste: 59,01%
- Nordeste: 54,73%
- Norte: 51,11%

### 2024

- Centro-Oeste: 64,70%
- Sudeste: 62,29%
- Sul: 60,97%
- Nordeste: 56,86%
- Norte: 50,76%

A ordenação das regiões muda entre os anos.

Portanto, a análise não sustenta uma conclusão fixa de que determinada região
possui sempre melhor resultado.

O resultado indica que existe componente territorial relevante, mas seu efeito
pode variar temporalmente e interagir com outras características.

`regiao` será mantida como feature candidata.

---

## 5. Rede e território

A relação entre rede de ensino e alfabetização também varia entre regiões.

No resultado nacional agregado, a rede Estadual apresenta taxa de alfabetização
superior à Municipal em 2023 e 2024.

Entretanto, ao analisar rede e região conjuntamente, essa relação não se mantém
em todas as regiões.

Isso sugere possível interação entre contexto territorial e rede de ensino.

Neste momento não será criada manualmente uma variável `regiao_rede`.

Primeiro serão avaliados modelos lineares e não lineares com as variáveis
originais. Modelos baseados em árvores poderão capturar interações sem a
necessidade de construí-las manualmente.

---

## 6. Cobertura territorial

A Gold contém dados de 26 Unidades da Federação no conjunto completo.

Roraima não possui registros em nenhum dos dois anos.

Além disso:

- Acre não possui registros em 2023;
- Distrito Federal não possui registros em 2023;
- São Paulo não possui registros em 2023;
- os três aparecem em 2024.

A base utilizada não permite concluir a causa dessas ausências.

Essa diferença será documentada como limitação de cobertura dos dados.

---

## 7. Cobertura municipal entre os anos

Foram identificados 5.547 municípios no universo combinado.

Distribuição:

- presentes em 2023 e 2024: 4.841;
- somente em 2023: 30;
- somente em 2024: 676.

Portanto, o teste temporal de 2024 contém um número relevante de municípios
que não aparecem no conjunto de desenvolvimento de 2023.

Essa característica será preservada porque permite avaliar a capacidade de
generalização do modelo para novos contextos territoriais.

Na avaliação final serão reportadas métricas para:

1. todo o conjunto de 2024;
2. municípios presentes também em 2023;
3. municípios que aparecem somente em 2024.

`id_municipio` continuará sendo usado apenas para agrupamento e avaliação e não
será fornecido diretamente como feature ao modelo.

---

## 8. Estabilidade temporal municipal

Entre os 4.841 municípios presentes nos dois anos, a correlação entre as taxas
municipais de alfabetização de 2023 e 2024 foi aproximadamente 0,644.

A variação absoluta foi:

- média: aproximadamente 12,53 pontos percentuais;
- mediana: aproximadamente 9,43 pontos percentuais.

A variação entre 2023 e 2024 apresentou:

- primeiro quartil: aproximadamente -6,46 pontos percentuais;
- mediana: aproximadamente +2,11 pontos percentuais;
- terceiro quartil: aproximadamente +11,83 pontos percentuais.

O resultado mostra persistência territorial moderada, mas também mudanças
substanciais entre anos.

Por esse motivo, indicadores de resultado municipal do ano anterior não serão
introduzidos no modelo principal neste momento.

Além do risco metodológico de trabalhar com resultados muito próximos ao
target, não existe histórico anterior equivalente disponível para construir a
mesma feature de forma simétrica para o treinamento de 2023.

---

## 9. Distribuição das variáveis socioeconômicas

A distribuição municipal de população é fortemente assimétrica.

Resumo de população em 2020:

- mínimo: 776
- Q1: 5.442
- mediana: 11.663
- Q3: 25.681
- máximo: 12.325.232

A distribuição de PIB por habitante também apresenta forte assimetria:

- mínimo: aproximadamente 4.920,62
- Q1: aproximadamente 11.526,53
- mediana: aproximadamente 20.141,47
- Q3: aproximadamente 33.925,76
- máximo: aproximadamente 590.594,94

Por esse motivo, serão avaliadas transformações logarítmicas para:

- `populacao_2020`;
- `pib_por_habitante_2020`.

A transformação deverá fazer parte do pipeline e ser reproduzível.

A hipótese inicial é utilizar `log1p` para reduzir a influência das caudas das
distribuições nos modelos lineares.

---

## 10. Estrutura econômica

As quatro participações econômicas somam aproximadamente 1:

- agropecuária;
- indústria;
- serviços;
- administração pública.

Usar simultaneamente as quatro variáveis em um modelo linear introduz
redundância praticamente perfeita.

Para o baseline linear, `participacao_servicos_2020` será tratada como
componente de referência e inicialmente excluída.

Serão mantidas:

- `participacao_agropecuaria_2020`;
- `participacao_industria_2020`;
- `participacao_administracao_publica_2020`.

A exclusão de serviços não significa que a variável seja considerada
irrelevante. A decisão é apenas uma forma de evitar redundância composicional
no modelo linear.

Essa decisão poderá ser reavaliada para modelos baseados em árvores.

---

## 11. Correlações socioeconômicas

As correlações entre características municipais e a taxa de alfabetização
foram baixas ou moderadas.

### 2023

- população: -0,041;
- PIB por habitante: 0,146;
- agropecuária: 0,170;
- indústria: 0,066;
- serviços: 0,079;
- administração pública: -0,281.

### 2024

- população: -0,040;
- PIB por habitante: 0,068;
- agropecuária: 0,111;
- indústria: 0,031;
- serviços: 0,035;
- administração pública: -0,163.

Nenhuma variável socioeconômica isolada apresenta correlação linear forte com
a alfabetização.

A participação da administração pública apresenta a maior associação em valor
absoluto, negativa nos dois anos.

Esses resultados representam associação estatística e não causalidade.

---

## 12. Capital e Amazônia Legal

As capitais apresentaram taxa de alfabetização menor que os demais municípios
nos dois anos.

### 2023

- não capital: 58,88%;
- capital: 55,77%.

### 2024

- não capital: 60,41%;
- capital: 57,11%.

O resultado é contraintuitivo e não será interpretado causalmente.

`capital_uf` permanecerá como variável candidata.

A Amazônia Legal apresentou diferença mais estável:

### 2023

- fora da Amazônia Legal: 59,67%;
- Amazônia Legal: 52,87%.

### 2024

- fora da Amazônia Legal: 60,83%;
- Amazônia Legal: 54,05%.

A diferença é próxima de 6,8 pontos percentuais nos dois anos.

`amazonia_legal` permanecerá como variável candidata.

---

## 13. Heterogeneidade municipal

Foi observada grande dispersão nas taxas municipais de alfabetização.

### 2023

- Q1: aproximadamente 46,73%;
- mediana: aproximadamente 62,10%;
- Q3: 76,00%.

### 2024

- Q1: aproximadamente 49,16%;
- mediana: aproximadamente 64,37%;
- Q3: aproximadamente 77,91%.

Isso reforça que a média nacional não representa adequadamente toda a
heterogeneidade territorial.

Também foram encontrados municípios do Nordeste entre os maiores resultados,
especialmente municípios do Ceará, ao mesmo tempo em que Nordeste e Norte
aparecem com taxas regionais agregadas menores.

Esse comportamento reforça que a região, isoladamente, não explica a
variabilidade municipal.

---

## 14. Estratégia temporal de modelagem

A separação principal será:

- 2023: desenvolvimento do modelo;
- 2024: teste temporal final.

Os dados de 2024 não serão utilizados para:

- seleção de features;
- ajuste de hiperparâmetros;
- escolha de modelo;
- definição de thresholds baseada em performance.

Dentro de 2023 será criada uma divisão de desenvolvimento apropriada para
treinamento, validação e otimização.

O conjunto de 2024 será utilizado apenas após a definição do pipeline final.

---

## 15. Seleção inicial de features

### Target

`alfabetizado`

Codificação:

- 0 = não alfabetizado;
- 1 = alfabetizado.

### Features categóricas iniciais

- `rede_nome`;
- `regiao`.

### Feature categórica experimental

- `sigla_uf`.

`sigla_uf` não será utilizada necessariamente no primeiro baseline porque
Acre, Distrito Federal e São Paulo não estão presentes no conjunto de 2023 e
aparecem em 2024.

Será possível comparar posteriormente um modelo com e sem UF para avaliar o
impacto dessa granularidade territorial na generalização temporal.

### Features binárias

- `capital_uf`;
- `amazonia_legal`.

### Features numéricas iniciais

- `populacao_2020`;
- `pib_por_habitante_2020`;
- `participacao_agropecuaria_2020`;
- `participacao_industria_2020`;
- `participacao_administracao_publica_2020`.

População e PIB por habitante serão candidatos à transformação `log1p`.

---

## 16. Variáveis excluídas de X

Não serão utilizadas diretamente como features:

- `ano`;
- `ano_referencia`;
- `aluno_key`;
- `id_escola_mascarado`;
- `id_municipio`;
- `rede_codigo`;
- `rede_agrupada`;
- `municipio`;
- `nome_uf`;
- `ano_contexto_socioeconomico`;
- `pib_2020`;
- `participacao_servicos_2020` no baseline linear;
- `source_batch_id`;
- `source_quality_status`;
- `gold_processed_at`.

Também permanecem proibidas as variáveis identificadas anteriormente como
leakage, principalmente:

- proficiência individual;
- taxa de alfabetização agregada do mesmo ano;
- proficiência média do mesmo ano;
- quantidade de alfabetizados do mesmo ano;
- distribuição de níveis de desempenho do mesmo ano.

---

## 17. Hipóteses para a modelagem

A EDA gera as seguintes hipóteses que serão avaliadas pelos modelos:

- fatores territoriais contribuem para diferenciar o risco de não alfabetização;
- a relação entre rede de ensino e alfabetização depende do contexto territorial;
- contexto econômico municipal possui associação com alfabetização, mas nenhum
  indicador isolado deve ser suficiente para explicar o resultado;
- variáveis de população e PIB por habitante podem beneficiar-se de
  transformação logarítmica em modelos lineares;
- o modelo precisará generalizar tanto temporalmente quanto para municípios não
  observados no treinamento;
- modelos não lineares podem capturar interações que um baseline linear não
  representa diretamente.

Essas hipóteses serão avaliadas empiricamente nas próximas etapas, sem
interpretação causal.