# Decisão 008: EDA do enriquecimento educacional e seleção de features

## Objetivo

Avaliar as novas variáveis educacionais provenientes do Censo Escolar antes
de incorporá-las à pipeline de Machine Learning.

A análise foi realizada exclusivamente com os dados de desenvolvimento de
2023.

Os dados de 2024 não foram utilizados para seleção de features,
transformações ou qualquer outra decisão de modelagem.

## Universo analisado

A análise considerou 1.502.809 alunos de 2023 e 5.881 contextos distintos de
município e rede.

Foram avaliadas 11 features educacionais.

As features contextuais foram analisadas principalmente no nível
município-rede para evitar que municípios com maior número de alunos
dominassem artificialmente as distribuições e correlações.

## Valores faltantes

O enriquecimento apresenta missingness residual.

Para a maior parte das variáveis educacionais existem 75 alunos sem valor,
equivalentes a aproximadamente 0,005% da população de desenvolvimento.

Para:

- `alunos_por_turma_anos_iniciais`;
- `razao_matriculas_docentes_anos_iniciais`;

existem 86 alunos com valores ausentes.

Os valores faltantes não serão preenchidos na camada de dados.

A imputação será realizada dentro da pipeline de Machine Learning, utilizando
mediana calculada exclusivamente no conjunto de treino.

## Redundância entre variáveis de escala

Foi identificado um grupo fortemente redundante composto por:

- `quantidade_escolas_anos_iniciais`;
- `total_matriculas_anos_iniciais`;
- `total_docentes_anos_iniciais`;
- `total_turmas_anos_iniciais`.

As principais correlações de Spearman observadas foram:

- matrículas x turmas: 0,9772;
- docentes x turmas: 0,9742;
- matrículas x docentes: 0,9545;
- escolas x turmas: 0,8629;
- escolas x matrículas: 0,8602;
- escolas x docentes: 0,8410.

Para evitar multicolinearidade, redundância e diluição da interpretabilidade,
o conjunto enriquecido principal não utilizará simultaneamente as quatro
variáveis.

`quantidade_escolas_anos_iniciais` foi escolhida como representante do grupo
de escala por apresentar boa interpretabilidade e maior associação
exploratória com a taxa de alfabetização entre as quatro variáveis.

As outras três variáveis continuarão disponíveis na base de dados para
experimentos futuros, mas não farão parte do conjunto enriquecido principal.

## Associação exploratória com alfabetização

As maiores associações de Spearman no nível município-rede foram observadas
para:

- biblioteca/sala de leitura: aproximadamente +0,283;
- internet para aprendizagem: aproximadamente +0,262;
- quantidade de escolas: aproximadamente -0,262;
- razão matrículas/docentes: aproximadamente -0,258;
- total de matrículas: aproximadamente -0,217;
- proporção de escolas rurais: aproximadamente -0,212;
- laboratório de informática: aproximadamente +0,209.

Diferenças entre grupos extremos chegaram a aproximadamente:

- +11,34 p.p. para biblioteca/sala de leitura;
- +10,85 p.p. para internet para aprendizagem;
- -11,29 p.p. para matrículas/docentes;
- -10,39 p.p. para quantidade de escolas.

Esses resultados representam associações exploratórias, não relações
causais.

## Diferenças entre redes

A EDA mostrou diferenças estruturais importantes entre as redes estadual e
municipal.

Em 2023, a rede estadual apresentou medianas superiores de infraestrutura
como internet, biblioteca e laboratório, além de maior taxa observada de
alfabetização.

Portanto, parte das associações encontradas entre infraestrutura e target pode
ser explicada por diferenças entre redes.

`rede_nome` permanece no modelo para que a contribuição adicional das
features educacionais seja avaliada condicionalmente às demais informações.

## Transformações

### Log1p

Será aplicado `log1p` em:

- `quantidade_escolas_anos_iniciais`;
- `alunos_por_turma_anos_iniciais`.

A quantidade de escolas apresentou assimetria elevada, reduzida de
aproximadamente 11,08 para 0,62 após `log1p`.

Alunos por turma apresentou assimetria de aproximadamente 1,19, reduzida para
aproximadamente -0,55 após a transformação.

### Sem log1p

`razao_matriculas_docentes_anos_iniciais` será inicialmente utilizada sem
logaritmo, pois sua assimetria bruta é inferior a 1.

As proporções serão mantidas em sua escala original [0,1].

Embora `proporcao_matriculas_integral_anos_iniciais` apresente assimetria
elevada, `log1p` produz melhora pequena e não resolve sua concentração próxima
de zero. Por isso ela será inicialmente preservada em sua escala original.

## Features educacionais selecionadas

O conjunto enriquecido principal utilizará:

1. `quantidade_escolas_anos_iniciais`;
2. `proporcao_escolas_rurais`;
3. `proporcao_internet_aprendizagem`;
4. `proporcao_biblioteca_sala_leitura`;
5. `proporcao_laboratorio_informatica`;
6. `alunos_por_turma_anos_iniciais`;
7. `razao_matriculas_docentes_anos_iniciais`;
8. `proporcao_matriculas_integral_anos_iniciais`.

As seguintes variáveis não integrarão o conjunto principal por redundância:

- `total_matriculas_anos_iniciais`;
- `total_docentes_anos_iniciais`;
- `total_turmas_anos_iniciais`.

## Conjuntos experimentais

Serão mantidos três conjuntos de features:

### BASELINE

9 features utilizadas na primeira etapa de modelagem.

### ENRICHED

9 features baseline + 8 features educacionais selecionadas.

Total: 17 features antes do encoding.

### ENRICHED + UF

17 features do conjunto enriquecido + `sigla_uf`.

Total: 18 features antes do encoding.

## Próxima etapa

Atualizar o preprocessing mantendo compatibilidade com a modelagem anterior e
incorporando:

- imputação dentro da pipeline;
- `log1p` somente nas variáveis definidas pela EDA;
- scaling das variáveis numéricas;
- One-Hot Encoding das variáveis categóricas;
- suporte aos três conjuntos experimentais.

A comparação dos conjuntos será realizada exclusivamente com dados de 2023.