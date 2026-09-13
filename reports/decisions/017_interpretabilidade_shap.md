# Decisão 017: Interpretabilidade com SHAP Values

## Objetivo

Explicar o comportamento do HistGradientBoosting final utilizando SHAP Values,
com foco na classe de risco de não alfabetização.

A análise foi realizada após o encerramento da modelagem, seleção do threshold e
teste temporal final.

Nenhuma decisão de modelagem foi alterada a partir dos resultados de SHAP.

## Configuração analisada

Modelo:

HistGradientBoostingClassifier

Feature set:

ENRICHED + UF

Features:

- 18 features originais;
- 45 features após preprocessing.

Threshold operacional:

- risco >= 0,41.

O modelo foi treinado utilizando todo o conjunto de desenvolvimento de 2023.

A interpretação foi realizada sobre uma amostra estratificada do conjunto de
2024.

## Amostra SHAP

Foram utilizados 20.000 registros.

A amostragem preservou simultaneamente:

- classe alfabetizado;
- sigla_uf.

Taxa real de risco:

- amostra SHAP: 0,40220;
- base completa de 2024: 0,40216.

A proximidade indica que a amostra preservou adequadamente a prevalência da
classe.

## Implementação

Foi utilizado:

- SHAP 0.52.0;
- TreeExplainer;
- model_output = raw.

O modelo original prevê a classe alfabetizado = 1.

Para tornar a explicação coerente com o objetivo de negócio, os SHAP Values
foram apresentados na perspectiva da classe de risco:

risco = não alfabetizado.

Para classificação binária no score bruto, o score de risco possui sinal oposto
ao score de alfabetização.

Assim:

- SHAP de risco > 0 empurra a previsão em direção a maior risco;
- SHAP de risco < 0 empurra a previsão em direção a maior alfabetização.

## Auditoria de aditividade

Foi realizada auditoria entre:

- score bruto produzido pelo modelo;
- valor esperado;
- soma dos SHAP Values.

O erro máximo observado foi aproximadamente:

5,77e-15.

O valor praticamente nulo confirma consistência numérica entre a decomposição
SHAP e o score bruto do modelo.

## Importância global SHAP

O ranking das dez features originais com maior média absoluta de SHAP de risco
foi:

1. sigla_uf — 0,23242
2. regiao — 0,07986
3. proporcao_internet_aprendizagem — 0,07219
4. participacao_administracao_publica_2020 — 0,05291
5. participacao_agropecuaria_2020 — 0,04813
6. populacao_2020 — 0,04246
7. quantidade_escolas_anos_iniciais — 0,04054
8. proporcao_matriculas_integral_anos_iniciais — 0,02997
9. alunos_por_turma_anos_iniciais — 0,02991
10. participacao_industria_2020 — 0,02626

A média absoluta de SHAP representa magnitude de contribuição para o score e não
efeito causal.

## Predominância territorial

sigla_uf apresentou importância global substancialmente superior às demais
features.

O resultado confirma a forte dependência territorial já identificada por:

- validação temporal;
- Permutation Importance;
- análise de generalização para novas UFs.

regiao também apresentou contribuição global relevante.

A convergência dessas evidências indica que o modelo utiliza fortemente
informações territoriais para diferenciar risco.

Isso não implica que a localização geográfica seja causa direta da
alfabetização.

UF e região funcionam como proxies contextuais capazes de representar diferentes
condições econômicas, sociais, administrativas e educacionais.

## UFs presentes no desenvolvimento

Entre as UFs conhecidas pelo modelo, os perfis SHAP demonstraram contribuições
territoriais bastante heterogêneas.

Exemplos de contribuição média em direção a maior risco:

- Sergipe;
- Bahia;
- Rio Grande do Norte;
- Tocantins;
- Alagoas;
- Mato Grosso do Sul.

Exemplos de contribuição média em direção à alfabetização:

- Ceará;
- Goiás;
- Espírito Santo;
- Rondônia;
- Paraná;
- Pernambuco.

Essas diferenças representam padrões aprendidos pelo modelo e não efeitos
causais das UFs.

Também não devem ser interpretadas como ranking de qualidade dos sistemas
educacionais.

## UFs ausentes do desenvolvimento

AC, DF e SP não estavam presentes no conjunto de desenvolvimento de 2023.

Por utilizar OneHotEncoder com tratamento de categorias desconhecidas, essas UFs
não possuem componentes próprios aprendidos durante o treinamento.

Portanto, os SHAP agregados observados para AC, DF e SP no conjunto de 2024 não
devem ser interpretados como efeitos específicos aprendidos dessas unidades da
federação.

Para esses registros, os indicadores one-hot das UFs conhecidas ficam
inativos.

Essa limitação é coerente com a perda de generalização identificada para esses
territórios no teste temporal.

## Internet para aprendizagem

A proporção de escolas com internet para aprendizagem apresentou um dos padrões
educacionais mais claros.

Nos menores níveis da variável, o SHAP médio empurra o modelo em direção a maior
risco.

Na faixa de aproximadamente 0 a 0,47:

- SHAP médio de risco: +0,075.

Na faixa superior a aproximadamente 0,90:

- SHAP médio de risco: -0,095.

Dentro do comportamento aprendido pelo modelo, maior disponibilidade de internet
para aprendizagem está associada a contribuições em direção a menor risco
previsto.

O resultado não representa evidência causal.

## Biblioteca ou sala de leitura

A menor faixa de disponibilidade apresentou:

- SHAP médio de risco: aproximadamente +0,043.

A faixa mais elevada apresentou:

- SHAP médio de risco: aproximadamente -0,042.

O padrão aprendido associa maior disponibilidade de biblioteca ou sala de
leitura a contribuições em direção a menor risco.

Novamente, não é possível inferir causalidade.

## Alunos por turma

A variável alunos_por_turma_anos_iniciais apresentou direção consistente nos
extremos.

Na faixa inferior:

- SHAP médio de risco: aproximadamente -0,039.

Na faixa superior:

- SHAP médio de risco: aproximadamente +0,042.

O modelo associa turmas maiores, condicionalmente às demais variáveis, a
contribuições em direção a maior risco.

## Razão matrículas/docentes

A menor faixa apresentou contribuição média negativa ao risco.

A faixa superior apresentou contribuição média positiva:

- menor faixa: aproximadamente -0,014;
- maior faixa: aproximadamente +0,024.

O padrão é coerente com maior carga relativa de matrículas por docente sendo
associada pelo modelo a maior risco previsto.

A interpretação permanece preditiva e não causal.

## Matrículas em tempo integral

A maior faixa de proporção de matrículas em tempo integral apresentou:

- SHAP médio de risco: aproximadamente -0,086.

As faixas inferiores apresentaram contribuições predominantemente positivas.

O modelo associa altos níveis de matrícula em tempo integral a contribuições em
direção a menor risco previsto.

Esse resultado deverá ser apresentado como associação aprendida pelo modelo e
não como impacto causal de uma política de tempo integral.

## Variáveis socioeconômicas

As relações aprendidas para variáveis socioeconômicas não são necessariamente
monotônicas.

### Participação da administração pública

As faixas intermediárias apresentaram contribuições negativas ao risco.

A faixa mais elevada apresentou contribuição média positiva de aproximadamente
+0,090.

### Participação agropecuária

A menor faixa apresentou contribuição média positiva de aproximadamente +0,086.

A maior faixa apresentou contribuição negativa próxima de -0,048.

### PIB por habitante

Não foi observado padrão monotônico simples entre PIB por habitante e SHAP de
risco.

Esses resultados mostram que o HGB utiliza relações não lineares e interações
entre variáveis.

Não devem ser transformados em afirmações do tipo:

"mais PIB reduz risco"

ou

"mais participação agropecuária reduz risco".

## SHAP e Permutation Importance

Os rankings das duas técnicas não são idênticos.

Isso é esperado.

Permutation Importance mede quanto a performance global cai quando a informação
de uma variável é destruída.

SHAP mede quanto as features contribuem para os scores individuais produzidos
pelo modelo.

Em presença de features correlacionadas ou parcialmente substituíveis, uma
variável pode possuir contribuição SHAP relevante e ainda apresentar perda
menor de performance na permutação.

Um exemplo foi proporcao_internet_aprendizagem:

- elevada importância global segundo SHAP;
- importância menor segundo Permutation Importance.

Não existe contradição metodológica entre os resultados.

Por outro lado, sigla_uf apareceu como variável dominante em ambas as técnicas,
reforçando a conclusão sobre dependência territorial.

## Features de menor contribuição

Entre as menores importâncias SHAP encontram-se:

- proporcao_escolas_rurais;
- amazonia_legal;
- capital_uf.

A baixa contribuição não significa ausência de associação descritiva com
alfabetização.

Significa apenas que, considerando simultaneamente as demais features do modelo,
essas variáveis apresentam pouca contribuição adicional para os scores.

## Limitações de interpretação

SHAP explica o comportamento do modelo e não o processo causal que produz a
alfabetização.

Os resultados estão sujeitos a:

- correlação entre features;
- interações não lineares;
- informação territorial compartilhada;
- dependência da distribuição do conjunto analisado;
- ausência de variáveis individuais socioeconômicas;
- representação contextual das variáveis educacionais.

Consequentemente, frases causais como:

"essa variável aumenta a alfabetização"

ou

"essa política reduz o risco"

não são suportadas por esta análise.

A linguagem correta é:

"o modelo associa"

ou

"a variável contribui para o score previsto".

## Decisão

A etapa de interpretabilidade do modelo supervisionado é considerada concluída.

O modelo final permanece inalterado.

As técnicas utilizadas foram:

- Permutation Importance;
- SHAP Values.

Os principais achados são:

- forte predominância das informações territoriais;
- contribuição relevante de indicadores educacionais;
- relações não lineares em variáveis socioeconômicas;
- associação entre maior disponibilidade de alguns recursos educacionais e
  menor risco previsto;
- limitação de generalização para territórios ausentes do desenvolvimento.

## Próxima etapa

A partir deste ponto, o projeto deixa de focar a explicação do modelo e passa a
transformar previsões e dados em inteligência educacional.

As próximas análises deverão responder às perguntas estratégicas do desafio,
incluindo:

- quais municípios apresentam maior risco educacional;
- quais regiões possuem padrões semelhantes;
- quais municípios apresentam maior risco de não atingir metas;
- como os resultados podem apoiar priorização de políticas públicas.

Nenhuma dessas análises será utilizada para retuning do modelo final.