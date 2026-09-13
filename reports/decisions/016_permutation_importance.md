# Decisão 016: Interpretabilidade por Permutation Importance

## Objetivo

Avaliar de quais variáveis o modelo final mais depende para realizar a predição
de risco de não alfabetização.

A análise foi realizada após o encerramento da modelagem e do teste temporal
final.

Nenhuma decisão sobre features, algoritmo, hiperparâmetros ou threshold foi
alterada a partir dos resultados de interpretabilidade.

## Modelo analisado

Modelo:

HistGradientBoostingClassifier

Feature set:

ENRICHED + UF

Threshold operacional de risco:

0,41

O pipeline foi treinado novamente utilizando todo o conjunto de desenvolvimento
de 2023, mantendo exatamente a configuração previamente congelada.

## Estratégia

Foi utilizada Permutation Importance no nível das features originais.

A lógica da técnica consiste em:

1. calcular a performance do pipeline sem alterações;
2. embaralhar uma feature;
3. gerar novamente as previsões;
4. medir quanto a performance diminui;
5. repetir o procedimento cinco vezes.

Quanto maior a perda de performance depois da permutação, maior a dependência
preditiva do modelo em relação àquela variável.

A métrica principal utilizada foi a queda no PR-AUC da classe de risco.

Também foram observadas:

- queda no ROC-AUC;
- queda na Balanced Accuracy no threshold 0,41.

## Amostra de interpretabilidade

Foi utilizada uma amostra de 250.000 registros do teste de 2024.

A amostragem foi realizada com StratifiedShuffleSplit considerando
simultaneamente:

- alfabetizado;
- sigla_uf.

O objetivo foi preservar aproximadamente a prevalência da classe e a composição
territorial da base.

A taxa real de risco da amostra foi 0,40215.

## Validação da amostra

Performance na amostra utilizada para interpretabilidade:

- PR-AUC risco: 0,52530
- ROC-AUC: 0,63146
- Balanced Accuracy: 0,59027

Performance no teste completo de 2024:

- PR-AUC risco: 0,52566
- ROC-AUC: 0,63218
- Balanced Accuracy: 0,59110

A proximidade dos resultados indica que a amostra utilizada reproduziu
adequadamente o comportamento global do modelo em 2024.

## Ranking por queda no PR-AUC

As dez features com maior importância foram:

1. sigla_uf — 0,07346
2. participacao_administracao_publica_2020 — 0,00714
3. participacao_agropecuaria_2020 — 0,00533
4. quantidade_escolas_anos_iniciais — 0,00513
5. populacao_2020 — 0,00424
6. pib_por_habitante_2020 — 0,00356
7. regiao — 0,00345
8. participacao_industria_2020 — 0,00317
9. rede_nome — 0,00289
10. proporcao_matriculas_integral_anos_iniciais — 0,00193

## Predominância da UF

A variável sigla_uf apresentou:

- queda média PR-AUC: 0,07346
- desvio-padrão: 0,00108
- queda média ROC-AUC: 0,07043
- queda média Balanced Accuracy: 0,04516

Sua importância no PR-AUC foi superior a dez vezes a importância da segunda
variável no ranking.

O resultado indica forte dependência preditiva do modelo em relação à informação
territorial representada pela UF.

Essa observação é coerente com o teste temporal de 2024, no qual a capacidade de
generalização foi substancialmente inferior em AC, DF e SP, unidades da
federação que não estavam representadas no desenvolvimento de 2023.

A análise não demonstra que a UF seja causa das diferenças de alfabetização.

A UF funciona como variável contextual que agrega diversas características
territoriais, sociais, econômicas, administrativas e educacionais presentes nos
dados.

## Variáveis econômicas e territoriais

Além da UF, o modelo demonstrou dependência de características estruturais dos
municípios, incluindo:

- participação da administração pública na economia;
- participação agropecuária;
- população;
- PIB por habitante;
- participação da indústria;
- região.

Esses resultados demonstram associação preditiva e não efeito causal.

Não é possível concluir, por exemplo, que mudanças isoladas nesses indicadores
produziriam alterações na alfabetização.

## Variáveis educacionais

A quantidade de escolas com anos iniciais apresentou a quarta maior importância
global, com queda média de PR-AUC de aproximadamente 0,00513.

Esse resultado reforça a utilidade do enriquecimento realizado com informações
do Censo Escolar.

Também apresentaram contribuição preditiva:

- proporção de matrículas em tempo integral;
- alunos por turma;
- razão matrículas/docentes;
- disponibilidade de biblioteca ou sala de leitura;
- laboratório de informática.

A magnitude dessas importâncias é inferior àquela observada para as principais
variáveis territoriais e econômicas.

## Rede administrativa

A variável rede_nome ficou na nona posição pelo critério principal.

Sua importância também foi observada na Balanced Accuracy.

Esse resultado é consistente com a heterogeneidade de performance identificada
entre as redes Municipal e Estadual no teste temporal.

A interpretação permanece associativa e não causal.

## Variáveis de baixa importância

Algumas variáveis apresentaram importância muito próxima de zero:

- proporcao_escolas_rurais;
- amazonia_legal;
- capital_uf.

capital_uf apresentou valor médio levemente negativo.

Valores negativos de pequena magnitude não significam efeito negativo sobre a
alfabetização.

Eles indicam que, nessa análise por permutação, a variável não demonstrou
contribuição preditiva independente relevante e pode possuir informação
redundante com outras features.

Da mesma forma, uma associação observada durante a EDA pode não resultar em alta
importância dentro do modelo multivariado.

## Limitação da Permutation Importance

Permutation Importance mede dependência preditiva, e não causalidade.

Features correlacionadas podem compartilhar informação.

Quando duas variáveis representam aspectos semelhantes, o modelo pode utilizar
uma como substituta da outra e a importância individual pode ser reduzida ou
mascarada.

Esse aspecto é particularmente relevante para variáveis territoriais como:

- sigla_uf;
- regiao;
- amazonia_legal;

e para indicadores socioeconômicos e educacionais correlacionados.

## Decisão metodológica

Os resultados não serão utilizados para remover features ou realizar novo
treinamento.

Modificar o conjunto de variáveis após observar o teste de 2024 constituiria uma
nova etapa de seleção baseada no conjunto de teste e comprometeria a separação
metodológica mantida no projeto.

A configuração final permanece inalterada.

A forte dependência de UF será registrada como:

- característica do modelo;
- limitação de generalização territorial;
- oportunidade de evolução futura.

## Evolução futura

Em um novo ciclo de desenvolvimento, poderá ser avaliado um modelo menos
dependente de identificação explícita da UF.

Uma possibilidade seria comparar:

- modelo com UF explícita;
- modelo sem UF explícita;
- maior uso de indicadores territoriais estruturais;
- validação leave-one-state-out.

Essa abordagem permitiria avaliar diretamente a capacidade de generalização para
unidades federativas completamente ausentes do treinamento.

Essa investigação não será realizada utilizando o teste final atual para
retuning.

## Próxima etapa

A próxima técnica de interpretabilidade será SHAP.

Permutation Importance responde principalmente:

"De quais variáveis o modelo depende?"

SHAP será utilizado para investigar:

- como diferentes valores das features estão associados às previsões;
- direção dos efeitos aprendidos pelo modelo;
- magnitude das contribuições;
- heterogeneidade entre observações.

As interpretações continuarão sendo tratadas como explicações do comportamento
preditivo do modelo, e não como inferência causal.