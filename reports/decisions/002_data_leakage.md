# Decisão 002: Tratamento de data leakage

## Proficiência individual

A auditoria mostrou que o ponto de corte de 743 pontos reproduz integralmente
a classificação de alfabetização nos registros avaliados.

Por isso, `proficiencia` é considerada leakage direto e não faz parte das
features da base de Machine Learning.

## Indicadores agregados do mesmo ano

Indicadores como:

- taxa de alfabetização municipal;
- média de português;
- quantidade de alfabetizados;
- proficiência média;
- distribuição por nível de desempenho;

também não são utilizados no modelo individual quando calculados para o mesmo
ano da avaliação.

Esses indicadores são derivados dos resultados dos próprios alunos que o
modelo tenta classificar.

## Identificadores

`id_aluno`, `id_escola` e `id_municipio` não são tratados como variáveis
numéricas preditoras.

O código de escola disponível na fonte é mascarado/fictício e, portanto, não
foi utilizado para integração direta com fontes externas de escolas.