# Decisão 018: Inteligência territorial de risco municipal

## Objetivo

Responder à pergunta de negócio:

**Quais municípios apresentam maior risco educacional?**

A análise utiliza exclusivamente as previsões produzidas pelo modelo final
congelado no teste temporal de 2024.

Nenhuma alteração de:

- features;
- algoritmo;
- hiperparâmetros;
- threshold;

foi realizada nesta etapa.

## Universo analisado

A base municipal de 2024 contém:

- 5.517 municípios;
- 1.851.852 alunos.

Os municípios foram classificados segundo o domínio territorial observado no
desenvolvimento:

- DOMINIO_FORTE: 4.841 municípios;
- DOMINIO_INTERMEDIARIO: 11 municípios;
- FORA_DOMINIO_UF: 665 municípios.

DOMINIO_FORTE representa municípios e UFs já observados no conjunto de
desenvolvimento de 2023.

DOMINIO_INTERMEDIARIO representa municípios novos localizados em UFs conhecidas.

FORA_DOMINIO_UF representa municípios localizados em AC, DF e SP, UFs
completamente ausentes do desenvolvimento de 2023.

Essa classificação não altera as previsões.

Ela é utilizada somente para comunicar a confiança de generalização.

## Duas dimensões de prioridade

Foram construídas duas perspectivas complementares.

### Severidade

Representa a concentração prevista de risco no município.

Indicador:

probabilidade média prevista de risco.

Pergunta respondida:

"Em quais municípios a concentração prevista de alunos em risco é maior?"

### Carga estimada

Representa o volume esperado de alunos associado ao risco.

Indicador:

carga_risco_estimada =
n_alunos × probabilidade_media_risco

Como a probabilidade municipal é a média das probabilidades individuais,
essa multiplicação equivale à soma esperada das probabilidades dos alunos.

A carga estimada não representa uma contagem observada de alunos não
alfabetizados.

Pergunta respondida:

"Em quais municípios pode estar concentrado o maior volume absoluto de alunos em
risco?"

## Ranking executivo por severidade

Para tornar o ranking de taxa mais estável e adequado à comunicação executiva,
foram considerados:

- apenas municípios em DOMINIO_FORTE;
- mínimo de 100 alunos avaliados.

O resultado observado de alfabetização em 2024 não participa da construção do
ranking.

Os primeiros municípios foram:

1. Nossa Senhora do Socorro / SE — 0,7389
2. Pacatuba / SE — 0,7273
3. Canindé de São Francisco / SE — 0,7207
4. Riachão do Dantas / SE — 0,7194
5. Porto da Folha / SE — 0,7167
6. Laranjeiras / SE — 0,7112
7. Japaratuba / SE — 0,7108
8. Arauá / SE — 0,7054
9. Casa Nova / BA — 0,7015
10. Boquim / SE — 0,7012

Entre os 20 primeiros municípios:

- 17 pertencem a Sergipe;
- 3 pertencem à Bahia;
- todos pertencem à região Nordeste.

## Interpretação da concentração territorial

A concentração do ranking em Sergipe é consistente com os resultados anteriores
de interpretabilidade.

UF foi identificada como a variável de maior dependência preditiva por:

- Permutation Importance;
- SHAP Values.

Portanto, o ranking municipal contém forte componente territorial.

Isso não permite concluir que a UF ou o sistema educacional estadual cause
diretamente o nível de alfabetização.

O resultado deve ser interpretado como priorização contextual aprendida pelo
modelo.

## Validação da ordenação municipal

No DOMINIO_FORTE, a correlação de Spearman entre:

- probabilidade média prevista de risco;
- taxa real de risco municipal em 2024;

foi aproximadamente:

**0,598**

O resultado real de 2024 foi utilizado apenas para diagnóstico posterior e não
para construir o ranking.

A correlação positiva moderada indica que a ordenação de risco prevista possui
aderência à ordenação observada, embora não reproduza perfeitamente as taxas
municipais.

Exemplos de proximidade incluem:

Nossa Senhora do Socorro / SE:

- risco previsto: 0,7389;
- risco observado: 0,7020.

Canindé de São Francisco / SE:

- risco previsto: 0,7207;
- risco observado: 0,7304.

Também existem divergências importantes.

Japaratuba / SE apresentou:

- risco previsto: 0,7108;
- risco observado: 0,4192.

Esse comportamento confirma que o modelo deve ser utilizado para priorização e
triagem, e não como estimador exato da taxa de alfabetização de cada município.

## Ranking por carga estimada

Ao considerar o volume estimado de alunos associado ao risco, as prioridades
mudam.

Os principais municípios no DOMINIO_FORTE foram:

1. Rio de Janeiro / RJ
   - 43.411 alunos
   - probabilidade média de risco: 0,4191
   - carga estimada: 18.192

2. Manaus / AM
   - 23.125 alunos
   - probabilidade média de risco: 0,4502
   - carga estimada: 10.412

3. Salvador / BA
   - 12.158 alunos
   - probabilidade média de risco: 0,5979
   - carga estimada: 7.269

4. Belo Horizonte / MG
   - 16.012 alunos
   - probabilidade média de risco: 0,3736
   - carga estimada: 5.982

5. Campo Grande / MS
   - 10.001 alunos
   - probabilidade média de risco: 0,5741
   - carga estimada: 5.742

6. Fortaleza / CE
   - 17.722 alunos
   - probabilidade média de risco: 0,2542
   - carga estimada: 4.504

Isso demonstra que severidade relativa e volume absoluto representam problemas
de política pública distintos.

Um município pode apresentar:

- alta concentração prevista de risco e população menor;
- risco médio moderado, mas grande quantidade absoluta de alunos expostos.

As duas dimensões devem ser consideradas na priorização de recursos.

## Complementaridade dos rankings

Os rankings de severidade e carga não são redundantes.

Entre os 30 principais municípios de cada ranking, a sobreposição é muito
pequena.

Isso demonstra que utilizar apenas a probabilidade média ou apenas o volume
absoluto ocultaria dimensões relevantes da priorização.

Não foi criado um score arbitrário combinando os dois indicadores.

As duas perspectivas permanecerão disponíveis separadamente para permitir que a
regra de política pública determine qual delas é mais adequada ao contexto.

## Panorama regional

No DOMINIO_FORTE, a probabilidade média de risco ponderada pelo número de alunos
foi aproximadamente:

- Norte: 0,4866;
- Nordeste: 0,4448;
- Centro-Oeste: 0,4050;
- Sudeste: 0,4036;
- Sul: 0,3165.

Esses valores representam o comportamento preditivo agregado do modelo.

Não devem ser interpretados como efeito causal das regiões.

Também foi observada heterogeneidade entre risco previsto e risco observado.

No Sul, por exemplo:

- risco previsto ponderado: aproximadamente 0,3165;
- risco observado ponderado: aproximadamente 0,3903.

Essa diferença deve ser considerada uma limitação de calibração territorial e
não será utilizada para recalibrar o modelo após o teste final.

## Territórios fora do domínio

AC, DF e SP estavam completamente ausentes do desenvolvimento de 2023.

O teste temporal demonstrou desempenho substancialmente inferior nessas UFs.

Por esse motivo, seus municípios foram excluídos do ranking executivo principal.

As previsões continuam disponíveis em arquivo separado para análise
exploratória.

A exclusão do ranking principal não significa ausência de necessidade
educacional.

Significa apenas que a confiança de generalização do modelo nesses territórios é
inferior.

Exemplos demonstram por que essa distinção é importante.

Caso todas as UFs fossem combinadas sem considerar a validade territorial:

- São Paulo / SP apresentaria carga estimada próxima de 39 mil;
- Brasília / DF apresentaria carga estimada próxima de 12 mil.

Esses números não devem ser comparados operacionalmente com estimativas do
DOMINIO_FORTE sem validação local.

## Uso recomendado

Os resultados municipais podem apoiar duas estratégias.

### Priorização por severidade

Indicada quando o objetivo é identificar territórios com elevada concentração
prevista de alunos em risco.

### Priorização por escala

Indicada quando o objetivo é direcionar políticas capazes de alcançar o maior
volume potencial de alunos em risco.

A decisão final de política pública deve combinar os resultados do modelo com
informações locais, disponibilidade de recursos e validação educacional.

## Limitações

O ranking possui as mesmas limitações do modelo supervisionado:

- predominância de variáveis contextuais;
- forte dependência territorial;
- ausência de variáveis socioeconômicas individuais;
- dificuldade de generalização para UFs não observadas;
- probabilidade de risco não é diagnóstico individual;
- rankings representam associação preditiva e não causalidade.

Além disso, rankings baseados em médias municipais podem ser mais instáveis em
municípios com poucos registros.

Por esse motivo, foi adotado mínimo de 100 alunos no ranking executivo de
severidade.

## Decisão

A pergunta:

**"Quais municípios apresentam maior risco educacional?"**

é considerada respondida por duas perspectivas complementares:

1. severidade prevista;
2. carga estimada de risco.

O ranking executivo principal utiliza somente municípios no domínio territorial
mais confiável do modelo.

Territórios fora do domínio permanecem disponíveis separadamente e devem passar
por validação local antes do uso operacional.

Nenhum resultado desta análise será utilizado para retuning do modelo.

## Próxima etapa

A próxima pergunta estratégica será:

**"Quais regiões ou municípios apresentam perfis semelhantes?"**

Essa questão será tratada por aprendizagem não supervisionada, utilizando
clustering sobre perfis municipais.

O clustering será uma análise complementar de inteligência educacional e não
alterará o modelo supervisionado final.