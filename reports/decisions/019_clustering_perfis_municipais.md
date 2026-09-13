# Decisão 019: Perfis municipais por aprendizagem não supervisionada

## Objetivo

Responder à pergunta estratégica:

**Quais municípios e regiões possuem padrões estruturais semelhantes?**

Foi utilizada aprendizagem não supervisionada para identificar grupos de
municípios semelhantes segundo características socioeconômicas e educacionais.

A análise é complementar ao modelo supervisionado e não altera nenhuma decisão
de modelagem anterior.

## Algoritmo

Foi utilizado K-Means.

Número final de clusters:

**K = 3**

A decisão foi tomada após avaliação conjunta de:

- Silhouette Score;
- Davies-Bouldin Index;
- inertia;
- tamanho dos clusters;
- interpretabilidade dos grupos.

## Seleção de K

K = 2 apresentou o maior Silhouette Score.

Entretanto, K = 3:

- manteve Silhouette próximo;
- apresentou o menor Davies-Bouldin Index;
- apresentou ganho expressivo de inertia em relação a K = 2;
- produziu três grupos com tamanhos substanciais;
- apresentou maior utilidade interpretativa para os perfis municipais.

Métricas finais de K = 3:

- Silhouette Score: 0,2034;
- Davies-Bouldin Index: 1,6429;
- Calinski-Harabasz: 1413,89;
- Inertia: 47.408,33.

Os clusters possuem:

- Cluster 0: 1.374 municípios;
- Cluster 1: 1.985 municípios;
- Cluster 2: 2.158 municípios.

## Reprodutibilidade

A execução final reproduziu exatamente a partição K = 3 obtida durante o
diagnóstico.

Adjusted Rand Index em relação ao candidato anterior:

**1,0000**

As diferenças das métricas entre as duas execuções foram numericamente
desprezíveis.

## Features utilizadas

Foram utilizadas 13 características estruturais.

### Socioeconômicas

- população;
- PIB por habitante;
- participação agropecuária;
- participação industrial;
- participação da administração pública.

### Educacionais

- quantidade de escolas dos anos iniciais;
- proporção de escolas rurais;
- proporção de internet para aprendizagem;
- proporção de biblioteca/sala de leitura;
- proporção de laboratório de informática;
- alunos por turma;
- razão matrículas/docentes;
- proporção de matrículas em tempo integral.

## Variáveis não utilizadas na formação dos clusters

Não participaram do clustering:

- id_municipio;
- UF;
- região;
- capital;
- Amazônia Legal;
- alfabetização;
- taxa real de risco;
- risco previsto;
- metas educacionais.

UF, região, alfabetização e risco foram utilizados somente após a formação dos
clusters para interpretação.

Essa separação evita circularidade analítica.

## Pré-processamento

As variáveis:

- população;
- PIB por habitante;
- quantidade de escolas;
- alunos por turma;

receberam transformação log1p.

Valores faltantes foram tratados por mediana.

Todas as features foram posteriormente padronizadas com StandardScaler.

O PCA foi utilizado apenas para visualização.

O K-Means foi treinado no espaço completo das 13 variáveis padronizadas.

## Perfil 0 — Urbano-industrial de maior escala

O Cluster 0 apresenta como principais características:

- população muito acima da média;
- PIB por habitante acima da média;
- forte participação industrial;
- baixa participação agropecuária;
- baixa participação relativa da administração pública;
- menor proporção de escolas rurais;
- infraestrutura educacional acima da média.

Valores médios aproximados:

- população: 111 mil habitantes;
- PIB por habitante: R$ 45,7 mil;
- participação industrial: 30,4%;
- participação agropecuária: 8,8%;
- proporção de escolas rurais: 25,1%;
- internet para aprendizagem: 80,1%;
- biblioteca ou sala de leitura: 70,4%.

O grupo contém 1.374 municípios.

A região mais representada dentro do cluster é o Sudeste.

O nome é descritivo e não representa classificação de qualidade ou desempenho.

## Perfil 1 — Pequeno porte agropecuário com maior disponibilidade de infraestrutura educacional

O Cluster 1 apresenta:

- população muito abaixo da média;
- participação agropecuária elevada;
- PIB por habitante relativamente acima da média;
- menor quantidade absoluta de escolas;
- elevada disponibilidade relativa de internet para aprendizagem;
- maior disponibilidade de biblioteca/sala de leitura;
- maior disponibilidade de laboratório de informática;
- menos alunos por turma;
- menor razão matrículas/docentes.

Valores médios aproximados:

- população: 7,2 mil habitantes;
- PIB por habitante: R$ 31,5 mil;
- participação agropecuária: 34,0%;
- internet para aprendizagem: 84,4%;
- biblioteca/sala de leitura: 78,1%;
- laboratório de informática: 54,2%;
- alunos por turma: 18,1;
- razão matrículas/docentes: 14,1.

O grupo contém 1.985 municípios.

Sul e Sudeste concentram a maior parte dos municípios desse perfil.

O nome do grupo é estrutural e não deve ser interpretado como indicador de
qualidade educacional.

## Perfil 2 — Rural, menor renda e maior presença da administração pública

O Cluster 2 apresenta:

- PIB por habitante abaixo da média;
- elevada presença da administração pública na estrutura econômica;
- alta proporção de escolas rurais;
- baixa disponibilidade de internet para aprendizagem;
- baixa disponibilidade de biblioteca/sala de leitura;
- baixa disponibilidade de laboratório de informática;
- maior razão matrículas/docentes;
- maior proporção de matrículas em tempo integral.

Valores médios aproximados:

- população: 20,1 mil habitantes;
- PIB por habitante: R$ 12,0 mil;
- participação da administração pública: 47,5%;
- proporção de escolas rurais: 68,4%;
- internet para aprendizagem: 47,2%;
- biblioteca/sala de leitura: 32,3%;
- laboratório de informática: 14,2%;
- razão matrículas/docentes: 18,8.

O grupo contém 2.158 municípios.

Nordeste e Norte concentram grande parte dos municípios desse perfil.

## Padrões regionais

Os clusters foram formados sem utilizar região ou UF.

Mesmo assim, a composição posterior revelou padrões territoriais claros.

### Nordeste

Aproximadamente 84,6% dos municípios pertencem ao Cluster 2.

### Norte

Aproximadamente 68,0% pertencem ao Cluster 2.

### Sul

Aproximadamente 66,1% pertencem ao Cluster 1.

### Centro-Oeste

Aproximadamente 57,3% pertencem ao Cluster 1.

### Sudeste

O Sudeste apresenta maior heterogeneidade:

- aproximadamente 45,9% no Cluster 1;
- aproximadamente 37,4% no Cluster 0;
- aproximadamente 16,7% no Cluster 2.

## Relação pós-hoc com risco educacional

Risco não participou da formação dos clusters.

Após a clusterização, foram observados diferentes níveis de risco entre os
perfis.

Valores ponderados aproximados:

### Cluster 0

- risco previsto: 41,0%;
- risco observado: 40,3%.

### Cluster 1

- risco previsto: 32,9%;
- risco observado: 31,0%.

### Cluster 2

- risco previsto: 45,0%;
- risco observado: 43,1%.

O Cluster 2 apresentou maior risco educacional agregado, enquanto o Cluster 1
apresentou o menor.

Essa relação é associativa e posterior à formação dos grupos.

Não é possível concluir que pertencer a determinado cluster cause maior ou menor
alfabetização.

## Interpretação estratégica

A segmentação mostra que municípios de uma mesma região administrativa podem
possuir estruturas bastante diferentes.

Da mesma forma, municípios geograficamente distantes podem compartilhar perfis
socioeconômicos e educacionais semelhantes.

Isso permite pensar políticas públicas por perfil estrutural, e não apenas por
fronteira geográfica.

Exemplos:

- municípios do Perfil 2 podem demandar atenção específica para infraestrutura
  educacional e condições de oferta em territórios rurais;

- municípios do Perfil 1 possuem características de pequeno porte e forte
  participação agropecuária, mas maior disponibilidade relativa de alguns
  recursos educacionais;

- municípios do Perfil 0 apresentam maior escala urbana e industrial, exigindo
  estratégias compatíveis com redes maiores e maior volume de alunos.

Essas aplicações representam hipóteses de priorização e não recomendações
causais automáticas.

## Limitações

O Silhouette Score de aproximadamente 0,20 indica que existe sobreposição entre
os grupos.

Portanto, os clusters devem ser interpretados como segmentação analítica útil e
não como classes naturais perfeitamente separadas.

K-Means também pressupõe distância Euclidiana e favorece estruturas relativamente
compactas no espaço padronizado.

Os resultados dependem:

- das features selecionadas;
- do pré-processamento;
- da escala temporal das fontes;
- da agregação município/rede;
- da escolha de K.

Além disso, características estruturais não devem ser confundidas com causas da
alfabetização.

## Decisão

O clustering final fica congelado em:

- algoritmo: K-Means;
- K = 3;
- 13 features estruturais;
- log1p nas variáveis assimétricas;
- imputação por mediana;
- StandardScaler;
- random_state = 42;
- n_init = 30.

Os perfis serão referenciados como:

- Cluster 0 — Urbano-industrial de maior escala;
- Cluster 1 — Pequeno porte agropecuário com maior disponibilidade de infraestrutura educacional;
- Cluster 2 — Rural, menor renda e maior presença da administração pública.

A pergunta:

**"Quais regiões possuem padrões semelhantes?"**

é considerada respondida por meio dos três perfis estruturais municipais e sua
composição regional.

## Próxima etapa

A próxima análise estratégica será dedicada à pergunta:

**"Quais municípios possuem maior risco de não atingir metas futuras?"**

Essa etapa utilizará metas e resultados somente após o encerramento da
modelagem supervisionada.

Nenhuma informação futura será utilizada como feature do modelo individual.