# Decisão 020: Triagem de risco de não atingir metas futuras

## Objetivo

Responder à pergunta estratégica:

**Quais municípios apresentam maior risco de não atingir metas futuras de alfabetização?**

A análise foi realizada após o encerramento da modelagem supervisionada e da
clusterização.

Nenhuma informação de meta foi utilizada como feature do modelo individual.

## Universo

A fonte municipal contempla a rede Municipal.

Foram analisados:

- 5.352 municípios;
- resultados observados de 2023 e 2024;
- metas municipais de 2025 a 2030.

Dos 5.352 municípios:

- 5.232 possuem resultado observado em 2023 e 2024;
- 120 não possuem taxa observada em 2023.

Os 120 municípios sem histórico de 2023 não recebem classificação baseada em
tendência.

## Por que não foi utilizado um modelo de séries temporais

A base possui somente dois anos municipais observados:

- 2023;
- 2024.

Dois pontos temporais não são suficientes para estimar de forma robusta:

- tendência de longo prazo;
- sazonalidade;
- autocorrelação;
- modelos ARIMA;
- Prophet;
- forecasting multianual confiável.

Por esse motivo, a análise foi tratada como:

**triagem de trajetória**

e não como previsão de série temporal.

## Método

Para cada município foi calculada a variação:

ganho observado =
taxa 2024 - taxa 2023

Para cada ano de meta foi calculado o ganho anual necessário.

Também foi criado um cenário de continuidade:

cenário bruto =
taxa 2024 + ganho observado × horizonte

O cenário bruto não é interpretado como taxa literal.

Ele pode matematicamente ficar abaixo de 0 ou acima de 100 e é utilizado
exclusivamente para medir a distância entre:

- trajetória recente;
- trajetória necessária para alcançar a meta.

Para apresentação humana, também é produzida uma versão limitada entre 0% e
100%.

## Déficit de trajetória

O indicador principal é:

déficit de trajetória =
meta - cenário bruto

Valores positivos significam que, caso a variação de 2023 para 2024 se
repetisse, a trajetória seria insuficiente para atingir a meta.

Valores negativos ou zero indicam trajetória suficiente dentro desse cenário.

O indicador não representa uma taxa prevista.

## Ranking principal

O ranking principal utiliza a meta de 2025.

Os municípios são ordenados pelo maior déficit de trajetória.

O ranking:

- não utiliza HGB;
- não utiliza clustering;
- não utiliza score com pesos arbitrários.

Clustering, região e UF são adicionados somente depois para interpretação.

## Por que o HGB não participa

As metas municipais pertencem especificamente à rede Municipal.

O modelo supervisionado foi desenvolvido em outro universo de agregação.

Combinar diretamente sua probabilidade municipal com a meta da rede Municipal
criaria uma incompatibilidade de universo.

Por esse motivo, o HGB não participa do ranking de cumprimento de metas.

## Resultado de 2025

Entre os 5.232 municípios com histórico:

- 2.450 estão abaixo da meta e apresentam trajetória insuficiente;
- 82 já atingiram a meta de 2025 em 2024, mas a trajetória recente indica risco
  de voltar a ficar abaixo dela;
- 474 estão abaixo da meta, mas o ritmo recente seria suficiente no cenário de
  continuidade;
- 2.226 já atingiram a meta e apresentam trajetória compatível com sua
  manutenção.

No total:

**2.532 municípios são sinalizados para monitoramento em 2025.**

Isso corresponde a aproximadamente:

**48,4% dos municípios com histórico.**

## Maiores alertas de trajetória

Os primeiros municípios no ranking são caracterizados por fortes quedas entre
2023 e 2024.

Exemplos:

### Sério / RS

- taxa 2023: 85,4%
- taxa 2024: 11,1%
- variação: -74,3 p.p.
- meta 2025: 80,0%
- déficit de trajetória: 143,2 p.p.

### Arroio do Padre / RS

- taxa 2023: 94,6%
- taxa 2024: 18,2%
- variação: -76,4 p.p.
- meta 2025: 80,0%
- déficit de trajetória: 138,2 p.p.

### São Vendelino / RS

- taxa 2023: 100,0%
- taxa 2024: 25,0%
- variação: -75,0 p.p.
- meta 2025: 80,0%
- déficit de trajetória: 130,0 p.p.

### Bozano / RS

- taxa 2023: 100,0%
- taxa 2024: 27,3%
- variação: -72,7 p.p.
- meta 2025: 80,0%
- déficit de trajetória: 125,4 p.p.

### Jaboticaba / RS

- taxa 2023: 85,5%
- taxa 2024: 25,0%
- variação: -60,5 p.p.
- meta 2025: 80,0%
- déficit de trajetória: 115,5 p.p.

Esses municípios devem ser interpretados como alertas para investigação.

Uma variação anual extrema pode refletir mudança real, mas também pode ser
influenciada por:

- tamanho da coorte;
- composição dos estudantes avaliados;
- volatilidade anual;
- características locais da avaliação.

## Resultado regional

A proporção de municípios sinalizados para 2025 foi aproximadamente:

- Sul: 68,9%;
- Norte: 56,6%;
- Nordeste: 51,3%;
- Sudeste: 35,3%;
- Centro-Oeste: 28,9%.

O Sul apresentou variação média 2023–2024 de aproximadamente -8,2 pontos
percentuais.

Sudeste e Centro-Oeste apresentaram ganho médio próximo de +7,8 pontos
percentuais.

A análise regional não deve ser confundida com o risco individual estimado pelo
modelo supervisionado.

São perguntas diferentes.

O modelo supervisionado mede risco contextual de não alfabetização.

A análise de metas mede adequação da trajetória recente em relação aos
benchmarks planejados.

## Relação com os clusters

Clustering não participou do ranking.

Após a classificação:

### Cluster 0

Urbano-industrial de maior escala

- 1.324 municípios com histórico;
- 675 sinalizados;
- 51,0% sinalizados;
- déficit médio entre sinalizados: 18,6 p.p.

### Cluster 1

Pequeno porte agropecuário com maior infraestrutura educacional

- 1.850 municípios com histórico;
- 841 sinalizados;
- 45,5% sinalizados;
- déficit médio entre sinalizados: 28,2 p.p.

### Cluster 2

Rural, menor renda e maior presença da administração pública

- 2.058 municípios com histórico;
- 1.016 sinalizados;
- 49,4% sinalizados;
- déficit médio entre sinalizados: 21,9 p.p.

O Cluster 1 não possui a maior proporção de municípios sinalizados, mas apresenta
a maior intensidade média de déficit entre seus casos de risco.

## Horizontes 2025–2030

A proporção de municípios sinalizados permanece próxima de metade:

- 2025: 48,4%;
- 2026: 49,2%;
- 2027: 49,4%;
- 2028: 49,6%;
- 2029: 49,4%;
- 2030: 49,3%.

Entretanto, o déficit médio de trajetória entre os municípios sinalizados cresce
fortemente:

- 2025: 23,1 p.p.;
- 2026: 34,9 p.p.;
- 2027: 46,8 p.p.;
- 2028: 58,5 p.p.;
- 2029: 70,3 p.p.;
- 2030: 81,9 p.p.

Esse crescimento ocorre porque uma única variação anual está sendo extrapolada
por horizontes progressivamente maiores.

Por esse motivo:

- 2025 é tratado como horizonte operacional principal;
- 2026–2030 são tratados como cenários de stress de trajetória para planejamento.

Eles não devem ser apresentados como forecasts pontuais.

## Uso recomendado

A análise pode ser utilizada para:

- identificar municípios que precisam de acompanhamento imediato;
- diferenciar distância atual da meta de deterioração recente;
- priorizar investigação de quedas abruptas;
- orientar acompanhamento anual;
- identificar territórios cuja trajetória precisa mudar para alcançar os
  benchmarks planejados.

O ranking não substitui avaliação educacional local.

## Limitações

As principais limitações são:

- somente dois anos observados;
- ausência de uma série temporal longa;
- grande sensibilidade a variações anuais;
- possíveis efeitos de coortes pequenas;
- resultados dependentes da participação na avaliação;
- extrapolações de longo prazo altamente incertas;
- ausência de interpretação causal.

O cenário bruto é utilizado somente como índice de distância de trajetória e
nunca como uma taxa literal de alfabetização.

## Decisão

A pergunta:

**"Quais municípios apresentam maior risco de não atingir metas futuras?"**

é considerada respondida por uma triagem de trajetória.

O horizonte principal para priorização será 2025.

Metas de 2026 a 2030 serão utilizadas como cenários estratégicos de planejamento,
não como previsões determinísticas.

Nenhuma informação dessa análise será utilizada para retuning do modelo
supervisionado.

## Próxima etapa

Com esta análise, ficam concluídas as principais perguntas estratégicas do Tech
Challenge.

A próxima etapa será o fechamento do projeto:

- consolidação do README;
- requirements e instruções de reprodução;
- organização final dos resultados;
- revisão do repositório;
- roteiro executivo do vídeo de até 5 minutos.