# Decisão 001 — População de modelagem e definição do target

## Contexto

A base original de alunos possui registros de alunos presentes e ausentes na
avaliação, além de casos em que o aluno estava presente mas não preencheu a prova.

A auditoria mostrou que registros sem avaliação efetivamente realizada são
classificados como não alfabetizados e não possuem proficiência observada.

## Decisão

A população utilizada no modelo será formada apenas por registros que atendam:

- `presenca_codigo = '1'`;
- `preenchimento_caderno_codigo = '1'`;
- `quality_status = 'VALID'`.

## Volume final

### 2023

- Total: 1.502.809
- Não alfabetizados: 625.382
- Alfabetizados: 877.427

### 2024

- Total: 1.851.852
- Não alfabetizados: 744.733
- Alfabetizados: 1.107.119

## Target

O target utilizado será `alfabetizado`:

- `0`: não alfabetizado;
- `1`: alfabetizado.

## Justificativa

A exclusão de ausentes e provas não preenchidas evita interpretar ausência de
avaliação como evidência de não alfabetização.