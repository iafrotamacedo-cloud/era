# Diário do monorepo

Registro do que cada sessão faz, para as três sessões (`faces`, `maps`,
`documents`) se manterem cientes umas das outras sem precisar ler o `git log`
inteiro ou adivinhar o que mudou por baixo dos pés.

## Por que existe

O monorepo tem três sessões de Claude Code rodando ao mesmo tempo na mesma
pasta, cada uma cuidando de um motor. Isso já causou conflito de verdade —
commits que colidiram, trabalho de uma sessão indo parar em cima do de outra.
O diário não substitui o cuidado com `git add` seletivo (ver `CLAUDE.md`), mas
dá visibilidade: antes de mexer em algo que pode se cruzar com outro motor
(o `internal/protowire` compartilhado, o `go.mod` raiz, o `CLAUDE.md`, o CI),
vale olhar se alguém mais mexeu ali recentemente.

## Formato

**Um arquivo por dia**, em `diario/AAAA-MM-DD.md`. Isso é o que faz conflito
de merge ser raro: sessões diferentes, no mesmo dia, apendam no fim do mesmo
arquivo pequeno; dias diferentes nunca tocam o mesmo arquivo. Se o arquivo do
dia não existir, crie com o cabeçalho abaixo.

Cada entrada é curta e vai no fim do arquivo, nesta forma:

```
## HH:MM — motor — resumo de uma linha

O que foi feito, em duas ou três frases. Cite o commit se já foi empurrado.
Vale mais uma entrada dizendo "tentei X, não funcionou, motivo Y" do que
nenhuma — é isso que evita a próxima sessão repetir o mesmo caminho.
```

Exemplo:

```
## 14:32 — faces — corrige overflow de 32 bits no parser onnx

tamanhoMaximo estava sem tipo e virava int em plataformas de 32 bits.
Corrigido em e5d9d25. CI verde nos 11 alvos.
```

## Regras

- **Só acrescenta.** Nunca edite ou apague a entrada de outra sessão — nem
  para corrigir. Se precisar corrigir algo, escreva uma entrada nova dizendo
  o que estava errado na anterior.
- **Sempre no fim do arquivo do dia.** Não insira no meio.
- **`git pull` antes de escrever**, como já é hábito para qualquer commit
  neste repositório. Reduz a chance de duas sessões apendarem ao mesmo tempo
  sem saber uma da outra.
- **Se der conflito de merge mesmo assim**, ele vai aparecer como marcadores
  `<<<<<<<` / `=======` / `>>>>>>>` cercando as duas entradas — quase sempre
  a resolução certa é manter as duas, uma depois da outra, e apagar só os
  marcadores. Não é um conflito de conteúdo, é só o Git não saber que as duas
  adições eram para conviver.
- **Uma entrada por evento que valha a pena outra sessão saber** — não é
  para logar cada `go test`. Vale registrar: uma fase fechada, uma decisão de
  arquitetura, uma mudança em algo compartilhado (`internal/protowire`,
  `go.mod`, `CLAUDE.md`, `.github/workflows/ci.yml`), um problema encontrado
  e não resolvido, um commit que teve que reescrever histórico.
- **Commite o diário junto com o trabalho que ele descreve**, no mesmo
  commit ou logo depois — não deixe acumulando sem commitar.

## O que não é

Não é changelog do usuário nem substitui a mensagem de commit (que continua
completa, com o porquê). É a camada mais rápida de ler: alguém abre o arquivo
de hoje e em trinta segundos sabe o que as outras sessões andaram fazendo.
