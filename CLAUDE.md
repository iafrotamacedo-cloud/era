# ERA

Monorepo de motores em Go puro. Três motores independentes sob um `go.mod`:

| Diretório | Motor | Estado |
|---|---|---|
| `faces/` | reconhecimento facial | Fase 5 de 7 |
| `maps/` | distâncias e rotas rodoviárias | Fase 2 de 7 |
| `documents/` | leitura de documentos | não iniciado |
| `internal/protowire/` | protobuf compartilhado — `faces` lê `.onnx`, `maps` lerá `.osm.pbf` | pronto |

`docs/` é documentação, não código. O motor de documentos é `documents/`.

## Várias sessões trabalham aqui ao mesmo tempo

Este repositório costuma ter **três sessões de Claude Code abertas na mesma
pasta**, uma por motor. Elas dividem uma árvore de trabalho e um índice do
git. Isso muda como se commita.

**Nunca use `git add -A`, `git add .` ou `git commit -a`.** Prepare só os
caminhos do motor em que você está mexendo:

```
git add maps/            # certo
git add -A               # varre o trabalho em andamento das outras sessões
```

Isso não é zelo excessivo — já aconteceu três vezes num único dia. Um pacote
inteiro foi parar num commit sobre outro assunto, e uma correção de
compilação foi parar num commit sobre cache.

Regras que decorrem disso:

- **`git status` mostra arquivos que não são seus.** Antes de preparar
  qualquer coisa, confira se o arquivo é do seu motor. Na dúvida, deixe.
- **Não reescreva histórico já publicado** sem dizer ao usuário. Se precisar,
  crie uma ramificação de segurança antes e prove que a árvore final não
  mudou (`git diff backup HEAD` vazio).
- **Rascunho não vai para a raiz.** Use o diretório de scratchpad da sessão.
  Se precisar de um arquivo temporário aqui, prefixe com `.tmp-` — já está
  no `.gitignore`.
- Uma mensagem de commit descreve o que aquele commit faz. Se ela precisar
  de um "e também", provavelmente são dois commits.

## Ambiente

O Go **não está no PATH do Git Bash**. Exporte antes de usar:

```
export PATH="$PATH:/c/Program Files/Go/bin"
```

Não há compilador C na máquina, então `go test -race` não roda localmente —
quem cobre isso é o CI, no Linux.

`go vet ./...` é lento nesta máquina (minutos). Rode em segundo plano ou com
folga de tempo.

## Comandos

```
go test ./...              # todos os motores
go test ./maps/...         # só um
go vet ./...
gofmt -l .                 # tem de sair vazio
go test ./maps/geo/ -run='^$' -bench=. -benchtime=2s
```

## Convenções

**Comentários e identificadores.** Identificadores em inglês, comentários em
português **sem acento** — o repositório inteiro é assim, e misturar as duas
grafias polui o diff. READMEs, esses sim, levam acento normal.

**Mensagens de erro** em português, minúsculas, prefixadas com o pacote:
`fmt.Errorf("dist: limites de faixa fora de ordem crescente: %v", limites)`.

**Referência antes de otimização.** Toda implementação rápida é conferida nos
testes contra uma versão óbvia — `geo/reference.go`, `faces/kernel/reference.go`.
Nunca otimize um arquivo de referência: o valor dele é ser simples o bastante
para dar para ler e afirmar que está certo.

**Meça, não afirme.** Números em README e em comentário saem de teste ou
benchmark que qualquer um pode rodar. Se um comentário diz "9× mais rápido",
existe um benchmark que mostra isso.

**Quando o benchmark contradisser o design, mude o texto, não o número.** Já
aconteceu com o cache do `dist`: foi escrito como otimização e mede pior que
não usá-lo. A documentação diz isso, com a tabela.

**Teste que não pode falhar não é teste.** Se um teste passaria mesmo com a
função quebrada, ele não está medindo nada — varra um intervalo de entradas
em vez de escolher um valor sortudo.

## O que o CI verifica a cada push

Todas as promessas do README são verificadas, porque promessa sem verificação
é só intenção:

- testes nos três sistemas, com `-shuffle=on`
- `-race`, com repetição no `faces/kernel`
- `gofmt`, `go vet`, `go mod tidy` sem diferença
- **zero dependências** — `go list -m all` tem de vir vazio
- **sem cgo** — nenhum `import "C"`
- cross-compile para 11 plataformas, do Raspberry Pi 32 bits ao WASM

As duas de negrito são as fáceis de quebrar sem perceber: basta um `go get`
distraído. Se um motor precisar mesmo de biblioteca externa, a saída acordada
é dar `go.mod` próprio àquele motor — não afrouxar a checagem.

## Dados

O repositório não guarda pesos de modelo nem extratos de mapa. Eles têm
licença própria e pesam centenas de MB; `.onnx`, `.pbf`, `.osm` e `.eramap`
estão no `.gitignore`.
