# ERA · documents

Motor de leitura de documentos. **Não iniciado.**

Este diretório existe para marcar o lugar do terceiro motor da ERA. Ainda não
há decisão de arquitetura tomada aqui — nem escopo, nem roteiro de fases.

Quando começar, a primeira conversa é sobre escopo, porque "leitura de
documentos" são vários problemas distintos e a resposta muda tudo:

- **Extração de PDF de texto** — o dado já está lá, é questão de parsear.
- **PDF escaneado / imagem** — precisa de OCR, e OCR em Go puro é um projeto
  do tamanho do `faces`.
- **Extração estruturada** — nota fiscal, boleto, ordem de compra: achar
  campos específicos, não texto corrido. É o caso que a Frota Macedo já
  resolve hoje por outros meios.
- **Classificação** — dizer que tipo de documento é.

Vale registrar desde já uma tensão real com os princípios do repositório: um
parser de PDF completo é a primeira coisa da ERA que teria motivo legítimo
para trazer dependência externa. Se for esse o caminho, a saída é dar `go.mod`
próprio a este motor — não afrouxar a checagem de zero dependências que vale
para os outros dois.
