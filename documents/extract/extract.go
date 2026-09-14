// Package extract reconhece campos tipados dentro de texto ja lido:
// CNPJ, CPF, data e valor monetario.
//
// A aposta do README esta aqui: "regras e geometria resolvem 90%" de um
// documento estruturado, sem modelo nenhum. CNPJ e CPF tem digito
// verificador -- um algoritmo publicado, nao um palpite -- entao dao para
// achar dentro de texto livre com confianca alta: um numero de 14 digitos
// que bate no calculo da Receita Federal nao e coincidencia. Data e valor
// monetario nao tem verificacao equivalente, entao este pacote so oferece
// o parser (aplicado a um campo ja isolado por layout ou por uma etiqueta
// conhecida), nao um buscador de texto livre -- buscar "qualquer numero
// parecido com dinheiro" numa pagina inteira erra demais para valer a pena.
//
// Nada aqui depende de imagem, rede ou das fases anteriores do roteiro:
// e string entrando, dado tipado saindo.
package extract

import "strings"

// somenteDigitos remove tudo que nao e digito -- o primeiro passo de
// qualquer validador de CNPJ/CPF, que aceita a entrada com ou sem
// pontuacao.
func somenteDigitos(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}
