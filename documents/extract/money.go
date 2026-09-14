package extract

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Money e um valor monetario em centavos de real, guardado como inteiro --
// nao float64. Dinheiro nao admite o erro de arredondamento binario de
// ponto flutuante (0,1 + 0,2 != 0,3 em float64), e um total que sai errado
// por causa disso e o tipo de bug que so aparece depois de somar milhares
// de linhas, tarde demais para notar de onde veio.
type Money int64

// String formata no padrao brasileiro: "R$ 1.234,56".
func (m Money) String() string {
	sinal := ""
	v := int64(m)
	if v < 0 {
		sinal, v = "-", -v
	}
	return fmt.Sprintf("%sR$ %s,%02d", sinal, milhar(v/100), v%100)
}

// milhar formata um inteiro nao negativo com ponto a cada 3 digitos, da
// direita para a esquerda -- o separador de milhar brasileiro.
func milhar(v int64) string {
	s := strconv.FormatInt(v, 10)
	if len(s) <= 3 {
		return s
	}
	var partes []string
	for len(s) > 3 {
		corte := len(s) - 3
		partes = append([]string{s[corte:]}, partes...)
		s = s[:corte]
	}
	partes = append([]string{s}, partes...)
	return strings.Join(partes, ".")
}

// moneyPattern acha um numero no padrao brasileiro: "R$" opcional na
// frente, ponto de milhar opcional, virgula decimal opcional (sem virgula,
// o numero e reais inteiros).
//
// A primeira alternativa do grupo exige PELO MENOS UM grupo ".ddd" (o `+`,
// nao `*`) -- de proposito: com `*` ela casaria so os 3 primeiros digitos
// de um numero sem separador nenhum ("1234" virava "123"), porque zero
// repeticoes do grupo opcional ja e um casamento valido e o motor de regex
// do Go nao troca de alternativa so para tentar achar um casamento mais
// longo. Com `+`, essa alternativa so aceita numero de fato agrupado por
// ponto; sem ponto nenhum, ela falha por completo e a segunda alternativa
// (digitos corridos, qualquer tamanho) assume.
var moneyPattern = regexp.MustCompile(`(?:R\$\s?)?(-?\d{1,3}(?:\.\d{3})+|-?\d+)(?:,(\d{2}))?`)

// ParseMoney le um valor monetario no padrao brasileiro dentro de s.
//
// Sem virgula, o numero e reais inteiros -- "5" vira R$ 5,00, nao 5
// centavos: e a leitura natural de uma nota que escreve "5" querendo dizer
// cinco reais.
//
// Ao contrario de FindCNPJs/FindCPFs, este pacote nao oferece um buscador
// de valor monetario em texto livre: nao ha digito verificador para um
// valor, entao "qualquer numero parecido com dinheiro" numa pagina inteira
// vai colidir com quantidade, numero de nota, CEP -- tudo numero. ParseMoney
// espera um campo ja isolado por layout ou por uma etiqueta conhecida
// ("Total:", "Valor:"), nao a pagina inteira.
func ParseMoney(s string) (Money, error) {
	s = strings.TrimSpace(s)
	m := moneyPattern.FindStringSubmatch(s)
	if m == nil {
		return 0, fmt.Errorf("extract: %q nao parece um valor monetario", s)
	}

	inteiroStr := strings.ReplaceAll(m[1], ".", "")
	negativo := strings.HasPrefix(inteiroStr, "-")
	inteiroStr = strings.TrimPrefix(inteiroStr, "-")
	inteiro, err := strconv.ParseInt(inteiroStr, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("extract: valor monetario invalido em %q: %w", s, err)
	}

	var centavos int64
	if m[2] != "" {
		centavos, err = strconv.ParseInt(m[2], 10, 64)
		if err != nil {
			return 0, fmt.Errorf("extract: centavos invalidos em %q: %w", s, err)
		}
	}

	total := inteiro*100 + centavos
	if negativo {
		total = -total
	}
	return Money(total), nil
}
