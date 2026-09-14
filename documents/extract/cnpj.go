package extract

import (
	"fmt"
	"regexp"
)

// cnpjPattern acha sequencias no formato de CNPJ dentro de texto livre,
// com ou sem pontuacao. So a forma -- a validade de verdade (o digito
// verificador) e conferida depois, em ValidCNPJ.
var cnpjPattern = regexp.MustCompile(`\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b`)

var pesosCNPJ1 = []int{5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2}
var pesosCNPJ2 = []int{6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2}

// ValidCNPJ confere se s (com ou sem pontuacao) e um CNPJ valido: 14
// digitos, e os dois digitos verificadores batem com o algoritmo da
// Receita Federal.
func ValidCNPJ(s string) bool {
	d := somenteDigitos(s)
	if len(d) != 14 {
		return false
	}
	digitos := make([]int, 14)
	for i, r := range d {
		digitos[i] = int(r - '0')
	}
	if digitoVerificador(digitos[:12], pesosCNPJ1) != digitos[12] {
		return false
	}
	return digitoVerificador(digitos[:13], pesosCNPJ2) == digitos[13]
}

// FormatCNPJ formata 14 digitos como XX.XXX.XXX/XXXX-XX. Nao valida --
// use ValidCNPJ antes se precisar da garantia.
func FormatCNPJ(s string) (string, error) {
	d := somenteDigitos(s)
	if len(d) != 14 {
		return "", fmt.Errorf("extract: CNPJ precisa de 14 digitos, %q tem %d", s, len(d))
	}
	return fmt.Sprintf("%s.%s.%s/%s-%s", d[0:2], d[2:5], d[5:8], d[8:12], d[12:14]), nil
}

// FindCNPJs varre texto livre (o que um reconhecedor devolveria) e devolve
// os CNPJs validos encontrados, formatados. Um candidato com o formato
// certo mas digito verificador errado -- ruido de reconhecimento comum --
// e descartado silenciosamente: e o proprio digito verificador que decide
// se 14 digitos sao um CNPJ ou coincidencia.
func FindCNPJs(text string) []string {
	var achados []string
	for _, cand := range cnpjPattern.FindAllString(text, -1) {
		if !ValidCNPJ(cand) {
			continue
		}
		if f, err := FormatCNPJ(cand); err == nil {
			achados = append(achados, f)
		}
	}
	return achados
}
