package extract

import (
	"fmt"
	"regexp"
)

var cpfPattern = regexp.MustCompile(`\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b`)

var pesosCPF1 = []int{10, 9, 8, 7, 6, 5, 4, 3, 2}
var pesosCPF2 = []int{11, 10, 9, 8, 7, 6, 5, 4, 3, 2}

// ValidCPF confere se s (com ou sem pontuacao) e um CPF valido: 11 digitos,
// os dois digitos verificadores batem, e os 11 digitos nao sao todos
// iguais.
//
// A regra dos digitos repetidos nao esta na formula -- 111.111.111-11
// passa no calculo do digito verificador tao bem quanto qualquer CPF real,
// dado que a soma ponderada de onze copias do mesmo digito ainda cai numa
// combinacao que fecha o resto certo -- mas nenhum CPF de verdade e assim,
// e todo validador de producao rejeita a sequencia por convencao. Sem essa
// checagem extra, ValidCPF aceitaria entrada obviamente falsa.
func ValidCPF(s string) bool {
	d := somenteDigitos(s)
	if len(d) != 11 {
		return false
	}
	if todosIguais(d) {
		return false
	}
	digitos := make([]int, 11)
	for i, r := range d {
		digitos[i] = int(r - '0')
	}
	if digitoVerificador(digitos[:9], pesosCPF1) != digitos[9] {
		return false
	}
	return digitoVerificador(digitos[:10], pesosCPF2) == digitos[10]
}

func todosIguais(s string) bool {
	for i := 1; i < len(s); i++ {
		if s[i] != s[0] {
			return false
		}
	}
	return true
}

// FormatCPF formata 11 digitos como XXX.XXX.XXX-XX. Nao valida -- use
// ValidCPF antes se precisar da garantia.
func FormatCPF(s string) (string, error) {
	d := somenteDigitos(s)
	if len(d) != 11 {
		return "", fmt.Errorf("extract: CPF precisa de 11 digitos, %q tem %d", s, len(d))
	}
	return fmt.Sprintf("%s.%s.%s-%s", d[0:3], d[3:6], d[6:9], d[9:11]), nil
}

// FindCPFs varre texto livre e devolve os CPFs validos encontrados,
// formatados -- mesma logica de FindCNPJs.
func FindCPFs(text string) []string {
	var achados []string
	for _, cand := range cpfPattern.FindAllString(text, -1) {
		if !ValidCPF(cand) {
			continue
		}
		if f, err := FormatCPF(cand); err == nil {
			achados = append(achados, f)
		}
	}
	return achados
}
