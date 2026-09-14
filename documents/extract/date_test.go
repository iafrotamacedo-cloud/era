package extract

import (
	"testing"
	"time"
)

func data(ano int, mes, dia int) time.Time {
	return time.Date(ano, time.Month(mes), dia, 0, 0, 0, 0, time.UTC)
}

func TestParseDateBRNumerica(t *testing.T) {
	casos := []struct {
		s    string
		want time.Time
	}{
		{"14/09/2026", data(2026, 9, 14)},
		{"14-09-2026", data(2026, 9, 14)},
		{"5/1/2026", data(2026, 1, 5)},
		{"05/01/26", data(2026, 1, 5)}, // pivo: < 70 -> 2000+
		{"01/01/71", data(1971, 1, 1)}, // pivo: >= 70 -> 1900+
	}
	for _, c := range casos {
		t.Run(c.s, func(t *testing.T) {
			got, err := ParseDateBR(c.s)
			if err != nil {
				t.Fatalf("ParseDateBR(%q): %v", c.s, err)
			}
			if !got.Equal(c.want) {
				t.Errorf("ParseDateBR(%q) = %v, quero %v", c.s, got, c.want)
			}
		})
	}
}

func TestParseDateBRExtenso(t *testing.T) {
	casos := []struct {
		s    string
		want time.Time
	}{
		{"14 de setembro de 2026", data(2026, 9, 14)},
		{"10 de março de 2025", data(2025, 3, 10)},
		{"10 de marco de 2025", data(2025, 3, 10)}, // sem acento tambem
		{"1 DE JANEIRO DE 2026", data(2026, 1, 1)}, // maiusculas
	}
	for _, c := range casos {
		t.Run(c.s, func(t *testing.T) {
			got, err := ParseDateBR(c.s)
			if err != nil {
				t.Fatalf("ParseDateBR(%q): %v", c.s, err)
			}
			if !got.Equal(c.want) {
				t.Errorf("ParseDateBR(%q) = %v, quero %v", c.s, got, c.want)
			}
		})
	}
}

func TestParseDateBRInvalida(t *testing.T) {
	casos := []string{
		"abacate",
		"32/13/2026", // dia e mes impossiveis
		"31/04/2026", // abril nao tem 31 dias
		"29/02/2027", // 2027 nao e bissexto
		"14 de nonexistente de 2026",
		"",
	}
	for _, s := range casos {
		if _, err := ParseDateBR(s); err == nil {
			t.Errorf("ParseDateBR(%q) deveria falhar", s)
		}
	}
}

func TestParseDateBR29FevereiroBissexto(t *testing.T) {
	got, err := ParseDateBR("29/02/2024")
	if err != nil {
		t.Fatalf("ParseDateBR: %v", err)
	}
	if !got.Equal(data(2024, 2, 29)) {
		t.Errorf("got %v, quero 29/02/2024", got)
	}
}
