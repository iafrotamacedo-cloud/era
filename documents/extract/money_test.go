package extract

import "testing"

func TestParseMoneyFormatosValidos(t *testing.T) {
	casos := []struct {
		s    string
		want Money
	}{
		{"R$ 1.234,56", 123456},
		{"1234,56", 123456},
		{"R$5", 500},
		{"1.234", 123400},
		{"-12,30", -1230},
		{"R$ 0,05", 5},
		{"1.234.567,89", 123456789},
	}
	for _, c := range casos {
		t.Run(c.s, func(t *testing.T) {
			got, err := ParseMoney(c.s)
			if err != nil {
				t.Fatalf("ParseMoney(%q): %v", c.s, err)
			}
			if got != c.want {
				t.Errorf("ParseMoney(%q) = %v centavos, quero %v", c.s, got, c.want)
			}
		})
	}
}

func TestParseMoneyInvalida(t *testing.T) {
	for _, s := range []string{"", "abacate", "R$"} {
		if _, err := ParseMoney(s); err == nil {
			t.Errorf("ParseMoney(%q) deveria falhar", s)
		}
	}
}

func TestMoneyString(t *testing.T) {
	casos := []struct {
		m    Money
		want string
	}{
		{0, "R$ 0,00"},
		{5, "R$ 0,05"},
		{500, "R$ 5,00"},
		{123456, "R$ 1.234,56"},
		{123456789, "R$ 1.234.567,89"},
		{-1230, "-R$ 12,30"},
	}
	for _, c := range casos {
		if got := c.m.String(); got != c.want {
			t.Errorf("Money(%d).String() = %q, quero %q", c.m, got, c.want)
		}
	}
}

// TestMoneyRoundTripCanonicaliza confere que ParseMoney seguido de String
// sempre produz a forma canonica com separador de milhar, mesmo quando a
// entrada nao tinha -- "1234,56" tem que virar "R$ 1.234,56" depois do
// ciclo, nao "R$ 1234,56".
func TestMoneyRoundTripCanonicaliza(t *testing.T) {
	m, err := ParseMoney("1234,56")
	if err != nil {
		t.Fatalf("ParseMoney: %v", err)
	}
	if got := m.String(); got != "R$ 1.234,56" {
		t.Errorf("round trip = %q, quero %q", got, "R$ 1.234,56")
	}
}
