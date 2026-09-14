package extract

import "testing"

// cpfValido foi calculado a mao pelo algoritmo da Receita Federal: base
// 111444777, DV1=3 (soma 162, resto 8, 11-8), DV2=5 (soma 204, resto 6,
// 11-6). E tambem um numero amplamente usado como exemplo em tutoriais e
// testes de validador de CPF em portugues -- as duas fontes concordam.
const cpfValido = "111.444.777-35"

func TestValidCPFVerdadeiro(t *testing.T) {
	for _, s := range []string{cpfValido, "11144477735"} {
		if !ValidCPF(s) {
			t.Errorf("ValidCPF(%q) = false, quero true", s)
		}
	}
}

func TestValidCPFFalso(t *testing.T) {
	casos := []string{
		"111.444.777-36", // ultimo digito errado
		"11144477",       // curto
		"111444777356",   // longo
		"abacate",
		"",
	}
	for _, s := range casos {
		if ValidCPF(s) {
			t.Errorf("ValidCPF(%q) = true, quero false", s)
		}
	}
}

// TestValidCPFRejeitaDigitosRepetidos confere a checagem extra que a
// formula sozinha nao faz: 111.111.111-11 fecha o digito verificador
// certinho (a soma ponderada de 11 copias do mesmo digito ainda cai num
// resto que funciona), mas nenhum CPF de verdade e assim.
func TestValidCPFRejeitaDigitosRepetidos(t *testing.T) {
	for d := '0'; d <= '9'; d++ {
		s := string(d) + string(d) + string(d) + string(d) + string(d) +
			string(d) + string(d) + string(d) + string(d) + string(d) + string(d)
		if ValidCPF(s) {
			t.Errorf("ValidCPF(%q) = true, quero false (digitos repetidos)", s)
		}
	}
}

func TestFormatCPF(t *testing.T) {
	got, err := FormatCPF("11144477735")
	if err != nil {
		t.Fatalf("FormatCPF: %v", err)
	}
	if got != cpfValido {
		t.Errorf("FormatCPF = %q, quero %q", got, cpfValido)
	}
	if _, err := FormatCPF("123"); err == nil {
		t.Error("FormatCPF com poucos digitos deveria falhar")
	}
}

func TestFindCPFs(t *testing.T) {
	texto := "Comprador: Fulano de Tal, CPF " + cpfValido + ". CPF invalido: 111.444.777-99."
	got := FindCPFs(texto)
	if len(got) != 1 {
		t.Fatalf("FindCPFs achou %d, quero 1: %v", len(got), got)
	}
	if got[0] != cpfValido {
		t.Errorf("FindCPFs[0] = %q, quero %q", got[0], cpfValido)
	}
}
