package extract

import "testing"

// cnpjValido foi calculado a mao pelo algoritmo da Receita Federal (ver o
// commit): base 112223330001, DV1=8 (soma 102, resto 3, 11-3), DV2=1 (soma
// 120, resto 10, 11-10). Serve de vetor de teste independente da
// implementacao -- se o codigo discordar deste numero, o bug esta no
// codigo, nao no teste.
const cnpjValido = "11.222.333/0001-81"

func TestValidCNPJVerdadeiro(t *testing.T) {
	for _, s := range []string{cnpjValido, "11222333000181"} {
		if !ValidCNPJ(s) {
			t.Errorf("ValidCNPJ(%q) = false, quero true", s)
		}
	}
}

func TestValidCNPJFalso(t *testing.T) {
	casos := []string{
		"11.222.333/0001-80", // ultimo digito errado
		"11.222.333/0001-18", // digitos verificadores trocados
		"1122233300018",      // 13 digitos, curto
		"112223330001812",    // 15 digitos, longo
		"abacate",
		"",
	}
	for _, s := range casos {
		if ValidCNPJ(s) {
			t.Errorf("ValidCNPJ(%q) = true, quero false", s)
		}
	}
}

func TestFormatCNPJ(t *testing.T) {
	got, err := FormatCNPJ("11222333000181")
	if err != nil {
		t.Fatalf("FormatCNPJ: %v", err)
	}
	if got != cnpjValido {
		t.Errorf("FormatCNPJ = %q, quero %q", got, cnpjValido)
	}
	if _, err := FormatCNPJ("123"); err == nil {
		t.Error("FormatCNPJ com poucos digitos deveria falhar")
	}
}

func TestFindCNPJs(t *testing.T) {
	texto := "Fornecedor: Distribuidora XYZ, CNPJ " + cnpjValido +
		". Pedido 998877. CNPJ invalido de teste: 11.222.333/0001-99."

	got := FindCNPJs(texto)
	if len(got) != 1 {
		t.Fatalf("FindCNPJs achou %d, quero 1 (so o valido): %v", len(got), got)
	}
	if got[0] != cnpjValido {
		t.Errorf("FindCNPJs[0] = %q, quero %q", got[0], cnpjValido)
	}
}

func TestFindCNPJsSemNenhum(t *testing.T) {
	if got := FindCNPJs("nenhum numero de documento aqui"); len(got) != 0 {
		t.Errorf("FindCNPJs em texto sem CNPJ = %v, quero vazio", got)
	}
}
