package extract

// pesoSoma calcula a soma ponderada usada no calculo dos digitos
// verificadores de CNPJ e CPF -- mesma conta nos dois documentos, so os
// pesos mudam.
func pesoSoma(digitos, pesos []int) int {
	soma := 0
	for i, d := range digitos {
		soma += d * pesos[i]
	}
	return soma
}

// digitoVerificador aplica a regra da Receita Federal: resto da soma
// ponderada por 11; menor que 2 vira 0, senao e 11 menos o resto.
func digitoVerificador(digitos, pesos []int) int {
	resto := pesoSoma(digitos, pesos) % 11
	if resto < 2 {
		return 0
	}
	return 11 - resto
}
