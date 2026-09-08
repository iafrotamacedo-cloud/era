package kernel

import "sync"

// parallelFor divide o intervalo [0, n) em faixas e chama fn uma vez por
// faixa, cada uma na sua goroutine.
//
// A regra que torna isso seguro esta em quem chama, nao aqui: cada faixa
// precisa escrever apenas na sua propria area de saida. Nos kernels da ERA
// isso vale porque a divisao e feita pela dimensao de saida -- linhas da
// matriz no matmul, canais na depthwise -- e faixas diferentes nunca tocam
// a mesma posicao.
//
// Nao ha lock e nao ha coordenacao: as goroutines simplesmente nao se
// encontram.
func parallelFor(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}

	workers := min(numThreads, n)
	if workers <= 1 {
		fn(0, n)
		return
	}

	chunk := (n + workers - 1) / workers

	var wg sync.WaitGroup
	for start := 0; start < n; start += chunk {
		end := min(start+chunk, n)
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}

// ceilDiv devolve o teto da divisao a/b, com b > 0 e a possivelmente
// negativo.
//
// A divisao inteira do Go trunca em direcao ao zero, o que para numerador
// negativo ja e o teto; para positivo, e preciso somar b-1 antes.
//
// Serve para calcular, sem testar posicao por posicao, a partir de qual
// coluna a janela do kernel para de cair no padding.
func ceilDiv(a, b int) int {
	if a <= 0 {
		return -((-a) / b)
	}
	return (a + b - 1) / b
}
