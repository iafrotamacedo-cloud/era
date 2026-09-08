// Package kernel concentra as operacoes numericas pesadas da ERA.
//
// Praticamente todo o tempo de execucao de uma rede neural cai aqui: uma
// convolucao vira multiplicacao de matrizes (ver Im2Col), e uma camada
// linear ja e multiplicacao de matrizes. Se este pacote estiver correto e
// rapido, a ERA esta correta e rapida.
package kernel

import (
	"fmt"
	"runtime"
)

// Tamanho dos blocos do matmul, em elementos.
//
// A ideia do bloqueio e simples: em vez de varrer as matrizes inteiras, o
// laco processa pedacos pequenos o bastante para caberem no cache L1/L2 do
// processador. As contas sao exatamente as mesmas -- so a ordem muda -- mas
// o processador para de esperar a memoria RAM a cada acesso.
//
// Estes valores miram um L1 de 32 KB e um L2 de 256 KB a 1 MB, que e o
// comum em x86-64 e ARM modernos.
const (
	tileM = 64  // linhas de A/C por bloco
	tileN = 256 // colunas de B/C por bloco
	tileK = 128 // profundidade do produto por bloco
)

// numThreads controla o paralelismo dos kernels. Zero ou negativo significa
// "use todos os nucleos".
var numThreads = runtime.NumCPU()

// SetThreads define quantas goroutines os kernels usam. Passar n <= 0
// restaura o padrao, que e um por nucleo logico.
func SetThreads(n int) {
	if n <= 0 {
		n = runtime.NumCPU()
	}
	numThreads = n
}

// Threads informa o paralelismo em vigor.
func Threads() int { return numThreads }

// MatMul calcula C = A x B.
//
// Todas as matrizes sao row-major e contiguas:
//
//	A: m x k    B: k x n    C: m x n
//
// C e sobrescrita por completo -- nao precisa vir zerada.
//
// O trabalho e dividido entre goroutines por faixas de linhas de C. Como
// cada goroutine escreve apenas nas suas proprias linhas, nao ha escrita
// compartilhada, nao ha lock e nao ha corrida.
func MatMul(a, b, c []float32, m, k, n int) {
	if m < 0 || k < 0 || n < 0 {
		panic(fmt.Sprintf("kernel: dimensoes negativas m=%d k=%d n=%d", m, k, n))
	}
	if len(a) < m*k {
		panic(fmt.Sprintf("kernel: A tem %d elementos, precisa de %d (%dx%d)", len(a), m*k, m, k))
	}
	if len(b) < k*n {
		panic(fmt.Sprintf("kernel: B tem %d elementos, precisa de %d (%dx%d)", len(b), k*n, k, n))
	}
	if len(c) < m*n {
		panic(fmt.Sprintf("kernel: C tem %d elementos, precisa de %d (%dx%d)", len(c), m*n, m, n))
	}
	if m == 0 || n == 0 {
		return
	}

	clear(c[:m*n])
	if k == 0 {
		return // C = A x B com k=0 e a matriz zero
	}

	parallelFor(m, func(start, end int) {
		matmulRange(a, b, c, k, n, start, end)
	})
}

// matmulRange acumula em C as linhas [rowStart, rowEnd).
//
// A ordem dos lacos e i -> k -> j de proposito. No laco mais interno, tanto
// B quanto C sao percorridos ao longo de j, que e a dimensao contigua na
// memoria. Isso transforma o miolo num "axpy" -- soma de vetor escalado --
// que e sequencial, previsivel e amigavel ao prefetcher do processador.
//
// A alternativa ingenua (i -> j -> k) percorre B pulando uma linha inteira a
// cada passo, provocando um cache miss por elemento.
func matmulRange(a, b, c []float32, k, n, rowStart, rowEnd int) {
	for i0 := rowStart; i0 < rowEnd; i0 += tileM {
		i1 := min(i0+tileM, rowEnd)

		for k0 := 0; k0 < k; k0 += tileK {
			k1 := min(k0+tileK, k)

			for j0 := 0; j0 < n; j0 += tileN {
				j1 := min(j0+tileN, n)
				width := j1 - j0

				for i := i0; i < i1; i++ {
					crow := c[i*n+j0 : i*n+j1]
					arow := a[i*k+k0 : i*k+k1]

					for kk, av := range arow {
						brow := b[(k0+kk)*n+j0 : (k0+kk)*n+j1]

						// Reatribuir com o mesmo comprimento de brow deixa
						// o compilador provar que o indice esta dentro dos
						// limites, e ele elimina a verificacao no miolo.
						cs := crow[:width]
						for j, bv := range brow {
							cs[j] += av * bv
						}
					}
				}
			}
		}
	}
}

// MatMulAdd calcula C += A x B, preservando o conteudo anterior de C.
//
// Util para acumular contribuicoes, por exemplo em convolucoes agrupadas.
func MatMulAdd(a, b, c []float32, m, k, n int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if len(a) < m*k || len(b) < k*n || len(c) < m*n {
		panic("kernel: MatMulAdd recebeu buffer pequeno demais")
	}

	parallelFor(m, func(start, end int) {
		matmulRange(a, b, c, k, n, start, end)
	})
}
