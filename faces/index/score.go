package index

import (
	"runtime"
	"sync"
)

// Pontuar todos os vetores cadastrados contra uma consulta e um produto
// matriz-vetor: [n, dim] x [dim] -> [n].
//
// # Por que nao usar o kernel.MatMul
//
// A tentacao era obvia -- o kernel ja e bloqueado, paralelo e testado. Mas
// ele e desenhado para matriz por matriz, e o laco mais interno dele percorre
// a dimensao de SAIDA. Numa consulta essa dimensao tem tamanho 1: o miolo
// otimizado roda uma iteracao e o que sobra e o custo de fatiar slices a cada
// passo.
//
// Medido com 1000 vetores de 128 dimensoes, tudo no mesmo binario
// (BenchmarkPontuar e vizinhos, em score_test.go):
//
//	ingenuo, serial                 119,4 us
//	via kernel.MatMul               100,9 us
//	desenrolado, serial              65,6 us
//	desenrolado e paralelo, em uso   43,2 us
//
// E a mesma armadilha que travava a convolucao depthwise em 0,86 GFLOPS, e
// que o comentario de nn.Linear diz ter aprendido. Reaproveitar codigo bom
// numa forma para a qual ele nao foi feito custa mais que escrever o pouco
// codigo certo.

// minimoParaParalelizar e quantos produtos escalares precisam existir para
// que valha a pena acordar goroutines.
//
// Abaixo disso o custo de criar e sincronizar supera o trabalho. Com 128
// dimensoes, sao 500 vetores.
const minimoParaParalelizar = 64 * 1024

// pontuar calcula o produto escalar da consulta com cada linha de base.
//
// Todos os vetores estao normalizados, entao cada produto escalar JA e a
// similaridade de cosseno -- nao ha divisao a fazer.
func pontuar(base, consulta, out []float32, n, dim int) {
	if n <= 0 || dim <= 0 {
		return
	}

	trabalhadores := 1
	if n*dim >= minimoParaParalelizar {
		trabalhadores = min(runtime.NumCPU(), n)
	}

	if trabalhadores <= 1 {
		pontuarFaixa(base, consulta, out, dim, 0, n)
		return
	}

	fatia := (n + trabalhadores - 1) / trabalhadores
	var wg sync.WaitGroup
	for inicio := 0; inicio < n; inicio += fatia {
		fim := min(inicio+fatia, n)
		wg.Add(1)
		go func(i, f int) {
			defer wg.Done()
			pontuarFaixa(base, consulta, out, dim, i, f)
		}(inicio, fim)
	}
	wg.Wait()
}

// pontuarFaixa resolve as linhas [inicio, fim).
//
// O acumulo e quebrado em quatro somas parciais de proposito. Num produto
// escalar ingenuo cada soma depende do resultado da anterior, e a unidade de
// ponto flutuante fica ociosa esperando -- a latencia da soma, e nao a
// vazao, vira o limite. Quatro cadeias independentes mantem o processador
// ocupado.
//
// A ordem da soma muda, e com ela os ultimos bits do resultado. Para
// similaridade de cosseno isso e irrelevante; os testes comparam com a
// versao ingenua dentro da tolerancia de float32.
func pontuarFaixa(base, consulta, out []float32, dim, inicio, fim int) {
	q := consulta[:dim]

	for i := inicio; i < fim; i++ {
		linha := base[i*dim : i*dim+dim : i*dim+dim]

		var s0, s1, s2, s3 float32
		j := 0
		for ; j+4 <= dim; j += 4 {
			s0 += linha[j] * q[j]
			s1 += linha[j+1] * q[j+1]
			s2 += linha[j+2] * q[j+2]
			s3 += linha[j+3] * q[j+3]
		}
		for ; j < dim; j++ {
			s0 += linha[j] * q[j]
		}

		out[i] = (s0 + s1) + (s2 + s3)
	}
}

// pontuarRef e a versao obvia, para os testes conferirem a otimizada.
//
// Nunca otimize esta funcao: o valor dela e ser simples o bastante para dar
// para ler e afirmar que esta certa.
func pontuarRef(base, consulta, out []float32, n, dim int) {
	for i := 0; i < n; i++ {
		var s float32
		for j := 0; j < dim; j++ {
			s += base[i*dim+j] * consulta[j]
		}
		out[i] = s
	}
}
