package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/kernel"
)

func baseAleatoria(r *rand.Rand, n, dim int) []float32 {
	out := make([]float32, n*dim)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// TestPontuarBateComReferencia confere o produto matriz-vetor otimizado
// contra a versao obvia.
//
// Varre tamanhos que nao sao multiplos de 4 de proposito: o laco principal
// avanca de quatro em quatro, e a sobra e onde um erro de indice se
// esconderia.
func TestPontuarBateComReferencia(t *testing.T) {
	r := rand.New(rand.NewSource(4242))

	for _, dim := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 63, 128, 129, 512} {
		for _, n := range []int{0, 1, 2, 7, 64, 513} {
			t.Run(fmt.Sprintf("n=%d_dim=%d", n, dim), func(t *testing.T) {
				base := baseAleatoria(r, n, dim)
				q := baseAleatoria(r, 1, dim)

				got := make([]float32, n)
				want := make([]float32, n)

				pontuar(base, q, got, n, dim)
				pontuarRef(base, q, want, n, dim)

				for i := range want {
					// A soma em quatro parciais muda a ORDEM das operacoes, e
					// com ela os ultimos bits.
					//
					// A tolerancia acompanha a magnitude dos TERMOS somados, e
					// nao a do resultado. Com vetores de direcao arbitraria ha
					// cancelamento -- termos grandes que quase se anulam -- e
					// o resultado fica minusculo perto deles. Medir o erro
					// contra o resultado faria a tolerancia explodir
					// exatamente onde nada esta errado.
					var escala float32
					for j := 0; j < dim; j++ {
						termo := base[i*dim+j] * q[j]
						if termo < 0 {
							termo = -termo
						}
						escala += termo
					}

					d := got[i] - want[i]
					if d < 0 {
						d = -d
					}
					if d > 1e-6*escala+1e-6 {
						t.Fatalf("linha %d: %v, a referencia diz %v (magnitude dos termos: %v)",
							i, got[i], want[i], escala)
					}
				}
			})
		}
	}
}

// TestPontuarParaleloBateComSerial: acima de um limite o trabalho e dividido
// entre goroutines. A divisao nao pode mudar o resultado.
func TestPontuarParaleloBateComSerial(t *testing.T) {
	const dim = 128
	// Grande o bastante para passar do limite de paralelizacao.
	n := 4 * minimoParaParalelizar / dim

	r := rand.New(rand.NewSource(5))
	base := baseAleatoria(r, n, dim)
	q := baseAleatoria(r, 1, dim)

	paralelo := make([]float32, n)
	serial := make([]float32, n)

	pontuar(base, q, paralelo, n, dim)
	pontuarFaixa(base, q, serial, dim, 0, n)

	for i := range serial {
		if paralelo[i] != serial[i] {
			t.Fatalf("linha %d: paralelo %v, serial %v", i, paralelo[i], serial[i])
		}
	}
}

// Os benchmarks abaixo sustentam as escolhas de score.go. Todos moram no
// mesmo arquivo de proposito.
//
// Comparar numeros de execucoes DIFERENTES nao vale nesta maquina: ela tem
// carga variavel, e numa medicao anterior o mesmo laco apareceu 2,6 vezes
// mais lento so por isso -- o que quase levou a apagar uma otimizacao que
// funciona.
//
// n e dim vem de uma funcao, e nao de constantes, para que o compilador nao
// propague valores num caminho e no outro nao.
//
// Rode com:  go test ./faces/index/ -run='^$' -bench=Pontuar

func dadosDeBench() (base, q, out []float32, n, dim int) {
	n, dim = 1000, 128
	r := rand.New(rand.NewSource(1))
	return baseAleatoria(r, n, dim), baseAleatoria(r, 1, dim), make([]float32, n), n, dim
}

// BenchmarkPontuar mede o caminho em uso: desenrolado e paralelo.
func BenchmarkPontuar(b *testing.B) {
	base, q, out, n, dim := dadosDeBench()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pontuar(base, q, out, n, dim)
	}
}

// BenchmarkPontuarSerial isola o ganho do paralelismo: mesmo miolo, sem
// goroutines.
func BenchmarkPontuarSerial(b *testing.B) {
	base, q, out, n, dim := dadosDeBench()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pontuarFaixa(base, q, out, dim, 0, n)
	}
}

// BenchmarkPontuarRef isola o ganho do desenrolamento: mesmo laco, com uma
// soma so e sem goroutines.
func BenchmarkPontuarRef(b *testing.B) {
	base, q, out, n, dim := dadosDeBench()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pontuarRef(base, q, out, n, dim)
	}
}

// BenchmarkPontuarViaMatMul mede o caminho descartado: o produto
// matriz-matriz do kernel, com a segunda matriz de uma coluna so.
func BenchmarkPontuarViaMatMul(b *testing.B) {
	base, q, out, n, dim := dadosDeBench()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kernel.MatMul(base, q, out, n, dim, 1)
	}
}
