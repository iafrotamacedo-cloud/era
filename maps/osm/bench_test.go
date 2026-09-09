package osm

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

var sink int

// arquivoGrande monta um .osm.pbf com a proporcao de um extrato real: muitos
// nos, poucos com etiqueta, e vias ligando-os.
//
// Blocos de 8.000 entidades, que e o limite que as ferramentas oficiais usam.
//
// Os passos entre coordenadas sao sorteados, e isso nao e detalhe. A primeira
// versao deste gerador andava em linha reta com passo constante: as
// diferencas ficavam todas iguais, o zlib esmagava o arquivo a 175 KB para
// 200.000 nos, e o benchmark media inflate sobre dado que nao existe no mundo.
// Extrato de verdade tem passo irregular, e e o custo dele que interessa.
func arquivoGrande(nos, vias int) []byte {
	a := arquivoPadrao()
	r := rand.New(rand.NewSource(7))

	const porBloco = 8000
	id := int64(1)
	lat, lon := -3.0, -38.0

	for restam := nos; restam > 0; {
		n := porBloco
		if restam < n {
			n = restam
		}
		bloco := make([]Node, n)
		for i := range bloco {
			bloco[i] = Node{ID: id, Point: geo.Point{Lat: lat, Lon: lon}}
			// Uma etiqueta a cada 50 nos, que e a ordem de grandeza real.
			if id%50 == 0 {
				bloco[i].Tags = Tags{{Key: "highway", Value: "crossing"}}
			}
			id += 1 + int64(r.Intn(4)) // ids reais tem buracos
			lat -= r.Float64() * 0.0001
			lon += r.Float64() * 0.0001
		}
		a.blocoDenso(bloco)
		restam -= n
	}

	for restam := vias; restam > 0; {
		n := porBloco
		if restam < n {
			n = restam
		}
		bloco := make([]Way, n)
		for i := range bloco {
			base := int64(r.Intn(nos) + 1)
			refs := make([]int64, 2+r.Intn(20)) // vias reais variam muito
			for j := range refs {
				base += int64(1 + r.Intn(50))
				refs[j] = base
			}
			bloco[i] = Way{
				ID:   int64(i + 1),
				Refs: refs,
				Tags: Tags{
					{Key: "highway", Value: "residential"},
					{Key: "name", Value: fmt.Sprintf("Rua %d", i)},
				},
			}
		}
		a.blocoVias(bloco)
		restam -= n
	}

	return a.bytes()
}

func BenchmarkScanTudo(b *testing.B) {
	dados := arquivoGrande(200_000, 20_000)
	b.SetBytes(int64(len(dados)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		n := 0
		err := Scan(bytes.NewReader(dados), Handler{
			Node: func(Node) error { n++; return nil },
			Way:  func(Way) error { n++; return nil },
		})
		if err != nil {
			b.Fatal(err)
		}
		sink = n
	}
	b.ReportMetric(float64(220_000*b.N)/b.Elapsed().Seconds(), "entidades/s")
}

// A comparacao que justifica o Handler ser um struct de funcoes em vez de uma
// interface: com Node nil, o leitor nao decodifica os nos -- que sao a maior
// parte do arquivo -- em vez de decodificar para jogar fora.
//
// E a primeira das duas passadas que montar um grafo rodoviario exige.
func BenchmarkScanSoVias(b *testing.B) {
	dados := arquivoGrande(200_000, 20_000)
	b.SetBytes(int64(len(dados)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		n := 0
		err := Scan(bytes.NewReader(dados), Handler{
			Way: func(Way) error { n++; return nil },
		})
		if err != nil {
			b.Fatal(err)
		}
		sink = n
	}
}

// O piso: percorrer o arquivo sem decodificar entidade nenhuma. Mede o custo
// do envelope e do inflate, que e o que sobra depois de pular tudo.
func BenchmarkScanNada(b *testing.B) {
	dados := arquivoGrande(200_000, 20_000)
	b.SetBytes(int64(len(dados)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := Scan(bytes.NewReader(dados), Handler{}); err != nil {
			b.Fatal(err)
		}
	}
}

// Quanto o paralelismo rende. Descomprimir e o gargalo, e os blocos sao
// independentes -- mas a leitura do arquivo e serial, entao ha um teto.
//
// Arquivo maior que o dos outros benchmarks de proposito. Com poucos blocos o
// escalonamento fica irregular -- um bloco grande no comeco segura a entrega
// ordenada de todos os que vem depois -- e o numero mede mais o sorteio de
// tamanhos do que o paralelismo.
func BenchmarkScanParalelismo(b *testing.B) {
	dados := arquivoGrande(1_000_000, 100_000)

	for _, p := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("%d", p), func(b *testing.B) {
			b.SetBytes(int64(len(dados)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				n := 0
				err := ScanN(bytes.NewReader(dados), Handler{
					Node: func(Node) error { n++; return nil },
					Way:  func(Way) error { n++; return nil },
				}, p)
				if err != nil {
					b.Fatal(err)
				}
				sink = n
			}
		})
	}
}

// Isola onde esta o teto do paralelismo.
//
// Este percorre o mesmo arquivo grande sem entregar entidade nenhuma: mede so
// o envelope, o inflate e a tabela de strings -- que sao a parte paralela.
// Comparado com BenchmarkScanParalelismo, que entrega tudo, a diferenca de
// escalonamento diz quanto do tempo esta preso na entrega serial.
func BenchmarkScanNadaParalelismo(b *testing.B) {
	dados := arquivoGrande(1_000_000, 100_000)

	for _, p := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("%d", p), func(b *testing.B) {
			b.SetBytes(int64(len(dados)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := ScanN(bytes.NewReader(dados), Handler{}, p); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
