package main

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Caso e um par comparado entre os dois motores.
type Caso struct {
	De, Para geo.Point

	NossoMetros float64
	NossoSegs   float64
	OsrmMetros  float64
	OsrmSegs    float64

	EncaixeMax float64 // o pior dos dois encaixes do OSRM
}

// ErroDistancia devolve o erro relativo da distancia, com sinal.
//
// Positivo quer dizer que a nossa rota e mais longa que a do OSRM.
func (c Caso) ErroDistancia() float64 {
	if c.OsrmMetros == 0 {
		return 0
	}
	return (c.NossoMetros - c.OsrmMetros) / c.OsrmMetros
}

// ErroTempo devolve o erro relativo do tempo, com sinal.
func (c Caso) ErroTempo() float64 {
	if c.OsrmSegs == 0 {
		return 0
	}
	return (c.NossoSegs - c.OsrmSegs) / c.OsrmSegs
}

// Relatorio junta o que a comparacao encontrou.
type Relatorio struct {
	Casos []Caso

	Tentados     int
	SemRotaNossa int
	SemRotaOsrm  int
	Erros        int
	Descartados  int // encaixe do OSRM longe demais para a comparacao valer
}

// Distribuicao resume os erros de uma grandeza.
type Distribuicao struct {
	Mediana float64
	P90     float64
	P99     float64
	Pior    float64
	Vies    float64 // mediana com sinal: diz se erramos para mais ou para menos
}

func (r *Relatorio) distribuicao(de func(Caso) float64) Distribuicao {
	if len(r.Casos) == 0 {
		return Distribuicao{}
	}

	absolutos := make([]float64, len(r.Casos))
	comSinal := make([]float64, len(r.Casos))
	for i, c := range r.Casos {
		v := de(c)
		absolutos[i] = math.Abs(v)
		comSinal[i] = v
	}
	sort.Float64s(absolutos)
	sort.Float64s(comSinal)

	return Distribuicao{
		Mediana: percentil(absolutos, 0.50),
		P90:     percentil(absolutos, 0.90),
		P99:     percentil(absolutos, 0.99),
		Pior:    absolutos[len(absolutos)-1],
		Vies:    percentil(comSinal, 0.50),
	}
}

// percentil interpola entre os dois vizinhos. Espera a fatia ja ordenada.
func percentil(ordenado []float64, p float64) float64 {
	if len(ordenado) == 0 {
		return 0
	}
	if len(ordenado) == 1 {
		return ordenado[0]
	}
	pos := p * float64(len(ordenado)-1)
	baixo := int(math.Floor(pos))
	alto := int(math.Ceil(pos))
	if baixo == alto {
		return ordenado[baixo]
	}
	peso := pos - float64(baixo)
	return ordenado[baixo]*(1-peso) + ordenado[alto]*peso
}

// Piores devolve os n casos de maior erro de distancia, do pior para o menos
// pior.
//
// Sao eles que valem alguma coisa. Uma mediana de 3% nao diz o que consertar;
// cinco pares com as coordenadas na mao dao para abrir no mapa e ver o que
// aconteceu -- e quase sempre e uma etiqueta lida errado, nao um erro de
// algoritmo.
func (r *Relatorio) Piores(n int) []Caso {
	ordenado := append([]Caso(nil), r.Casos...)
	sort.Slice(ordenado, func(i, j int) bool {
		return math.Abs(ordenado[i].ErroDistancia()) > math.Abs(ordenado[j].ErroDistancia())
	})
	if n > len(ordenado) {
		n = len(ordenado)
	}
	return ordenado[:n]
}

// String monta o relatorio legivel.
func (r *Relatorio) String() string {
	var b strings.Builder

	fmt.Fprintf(&b, "%d pares tentados, %d comparados\n", r.Tentados, len(r.Casos))
	if r.SemRotaNossa > 0 || r.SemRotaOsrm > 0 || r.Descartados > 0 || r.Erros > 0 {
		fmt.Fprintf(&b, "  descartados: %d sem rota aqui, %d sem rota no osrm, %d por encaixe distante, %d por erro\n",
			r.SemRotaNossa, r.SemRotaOsrm, r.Descartados, r.Erros)
	}
	if len(r.Casos) == 0 {
		b.WriteString("\nnada a comparar\n")
		return b.String()
	}

	d := r.distribuicao(Caso.ErroDistancia)
	t := r.distribuicao(Caso.ErroTempo)

	fmt.Fprintf(&b, "\n%-12s %9s %9s %9s %9s %9s\n", "", "mediana", "p90", "p99", "pior", "vies")
	fmt.Fprintf(&b, "%-12s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%%\n",
		"distancia", d.Mediana*100, d.P90*100, d.P99*100, d.Pior*100, d.Vies*100)
	fmt.Fprintf(&b, "%-12s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%%\n",
		"tempo", t.Mediana*100, t.P90*100, t.P99*100, t.Pior*100, t.Vies*100)

	b.WriteString("\nO tempo diverge mais que a distancia, e isso e esperado: o perfil de\n")
	b.WriteString("velocidade do OSRM e bem mais detalhado que o nosso, com penalidade de\n")
	b.WriteString("conversao e velocidade por tipo de via afinada. A distancia e a\n")
	b.WriteString("comparacao que fala do roteamento; o tempo fala do perfil.\n")

	piores := r.Piores(5)
	if len(piores) > 0 {
		b.WriteString("\nPiores casos, para abrir no mapa:\n")
		for _, c := range piores {
			fmt.Fprintf(&b, "  %+.6f,%+.6f -> %+.6f,%+.6f  %8.0f m aqui, %8.0f m no osrm  (%+.1f%%)\n",
				c.De.Lat, c.De.Lon, c.Para.Lat, c.Para.Lon,
				c.NossoMetros, c.OsrmMetros, c.ErroDistancia()*100)
		}
	}

	return b.String()
}
