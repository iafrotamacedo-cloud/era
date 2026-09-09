package main

import (
	"math"
	"strings"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

func caso(nosso, osrm float64) Caso {
	return Caso{
		De: geo.Point{Lat: -3.7, Lon: -38.5}, Para: geo.Point{Lat: -3.8, Lon: -38.6},
		NossoMetros: nosso, NossoSegs: nosso / 10,
		OsrmMetros: osrm, OsrmSegs: osrm / 10,
	}
}

func TestErroRelativoTemSinal(t *testing.T) {
	// O sinal e o que separa "erramos por perfil" de "erramos por rota". Um
	// vies consistente para mais quer dizer que estamos pegando caminhos mais
	// longos; um vies para menos, que estamos passando por onde nao se passa.
	if e := caso(1100, 1000).ErroDistancia(); math.Abs(e-0.1) > 1e-12 {
		t.Errorf("mais longo = %v, esperado +0,1", e)
	}
	if e := caso(900, 1000).ErroDistancia(); math.Abs(e+0.1) > 1e-12 {
		t.Errorf("mais curto = %v, esperado -0,1", e)
	}
	if e := caso(1000, 0).ErroDistancia(); e != 0 {
		t.Errorf("divisao por zero = %v, esperado 0", e)
	}
}

func TestDistribuicaoSeparaAbsolutoDeSinal(t *testing.T) {
	// Metade erra 10% para mais, metade 10% para menos. O erro absoluto
	// mediano e 10%; o vies e zero. Um relatorio que so mostrasse um dos dois
	// contaria metade da historia.
	r := &Relatorio{}
	for i := 0; i < 50; i++ {
		r.Casos = append(r.Casos, caso(1100, 1000))
		r.Casos = append(r.Casos, caso(900, 1000))
	}

	d := r.distribuicao(Caso.ErroDistancia)
	if math.Abs(d.Mediana-0.1) > 1e-9 {
		t.Errorf("mediana absoluta = %.6f, esperada 0,1", d.Mediana)
	}
	if math.Abs(d.Vies) > 1e-9 {
		t.Errorf("vies = %.6f, esperado 0", d.Vies)
	}
	if math.Abs(d.Pior-0.1) > 1e-9 {
		t.Errorf("pior = %.6f, esperado 0,1", d.Pior)
	}
}

func TestDistribuicaoComVies(t *testing.T) {
	r := &Relatorio{}
	for i := 0; i < 100; i++ {
		r.Casos = append(r.Casos, caso(1050, 1000))
	}
	d := r.distribuicao(Caso.ErroDistancia)
	if math.Abs(d.Vies-0.05) > 1e-9 {
		t.Errorf("vies = %.6f, esperado 0,05", d.Vies)
	}
}

func TestPercentil(t *testing.T) {
	if got := percentil(nil, 0.5); got != 0 {
		t.Errorf("fatia vazia = %v", got)
	}
	if got := percentil([]float64{7}, 0.9); got != 7 {
		t.Errorf("um elemento = %v", got)
	}
	if got := percentil([]float64{1, 2, 3, 4}, 0.5); got != 2.5 {
		t.Errorf("mediana de 1,2,3,4 = %v, esperado 2,5", got)
	}
	dez := []float64{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	if got := percentil(dez, 0.90); got != 90 {
		t.Errorf("p90 = %v, esperado 90", got)
	}
	if got := percentil(dez, 0.99); math.Abs(got-99) > 1e-9 {
		t.Errorf("p99 = %v, esperado 99", got)
	}
}

// Os piores casos saem ordenados pelo erro absoluto, e nao pelo com sinal --
// uma rota 40% mais curta que a do OSRM e tao suspeita quanto uma 40% mais
// longa.
func TestPioresOrdenaPorErroAbsoluto(t *testing.T) {
	r := &Relatorio{Casos: []Caso{
		caso(1010, 1000), // +1%
		caso(600, 1000),  // -40%
		caso(1200, 1000), // +20%
		caso(1000, 1000), // 0
	}}

	piores := r.Piores(3)
	if len(piores) != 3 {
		t.Fatalf("%d piores, esperava 3", len(piores))
	}
	if math.Abs(piores[0].ErroDistancia()+0.4) > 1e-9 {
		t.Errorf("o pior deveria ser o de -40%%, veio %.4f", piores[0].ErroDistancia())
	}
	if math.Abs(piores[1].ErroDistancia()-0.2) > 1e-9 {
		t.Errorf("o segundo deveria ser o de +20%%, veio %.4f", piores[1].ErroDistancia())
	}

	// Pedir mais do que existe devolve o que existe.
	if len(r.Piores(99)) != len(r.Casos) {
		t.Error("Piores nao deveria estourar")
	}
	// E nao pode reordenar a fatia do relatorio.
	if r.Casos[0].NossoMetros != 1010 {
		t.Error("Piores embaralhou os casos do relatorio")
	}
}

func TestRelatorioVazio(t *testing.T) {
	r := &Relatorio{Tentados: 10, SemRotaOsrm: 10}
	saida := r.String()
	if !strings.Contains(saida, "nada a comparar") {
		t.Errorf("relatorio sem casos deveria dizer isso:\n%s", saida)
	}
	if !strings.Contains(saida, "10 sem rota no osrm") {
		t.Errorf("relatorio deveria contar os descartes:\n%s", saida)
	}
}

func TestRelatorioMostraOsNumeros(t *testing.T) {
	r := &Relatorio{Tentados: 3}
	r.Casos = append(r.Casos, caso(1100, 1000), caso(1000, 1000), caso(1050, 1000))

	saida := r.String()
	for _, querido := range []string{"3 pares tentados", "distancia", "tempo", "Piores casos"} {
		if !strings.Contains(saida, querido) {
			t.Errorf("faltou %q no relatorio:\n%s", querido, saida)
		}
	}
	// As coordenadas dos piores tem de aparecer: sao elas que dao para abrir
	// no mapa, e sem isso o relatorio nao ajuda a consertar nada.
	if !strings.Contains(saida, "-3.700000,-38.500000") {
		t.Errorf("faltaram as coordenadas dos piores:\n%s", saida)
	}
}
