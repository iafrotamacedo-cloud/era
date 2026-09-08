package geo

import (
	"math/rand"
	"sort"
	"testing"
)

// varreduraLinear e a implementacao de referencia da grade: olha todos os
// pontos, um por um. E obviamente correta e obviamente lenta -- exatamente o
// que se quer de uma referencia.
func varreduraLinear(ids []int, pts []Point, centro Point, raioM float64) []Match {
	var out []Match
	for i, p := range pts {
		if d := Haversine(centro, p); d <= raioM {
			out = append(out, Match{ID: ids[i], Point: p, Meters: d})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Meters < out[j].Meters })
	return out
}

func idsDe(ms []Match) []int {
	out := make([]int, len(ms))
	for i, m := range ms {
		out[i] = m.ID
	}
	sort.Ints(out)
	return out
}

func mesmosIDs(a, b []Match) bool {
	x, y := idsDe(a), idsDe(b)
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

// pontosEspalhados devolve n pontos dentro de um raio em torno de um centro,
// no formato em que a grade e a varredura linear consomem.
func pontosEspalhados(r *rand.Rand, centro Point, n int, alcanceM float64) ([]int, []Point) {
	ids := make([]int, n)
	pts := make([]Point, n)
	for i := 0; i < n; i++ {
		ids[i] = i
		pts[i] = Destination(centro, r.Float64()*360, r.Float64()*alcanceM)
	}
	return ids, pts
}

// O teste central da grade: para a mesma pergunta, ela tem de dar
// exatamente a mesma resposta da varredura linear. Toda a razao de existir
// da estrutura e ser mais rapida sem mudar o resultado.
func TestGridWithinBateComVarreduraLinear(t *testing.T) {
	r := rand.New(rand.NewSource(20))
	ids, pts := pontosEspalhados(r, fortaleza, 3000, 400_000)

	// Varios tamanhos de celula, inclusive absurdamente grandes e pequenos
	// para o problema: o resultado nao pode depender dessa escolha, so o
	// tempo.
	for _, celula := range []float64{0.01, 0.05, 0.2, 1.0, 5.0} {
		g := NewGrid(celula)
		for i := range pts {
			g.Add(ids[i], pts[i])
		}

		for consulta := 0; consulta < 200; consulta++ {
			centro := Destination(fortaleza, r.Float64()*360, r.Float64()*400_000)
			raio := r.Float64() * 100_000

			got := g.Within(centro, raio)
			want := varreduraLinear(ids, pts, centro, raio)

			if !mesmosIDs(got, want) {
				t.Fatalf("celula %v, centro %v, raio %.0f m: grade achou %d pontos, varredura achou %d",
					celula, centro, raio, len(got), len(want))
			}
			for i := 1; i < len(got); i++ {
				if got[i-1].Meters > got[i].Meters {
					t.Fatalf("resultado fora de ordem na posicao %d", i)
				}
			}
		}
	}
}

func TestGridNearestBateComVarreduraLinear(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	ids, pts := pontosEspalhados(r, fortaleza, 2000, 300_000)

	g := NewGridForRadius(10_000)
	for i := range pts {
		g.Add(ids[i], pts[i])
	}

	todos := varreduraLinear(ids, pts, fortaleza, meiaVoltaAoMundo)

	for _, k := range []int{1, 3, 10, 100, 2000, 5000} {
		centro := fortaleza
		got := g.Nearest(centro, k)

		want := varreduraLinear(ids, pts, centro, meiaVoltaAoMundo)
		if k < len(want) {
			want = want[:k]
		}

		if len(got) != len(want) {
			t.Fatalf("k=%d: Nearest devolveu %d, esperado %d", k, len(got), len(want))
		}
		for i := range got {
			if relativo(got[i].Meters, want[i].Meters) > 1e-12 {
				t.Fatalf("k=%d, posicao %d: distancia %v, esperada %v",
					k, i, got[i].Meters, want[i].Meters)
			}
		}
	}

	if len(todos) != 2000 {
		t.Fatalf("a varredura deveria ver os 2000 pontos, viu %d", len(todos))
	}
}

// Nearest a partir de pontos aleatorios, nao so do centro da nuvem: e onde a
// expansao do raio tem de dar mais voltas antes de encontrar k pontos.
func TestGridNearestDeLongeDaNuvem(t *testing.T) {
	r := rand.New(rand.NewSource(22))
	ids, pts := pontosEspalhados(r, fortaleza, 500, 50_000)

	g := NewGridForRadius(1_000) // celula bem menor que a distancia da consulta
	for i := range pts {
		g.Add(ids[i], pts[i])
	}

	// Sao Paulo: a uns 2.300 km da nuvem de pontos.
	distante := Point{Lat: -23.5505, Lon: -46.6333}

	got := g.Nearest(distante, 5)
	want := varreduraLinear(ids, pts, distante, meiaVoltaAoMundo)[:5]

	for i := range got {
		if got[i].ID != want[i].ID {
			t.Fatalf("posicao %d: id %d, esperado %d", i, got[i].ID, want[i].ID)
		}
	}
}

func TestGridVazio(t *testing.T) {
	g := NewGridForRadius(10_000)

	if g.Len() != 0 || g.Cells() != 0 {
		t.Errorf("grade nova tem %d pontos em %d celulas", g.Len(), g.Cells())
	}
	if got := g.Within(fortaleza, 100_000); got != nil {
		t.Errorf("Within numa grade vazia = %v, esperado nil", got)
	}
	if got := g.Nearest(fortaleza, 5); got != nil {
		t.Errorf("Nearest numa grade vazia = %v, esperado nil", got)
	}
}

func TestGridConsultasDegeneradas(t *testing.T) {
	g := NewGridForRadius(10_000)
	g.Add(1, fortaleza)

	if got := g.Within(fortaleza, 0); got != nil {
		t.Errorf("raio zero = %v, esperado nil", got)
	}
	if got := g.Within(fortaleza, -1); got != nil {
		t.Errorf("raio negativo = %v, esperado nil", got)
	}
	if got := g.Nearest(fortaleza, 0); got != nil {
		t.Errorf("k zero = %v, esperado nil", got)
	}
	if got := g.Nearest(fortaleza, -3); got != nil {
		t.Errorf("k negativo = %v, esperado nil", got)
	}

	// O proprio ponto, a distancia zero, tem de aparecer.
	if got := g.Within(fortaleza, 1); len(got) != 1 || got[0].Meters != 0 {
		t.Errorf("o ponto no centro da busca deveria aparecer com distancia 0, veio %v", got)
	}
}

// A grade e um mapa de celulas indexado por longitude; o antimeridiano e a
// costura desse mapa. Dois pontos separados por poucos quilometros mas em
// lados opostos da costura tem de continuar sendo vizinhos.
func TestGridAtravessaOAntimeridiano(t *testing.T) {
	g := NewGrid(0.5)

	oeste := Point{Lat: 0, Lon: 179.8}
	leste := Point{Lat: 0, Lon: -179.8}
	longe := Point{Lat: 0, Lon: 0}

	g.Add(1, oeste)
	g.Add(2, leste)
	g.Add(3, longe)

	got := g.Within(oeste, 100_000) // ~100 km cobre os 0,4 grau entre os dois
	if len(got) != 2 {
		t.Fatalf("esperava 2 vizinhos atraves da linha de data, veio %v", got)
	}
	if ids := idsDe(got); ids[0] != 1 || ids[1] != 2 {
		t.Errorf("ids = %v, esperado [1 2]", ids)
	}
}

func TestGridPertoDoPolo(t *testing.T) {
	g := NewGrid(1.0)

	// Um anel de pontos em volta do polo norte, em longitudes opostas. Eles
	// estao a poucas centenas de quilometros entre si passando por cima do
	// polo, embora a longitude difira em 180 graus.
	g.Add(1, Point{Lat: 89.5, Lon: 0})
	g.Add(2, Point{Lat: 89.5, Lon: 180})
	g.Add(3, Point{Lat: 89.5, Lon: 90})

	got := g.Within(Point{Lat: 89.9, Lon: 0}, 200_000)
	if len(got) != 3 {
		t.Fatalf("perto do polo todos os tres deveriam aparecer, veio %d: %v", len(got), got)
	}
}

func TestGridPontoInvalidoEntraEmPanico(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("esperava panico ao registrar ponto invalido")
		}
	}()
	NewGridForRadius(1000).Add(1, Point{Lat: 200, Lon: 0})
}

func TestNewGridLadoInvalidoEntraEmPanico(t *testing.T) {
	for _, lado := range []float64{0, -1, 91} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("esperava panico para celula de %v graus", lado)
				}
			}()
			NewGrid(lado)
		}()
	}
}

func TestNewGridForRadiusLimita(t *testing.T) {
	// Raio absurdamente pequeno: a celula nao pode virar zero nem negativa.
	if g := NewGridForRadius(0.000001); g.cell <= 0 || g.cols <= 0 || g.rows <= 0 {
		t.Errorf("grade degenerada para raio minusculo: %+v", g)
	}
	// Raio maior que o planeta: a celula tem de parar num tamanho util.
	if g := NewGridForRadius(1e12); g.cell > 90 {
		t.Errorf("celula de %v graus, acima do limite", g.cell)
	}
}

func TestGridContabilidade(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	g := NewGrid(0.1)

	_, pts := pontosEspalhados(r, fortaleza, 500, 100_000)
	for i, p := range pts {
		g.Add(i, p)
	}

	if g.Len() != 500 {
		t.Errorf("Len() = %d, esperado 500", g.Len())
	}
	if g.Cells() < 2 || g.Cells() > 500 {
		t.Errorf("Cells() = %d; esperava os pontos espalhados por varias celulas", g.Cells())
	}

	// Registrar o mesmo id duas vezes cria duas entradas -- comportamento
	// documentado, e vale ter um teste que o fixa.
	g.Add(0, fortaleza)
	g.Add(0, fortaleza)
	if got := g.Within(fortaleza, 1); len(got) != 2 {
		t.Errorf("dois Add com o mesmo id deveriam gerar duas entradas, veio %d", len(got))
	}
}
