package geo

import (
	"math"
	"math/rand"
	"testing"
)

func TestPointValid(t *testing.T) {
	casos := []struct {
		nome string
		p    Point
		want bool
	}{
		{"Fortaleza", fortaleza, true},
		{"origem", Point{0, 0}, true},
		{"polo norte", Point{90, 0}, true},
		{"polo sul", Point{-90, 180}, true},
		{"antimeridiano oeste", Point{0, -180}, true},
		{"latitude alem do polo", Point{90.1, 0}, false},
		{"latitude negativa alem do polo", Point{-90.1, 0}, false},
		{"longitude alem da volta", Point{0, 180.1}, false},
		{"lat e lon trocadas", Point{Lat: -38.5267, Lon: -3.7319}, true},
		{"NaN na latitude", Point{math.NaN(), 0}, false},
		{"NaN na longitude", Point{0, math.NaN()}, false},
		{"infinito", Point{math.Inf(1), 0}, false},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := c.p.Valid(); got != c.want {
				t.Errorf("%v.Valid() = %v, esperado %v", c.p, got, c.want)
			}
		})
	}
}

// O caso "lat e lon trocadas" acima passa de proposito, e vale explicar por
// que: as coordenadas de Fortaleza invertidas continuam sendo um ponto legal
// do planeta -- fica no oceano Indico, mas existe. Valid nao protege contra
// esse erro, e nenhuma checagem de faixa protegeria. Quem protege e o
// contexto: um Box da regiao de operacao rejeita o ponto na hora.
func TestBoxPegaCoordenadaTrocada(t *testing.T) {
	ceara := Box{MinLat: -8, MinLon: -42, MaxLat: -2, MaxLon: -37}

	if !ceara.Contains(fortaleza) {
		t.Error("Fortaleza deveria estar dentro do Ceara")
	}
	trocado := Point{Lat: fortaleza.Lon, Lon: fortaleza.Lat}
	if ceara.Contains(trocado) {
		t.Error("coordenada trocada passou pelo retangulo do Ceara")
	}
}

func TestNormalizeLon(t *testing.T) {
	casos := []struct{ in, want float64 }{
		{0, 0},
		{-38.5267, -38.5267},
		{179.9, 179.9},
		{-180, -180},
		{180, -180}, // +180 e -180 sao o mesmo lugar; escolhemos um
		{182, -178}, // dois graus depois do antimeridiano
		{-182, 178}, // dois graus antes
		{360, 0},    // volta inteira
		{-360, 0},   //
		{540, -180}, // uma volta e meia
		{720 + 45, 45},
	}
	for _, c := range casos {
		if got := NormalizeLon(c.in); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("NormalizeLon(%v) = %v, esperado %v", c.in, got, c.want)
		}
	}
}

func TestNormalizeLonSempreNaFaixa(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	for i := 0; i < 10000; i++ {
		lon := (r.Float64() - 0.5) * 4000
		got := NormalizeLon(lon)
		if got < -180 || got >= 180 {
			t.Fatalf("NormalizeLon(%v) = %v, fora de [-180, 180)", lon, got)
		}
		// Normalizar nao pode mudar o lugar: a diferenca tem de ser um
		// numero inteiro de voltas.
		voltas := (lon - got) / 360
		if math.Abs(voltas-math.Round(voltas)) > 1e-9 {
			t.Fatalf("NormalizeLon(%v) = %v moveu o ponto", lon, got)
		}
	}
}

func TestBoxContains(t *testing.T) {
	normal := Box{MinLat: -10, MinLon: -50, MaxLat: 10, MaxLon: -30}
	cruzado := Box{MinLat: -10, MinLon: 170, MaxLat: 10, MaxLon: -170}

	casos := []struct {
		nome string
		b    Box
		p    Point
		want bool
	}{
		{"dentro", normal, Point{0, -40}, true},
		{"na borda inferior", normal, Point{-10, -50}, true},
		{"na borda superior", normal, Point{10, -30}, true},
		{"latitude fora", normal, Point{20, -40}, false},
		{"longitude fora", normal, Point{0, -20}, false},

		{"cruzado, lado leste", cruzado, Point{0, 175}, true},
		{"cruzado, lado oeste", cruzado, Point{0, -175}, true},
		{"cruzado, em cima da linha", cruzado, Point{0, 180}, true},
		{"cruzado, do outro lado do mundo", cruzado, Point{0, 0}, false},
		{"cruzado, latitude fora", cruzado, Point{20, 175}, false},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := c.b.Contains(c.p); got != c.want {
				t.Errorf("Contains(%v) = %v, esperado %v", c.p, got, c.want)
			}
		})
	}
}

func TestBoxIntersects(t *testing.T) {
	a := Box{MinLat: -10, MinLon: -50, MaxLat: 10, MaxLon: -30}

	casos := []struct {
		nome string
		o    Box
		want bool
	}{
		{"sobreposto", Box{-5, -40, 5, -35}, true},
		{"encostando na borda", Box{-10, -30, 10, -10}, true},
		{"separado em longitude", Box{-10, -20, 10, -10}, false},
		{"separado em latitude", Box{20, -50, 30, -30}, false},
		{"contendo o outro", Box{-90, -180, 90, 180}, true},
		{"cruzando o antimeridiano, longe", Box{-10, 170, 10, -170}, false},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			if got := a.Intersects(c.o); got != c.want {
				t.Errorf("a.Intersects(o) = %v, esperado %v", got, c.want)
			}
			// Interseccao e simetrica; se nao for, ha um caso mal tratado.
			if got := c.o.Intersects(a); got != c.want {
				t.Errorf("o.Intersects(a) = %v, esperado %v (assimetria)", got, c.want)
			}
		})
	}
}

func TestBoxIntersectsAmbosCruzandoOAntimeridiano(t *testing.T) {
	// Dois retangulos que dao a volta pelo Pacifico contem ambos a linha de
	// 180 graus, entao se as latitudes se tocam eles necessariamente se
	// cruzam.
	a := Box{MinLat: -10, MinLon: 170, MaxLat: 10, MaxLon: -170}
	b := Box{MinLat: 0, MinLon: 100, MaxLat: 20, MaxLon: -100}
	if !a.Intersects(b) {
		t.Error("dois retangulos sobre o antimeridiano com latitudes comuns devem se cruzar")
	}

	c := Box{MinLat: 50, MinLon: 170, MaxLat: 60, MaxLon: -170}
	if a.Intersects(c) {
		t.Error("latitudes disjuntas nao podem se cruzar")
	}
}

// BoxAround so tem uma obrigacao: nao perder ninguem. Ele pode sobrar -- os
// cantos do retangulo ficam fora do circulo --, mas nao pode faltar, porque
// e ele que decide quais celulas da grade a busca vai olhar.
func TestBoxAroundNaoPerdeNinguem(t *testing.T) {
	r := rand.New(rand.NewSource(11))

	for i := 0; i < 500; i++ {
		centro := Point{Lat: r.Float64()*160 - 80, Lon: r.Float64()*360 - 180}
		raio := math.Pow(10, r.Float64()*6) // de 1 m a 1.000 km

		b := BoxAround(centro, raio)

		for grau := 0.0; grau < 360; grau += 5 {
			// Um ponto exatamente na borda do circulo.
			na := Destination(centro, grau, raio)
			if !b.Contains(na) {
				t.Fatalf("centro %v, raio %.0f m: ponto na borda %v ficou fora de %+v",
					centro, raio, na, b)
			}
			// E um bem dentro, para o caso do retangulo sair torto.
			dentro := Destination(centro, grau, raio*0.5)
			if !b.Contains(dentro) {
				t.Fatalf("centro %v, raio %.0f m: ponto interno %v ficou fora de %+v",
					centro, raio, dentro, b)
			}
		}
	}
}

func TestBoxAroundPertoDoPolo(t *testing.T) {
	// A 100 km do polo norte, um raio de 500 km ja da a volta em todas as
	// longitudes: o retangulo tem de abrir o mundo inteiro em longitude.
	b := BoxAround(Point{Lat: 89, Lon: 0}, 500_000)
	if b.MinLon != -180 || b.MaxLon != 180 {
		t.Errorf("perto do polo a faixa de longitude deveria ser total, veio %+v", b)
	}
	if b.MaxLat < 90 {
		t.Errorf("o retangulo deveria alcancar o polo, veio MaxLat = %v", b.MaxLat)
	}
}

func TestBoxAroundCruzandoOAntimeridiano(t *testing.T) {
	b := BoxAround(Point{Lat: 0, Lon: 179.9}, 50_000)
	if !b.CrossesAntimeridian() {
		t.Fatalf("esperava retangulo cruzando a linha de data, veio %+v", b)
	}
	if !b.Contains(Point{Lat: 0, Lon: -179.9}) {
		t.Error("o ponto do outro lado da linha deveria estar dentro")
	}
	if b.Contains(Point{Lat: 0, Lon: 0}) {
		t.Error("o lado oposto do mundo nao deveria estar dentro")
	}
}

func TestBoxCenter(t *testing.T) {
	b := Box{MinLat: -10, MinLon: -50, MaxLat: 10, MaxLon: -30}
	if got := b.Center(); got.Lat != 0 || got.Lon != -40 {
		t.Errorf("Center() = %v, esperado (0, -40)", got)
	}

	// Cruzando o antimeridiano, o meio de [170, -170] e 180, nao 0.
	c := Box{MinLat: 0, MinLon: 170, MaxLat: 0, MaxLon: -170}
	if got := c.Center(); math.Abs(math.Abs(got.Lon)-180) > 1e-9 {
		t.Errorf("Center() = %v, esperado longitude 180", got)
	}
}
