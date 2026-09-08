package geo

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// fortaleza e o ponto de referencia dos testes: e onde a operacao que
// motivou este pacote acontece, e fica perto do equador, onde o achatamento
// da Terra e mais visivel.
var fortaleza = Point{Lat: -3.7319, Lon: -38.5267}

func relativo(a, b float64) float64 {
	if b == 0 {
		return math.Abs(a)
	}
	return math.Abs(a-b) / math.Abs(b)
}

// Haversine e HaversineRef sao a mesma formula escrita de duas maneiras. Se
// divergirem, uma das duas foi mexida por engano.
func TestHaversineConcordaComReferencia(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 5000; i++ {
		a := Point{Lat: r.Float64()*180 - 90, Lon: r.Float64()*360 - 180}
		b := Point{Lat: r.Float64()*180 - 90, Lon: r.Float64()*360 - 180}
		if got, want := Haversine(a, b), HaversineRef(a, b); relativo(got, want) > 1e-9 {
			t.Fatalf("Haversine%v%v = %v, referencia = %v", a, b, got, want)
		}
	}
}

// Um grau de latitude tem o mesmo comprimento em qualquer lugar da esfera, e
// vale exatamente R * pi/180. E o unico valor que da para afirmar sem
// consultar tabela nenhuma, entao e por ele que se comeca.
func TestHaversineUmGrau(t *testing.T) {
	casos := []struct {
		nome string
		a, b Point
	}{
		{"latitude no equador", Point{0, 0}, Point{1, 0}},
		{"latitude no Ceara", Point{-3, -38}, Point{-4, -38}},
		{"latitude no Alasca", Point{60, -150}, Point{61, -150}},
		{"longitude no equador", Point{0, 0}, Point{0, 1}},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			got := Haversine(c.a, c.b)
			if relativo(got, metrosPorGrauLat) > 1e-9 {
				t.Errorf("= %.4f m, esperado %.4f m", got, metrosPorGrauLat)
			}
		})
	}
}

// Um grau de longitude encolhe com o cosseno da latitude: no equador vale um
// grau inteiro, a 60 graus vale metade.
//
// A comparacao tem uma sutileza que vale registrar. R*cos(lat) e o
// comprimento do arco *do paralelo* -- andar sempre para leste. Haversine
// mede o grande circulo, que e o caminho mais curto e passa por dentro, um
// pouco mais perto do polo. Entao o valor medido tem de ser menor que o do
// paralelo, e a diferenca cresce com a latitude: nula no equador, ~34 cm em
// 28,8 km a 75 graus. Nao e erro numerico -- e a razao pela qual voos entre
// cidades de mesma latitude sobem em direcao ao polo.
func TestHaversineLongitudeEncolheComALatitude(t *testing.T) {
	for _, lat := range []float64{0, 15, 30, 45, 60, 75} {
		got := Haversine(Point{lat, 0}, Point{lat, 1})
		paralelo := metrosPorGrauLat * math.Cos(lat*grausParaRad)

		if got > paralelo {
			t.Errorf("lat %.0f: grande circulo (%.2f m) maior que o paralelo (%.2f m)",
				lat, got, paralelo)
		}
		if relativo(got, paralelo) > 5e-5 {
			t.Errorf("lat %.0f: 1 grau de longitude = %.2f m, paralelo = %.2f m, "+
				"diferenca alem do atalho do grande circulo", lat, got, paralelo)
		}
	}
}

func TestHaversinePropriedades(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	ponto := func() Point {
		return Point{Lat: r.Float64()*180 - 90, Lon: r.Float64()*360 - 180}
	}

	for i := 0; i < 2000; i++ {
		a, b, c := ponto(), ponto(), ponto()

		if d := Haversine(a, a); d != 0 {
			t.Fatalf("distancia de um ponto a ele mesmo = %v, esperado 0", d)
		}
		ab, ba := Haversine(a, b), Haversine(b, a)
		if relativo(ab, ba) > 1e-12 {
			t.Fatalf("assimetria: d(a,b) = %v, d(b,a) = %v", ab, ba)
		}
		if ab < 0 || ab > meiaVoltaAoMundo*(1+1e-12) {
			t.Fatalf("distancia fora da faixa possivel: %v", ab)
		}
		// Desigualdade triangular, com folga para arredondamento.
		if ac, cb := Haversine(a, c), Haversine(c, b); ab > ac+cb+1e-6 {
			t.Fatalf("triangulo violado: d(a,b)=%v > d(a,c)+d(c,b)=%v", ab, ac+cb)
		}
	}
}

// A esfera nao e a Terra. Este teste mede o tamanho da mentira contra o
// elipsoide WGS-84 -- o mesmo modelo que o GPS usa -- em vez de afirma-lo num
// comentario.
func TestHaversineContraVincenty(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	var pior float64
	var piorA, piorB Point

	for i := 0; i < 5000; i++ {
		a := Point{Lat: r.Float64()*180 - 90, Lon: r.Float64()*360 - 180}
		b := Point{Lat: r.Float64()*180 - 90, Lon: r.Float64()*360 - 180}

		exato, err := Vincenty(a, b)
		if err != nil {
			continue // quase antipodais: mal condicionado, nao interessa aqui
		}
		if exato < 1000 {
			continue // erro relativo em distancias minusculas nao diz nada
		}
		if e := relativo(Haversine(a, b), exato); e > pior {
			pior, piorA, piorB = e, a, b
		}
	}

	t.Logf("erro maximo do modelo esferico: %.4f%% (entre %v e %v)", pior*100, piorA, piorB)
	if pior > 0.006 {
		t.Errorf("erro do modelo esferico = %.4f%%, acima do limite esperado", pior*100)
	}
}

// Valores de referencia externos ao pacote, do elipsoide WGS-84. Se Vincenty
// estivesse errada, todos os limites de erro medidos nos outros testes
// estariam medindo contra a coisa errada.
func TestVincentyContraValoresConhecidos(t *testing.T) {
	casos := []struct {
		nome string
		a, b Point
		want float64
	}{
		{"1 grau de latitude no equador", Point{0, 0}, Point{1, 0}, 110574.4},
		{"1 grau de longitude no equador", Point{0, 0}, Point{0, 1}, 111319.5},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			got, err := Vincenty(c.a, c.b)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(got-c.want) > 5 {
				t.Errorf("= %.1f m, esperado %.1f m", got, c.want)
			}
		})
	}
}

func TestVincentyPontosCoincidentes(t *testing.T) {
	d, err := Vincenty(fortaleza, fortaleza)
	if err != nil {
		t.Fatal(err)
	}
	if d != 0 {
		t.Errorf("= %v, esperado 0", d)
	}
}

// Pontos quase antipodais sao o caso em que o metodo de Vincenty nao fecha:
// a geodesica fica mal condicionada e a iteracao oscila. E um limite
// conhecido do metodo, nao um defeito desta implementacao -- e precisa
// devolver erro em vez de um numero inventado ou um laco infinito.
func TestVincentyAntipodalDevolveErro(t *testing.T) {
	_, err := Vincenty(Point{Lat: 0, Lon: 0}, Point{Lat: 0.5, Lon: 179.7})
	if err == nil {
		t.Fatal("esperava ErrVincentyNaoConvergiu para par quase antipodal")
	}
	if !errors.Is(err, ErrVincentyNaoConvergiu) {
		t.Fatalf("erro = %v, esperado ErrVincentyNaoConvergiu", err)
	}
}

// A lei dos cossenos e algebricamente igual a haversine e numericamente pior.
// Este teste existe para mostrar quanto pior, com numeros, e assim justificar
// a forma em que Haversine foi escrita.
func TestLeiDosCossenosQuebraEmDistanciaCurta(t *testing.T) {
	var pior float64
	for grau := 0; grau < 360; grau += 7 {
		// Um centimetro: dois pontos no mesmo endereco.
		b := Destination(fortaleza, float64(grau), 0.01)
		if e := relativo(LawOfCosinesRef(fortaleza, b), Haversine(fortaleza, b)); e > pior {
			pior = e
		}
	}
	t.Logf("erro da lei dos cossenos a 1 cm: %.1f%%", pior*100)
	if pior < 0.01 {
		t.Errorf("erro = %.4f%%; esperava-se degradacao clara, o teste perdeu o proposito", pior*100)
	}

	// A 100 km as duas concordam: o problema e so nas distancias curtas.
	b := Destination(fortaleza, 45, 100_000)
	if e := relativo(LawOfCosinesRef(fortaleza, b), Haversine(fortaleza, b)); e > 1e-9 {
		t.Errorf("a 100 km as duas formulas deveriam concordar; erro = %v", e)
	}
}

func TestBearingDirecoesCardeais(t *testing.T) {
	casos := []struct {
		nome string
		b    Point
		want float64
	}{
		{"norte", Point{fortaleza.Lat + 1, fortaleza.Lon}, 0},
		{"leste", Point{fortaleza.Lat, fortaleza.Lon + 1}, 90},
		{"sul", Point{fortaleza.Lat - 1, fortaleza.Lon}, 180},
		{"oeste", Point{fortaleza.Lat, fortaleza.Lon - 1}, 270},
	}
	for _, c := range casos {
		t.Run(c.nome, func(t *testing.T) {
			// Leste e oeste nao sao exatos: seguir o paralelo nao e seguir o
			// grande circulo. Meio grau de folga cobre a diferenca aqui.
			if got := Bearing(fortaleza, c.b); math.Abs(got-c.want) > 0.5 {
				t.Errorf("= %.4f graus, esperado %.0f", got, c.want)
			}
		})
	}
}

// Destination e Haversine/Bearing sao inversas. Fechar o ciclo confere as
// tres de uma vez, sem precisar de nenhum valor decorado.
func TestDestinationFechaOCiclo(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	for i := 0; i < 2000; i++ {
		origem := Point{Lat: r.Float64()*160 - 80, Lon: r.Float64()*360 - 180}
		azimute := r.Float64() * 360
		dist := r.Float64() * 2_000_000 // ate 2.000 km

		destino := Destination(origem, azimute, dist)

		if got := Haversine(origem, destino); relativo(got, dist) > 1e-9 {
			t.Fatalf("de %v, %.0f m no azimute %.1f: medida de volta = %.4f m",
				origem, dist, azimute, got)
		}
		if dist > 1000 {
			got := Bearing(origem, destino)
			if diff := math.Abs(got - azimute); diff > 1e-6 && math.Abs(diff-360) > 1e-6 {
				t.Fatalf("azimute de volta = %.9f, esperado %.9f", got, azimute)
			}
		}
	}
}

// Ruler troca exatidao por velocidade. Este teste mede o preco dentro da
// faixa de uso declarada: uma regiao do tamanho de um estado.
func TestRulerErroDentroDeUmEstado(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	ruler := NewRuler(fortaleza)

	ponto := func(alcanceM float64) Point {
		return Destination(fortaleza, r.Float64()*360, r.Float64()*alcanceM)
	}

	for _, alcance := range []float64{50_000, 250_000, 500_000, 1_000_000} {
		var pior float64
		for i := 0; i < 20000; i++ {
			a, b := ponto(alcance), ponto(alcance)
			exato := Haversine(a, b)
			if exato < 100 {
				continue
			}
			if e := relativo(ruler.Distance(a, b), exato); e > pior {
				pior = e
			}
		}
		t.Logf("raio de %7.0f km da referencia: erro maximo %.4f%%", alcance/1000, pior*100)
		if alcance <= 500_000 && pior > 0.01 {
			t.Errorf("raio de %.0f km: erro %.4f%%, acima de 1%%", alcance/1000, pior*100)
		}
	}
}

func TestRulerSquaredDistanceEhOQuadrado(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	ruler := NewRuler(fortaleza)
	for i := 0; i < 1000; i++ {
		a := Destination(fortaleza, r.Float64()*360, r.Float64()*200_000)
		b := Destination(fortaleza, r.Float64()*360, r.Float64()*200_000)
		d := ruler.Distance(a, b)
		if got := ruler.SquaredDistance(a, b); relativo(got, d*d) > 1e-12 {
			t.Fatalf("SquaredDistance = %v, Distance ao quadrado = %v", got, d*d)
		}
	}
}

// O antimeridiano e o unico lugar onde a aritmetica de longitude tem um
// degrau. Um passo de dois graus por cima dele nao pode virar uma volta de
// 358 graus ao mundo.
func TestDistanciasCruzandoOAntimeridiano(t *testing.T) {
	a := Point{Lat: 0, Lon: 179}
	b := Point{Lat: 0, Lon: -179}

	want := 2 * metrosPorGrauLat
	if got := Haversine(a, b); relativo(got, want) > 1e-9 {
		t.Errorf("Haversine = %.2f m, esperado %.2f m", got, want)
	}
	if got := NewRuler(a).Distance(a, b); relativo(got, want) > 1e-9 {
		t.Errorf("Ruler = %.2f m, esperado %.2f m", got, want)
	}
	if got, err := Vincenty(a, b); err != nil {
		t.Error(err)
	} else if math.Abs(got-222638) > 20 {
		t.Errorf("Vincenty = %.1f m, esperado cerca de 222.638 m", got)
	}
}
