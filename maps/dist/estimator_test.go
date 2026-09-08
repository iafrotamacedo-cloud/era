package dist

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

func estimadorDeTeste(t *testing.T) *Estimator {
	t.Helper()
	e, err := NewEstimator(DefaultCalibration())
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestNewEstimatorRejeitaCalibracaoVazia(t *testing.T) {
	if _, err := NewEstimator(Calibration{}); !errors.Is(err, ErrCalibracaoInvalida) {
		t.Errorf("erro = %v, esperado ErrCalibracaoInvalida", err)
	}
	if _, err := NewEstimator(Calibration{Factor: 1.3}); !errors.Is(err, ErrCalibracaoInvalida) {
		t.Errorf("calibracao sem velocidade deveria ser rejeitada, erro = %v", err)
	}
}

// A distancia estimada e a linha reta vezes o fator da faixa. Este teste
// confere a conta contra o geo, que e a fonte da linha reta.
func TestDistanceAplicaOFatorDaFaixa(t *testing.T) {
	cal := DefaultCalibration()
	e := estimadorDeTeste(t)
	ctx := context.Background()

	for _, crowQuerido := range []float64{500, 2_000, 8_000, 40_000, 250_000} {
		destino := geo.Destination(fortaleza, 30, crowQuerido)
		crow := geo.Haversine(fortaleza, destino)
		fator, vel := cal.For(crow)

		leg, err := e.Distance(ctx, fortaleza, destino)
		if err != nil {
			t.Fatal(err)
		}

		if got, want := leg.Meters, crow*fator; relativo(got, want) > 1e-9 {
			t.Errorf("linha reta %.0f m: distancia %.1f m, esperada %.1f m", crow, got, want)
		}
		if got, want := leg.Seconds, crow*fator/(vel/3.6); relativo(got, want) > 1e-9 {
			t.Errorf("linha reta %.0f m: tempo %.1f s, esperado %.1f s", crow, got, want)
		}
		// A estimativa nunca pode ser menor que a linha reta: nao existe
		// estrada mais curta que a reta entre dois pontos.
		if leg.Meters < crow {
			t.Errorf("estimativa %.1f m menor que a linha reta %.1f m", leg.Meters, crow)
		}
	}
}

func TestDistancePontosCoincidentes(t *testing.T) {
	leg, err := estimadorDeTeste(t).Distance(context.Background(), fortaleza, fortaleza)
	if err != nil {
		t.Fatal(err)
	}
	if leg.Meters != 0 || leg.Seconds != 0 {
		t.Errorf("mesmo ponto = %+v, esperado zero", leg)
	}
}

// Matrix e Distance tem de concordar celula por celula. E a razao pela qual
// Matrix usa Haversine em vez do Ruler, que seria mais rapido: uma matriz que
// discorda da consulta individual gera bug que ninguem acha.
func TestMatrixConcordaComDistance(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	e := estimadorDeTeste(t)
	ctx := context.Background()

	origens := make([]geo.Point, 37)
	destinos := make([]geo.Point, 23)
	for i := range origens {
		origens[i] = geo.Destination(fortaleza, r.Float64()*360, r.Float64()*200_000)
	}
	for j := range destinos {
		destinos[j] = geo.Destination(fortaleza, r.Float64()*360, r.Float64()*200_000)
	}

	m, err := e.Matrix(ctx, origens, destinos)
	if err != nil {
		t.Fatal(err)
	}
	if m.Rows() != 37 || m.Cols() != 23 {
		t.Fatalf("matriz %dx%d, esperada 37x23", m.Rows(), m.Cols())
	}

	for i := range origens {
		for j := range destinos {
			want, err := e.Distance(ctx, origens[i], destinos[j])
			if err != nil {
				t.Fatal(err)
			}
			got := m.At(i, j)
			// float32 na matriz contra float64 na consulta: a tolerancia e
			// a precisao do float32, nao zero.
			if relativo(got.Meters, want.Meters) > 1e-6 {
				t.Fatalf("(%d,%d): matriz %.4f m, Distance %.4f m", i, j, got.Meters, want.Meters)
			}
			if relativo(got.Seconds, want.Seconds) > 1e-6 {
				t.Fatalf("(%d,%d): matriz %.4f s, Distance %.4f s", i, j, got.Seconds, want.Seconds)
			}
		}
	}
}

func TestMatrixVazia(t *testing.T) {
	e := estimadorDeTeste(t)
	ctx := context.Background()

	m, err := e.Matrix(ctx, nil, []geo.Point{fortaleza})
	if err != nil {
		t.Fatal(err)
	}
	if m.Rows() != 0 || m.Cols() != 1 {
		t.Errorf("matriz %dx%d, esperada 0x1", m.Rows(), m.Cols())
	}

	if m, err = e.Matrix(ctx, []geo.Point{fortaleza}, nil); err != nil {
		t.Fatal(err)
	}
	if m.Rows() != 1 || m.Cols() != 0 {
		t.Errorf("matriz %dx%d, esperada 1x0", m.Rows(), m.Cols())
	}
}

func TestMatrixRespeitaCancelamento(t *testing.T) {
	e := estimadorDeTeste(t)
	ctx, cancela := context.WithCancel(context.Background())
	cancela()

	pontos := make([]geo.Point, 500)
	for i := range pontos {
		pontos[i] = geo.Destination(fortaleza, float64(i), 50_000)
	}

	if _, err := e.Matrix(ctx, pontos, pontos); !errors.Is(err, context.Canceled) {
		t.Errorf("erro = %v, esperado context.Canceled", err)
	}
	if _, err := e.Distance(ctx, fortaleza, fortaleza); !errors.Is(err, context.Canceled) {
		t.Errorf("Distance com contexto cancelado: erro = %v", err)
	}
}

func TestMatrixIndiceForaDaFaixaEntraEmPanico(t *testing.T) {
	m := NewMatrix([]geo.Point{fortaleza}, []geo.Point{fortaleza})
	for _, c := range [][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("esperava panico em At(%d,%d)", c[0], c[1])
				}
			}()
			m.At(c[0], c[1])
		}()
	}
}

// Um estimador calibrado com o historico da propria operacao tem de errar
// menos que o andaime generico. Se nao errar, calibrar nao serve para nada.
func TestCalibradoGanhaDoGenerico(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	// Uma operacao com desvio bem maior que o generico -- centro historico,
	// rio no meio, o que for.
	real := []float64{1.95, 1.75, 1.50, 1.30}
	historico := frotaSintetica(r, 3000, real, 0.08)

	cal, rep, err := Calibrate(historico)
	if err != nil {
		t.Fatal(err)
	}
	calibrado, err := NewEstimator(cal)
	if err != nil {
		t.Fatal(err)
	}
	generico := estimadorDeTeste(t)
	ctx := context.Background()

	// Viagens novas, que nao entraram na calibracao.
	novas := frotaSintetica(r, 1000, real, 0.08)

	var errosCal, errosGen []float64
	for _, o := range novas {
		c, err := calibrado.Distance(ctx, o.From, o.To)
		if err != nil {
			t.Fatal(err)
		}
		g, err := generico.Distance(ctx, o.From, o.To)
		if err != nil {
			t.Fatal(err)
		}
		errosCal = append(errosCal, math.Abs(c.Meters-o.Meters)/o.Meters)
		errosGen = append(errosGen, math.Abs(g.Meters-o.Meters)/o.Meters)
	}

	medCal, medGen := mediana(errosCal), mediana(errosGen)
	t.Logf("erro mediano em viagens novas: calibrado %.2f%%, generico %.2f%%", medCal*100, medGen*100)
	t.Logf("%v", rep)

	if medCal >= medGen {
		t.Errorf("calibrar nao ajudou: %.4f contra %.4f", medCal, medGen)
	}
	if medCal > 0.06 {
		t.Errorf("erro do calibrado %.2f%% acima do ruido de 8%% do proprio dado", medCal*100)
	}
}
