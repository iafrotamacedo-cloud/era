package dist

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

var fortaleza = geo.Point{Lat: -3.7319, Lon: -38.5267}

func relativo(a, b float64) float64 {
	if b == 0 {
		return math.Abs(a)
	}
	return math.Abs(a-b) / math.Abs(b)
}

// frotaSintetica fabrica um historico de viagens com um fator de desvio
// conhecido por faixa, para que o teste possa perguntar se Calibrate
// reencontra os numeros que geraram o dado.
//
// E o mesmo movimento do geo: em vez de decorar valores, gera-se a entrada a
// partir da resposta e verifica-se se o caminho de volta fecha.
func frotaSintetica(r *rand.Rand, n int, fatorPorFaixa []float64, ruido float64) []Observation {
	limites := append(append([]float64(nil), DefaultBandLimits...), math.Inf(1))
	obs := make([]Observation, 0, n)

	for i := 0; i < n; i++ {
		// Distancias espalhadas por varias ordens de grandeza, de 500 m a
		// 300 km, para todas as faixas receberem amostra.
		crow := math.Pow(10, 2.7+r.Float64()*2.8)
		destino := geo.Destination(fortaleza, r.Float64()*360, crow)
		crowReal := geo.Haversine(fortaleza, destino)

		fator := fatorPorFaixa[faixaDe(crowReal, limites)]
		fator *= 1 + (r.Float64()*2-1)*ruido

		metros := crowReal * fator
		obs = append(obs, Observation{
			From:    fortaleza,
			To:      destino,
			Meters:  metros,
			Seconds: metros / (40 / 3.6), // 40 km/h em todas, por simplicidade
		})
	}
	return obs
}

// O teste central: se o historico foi gerado com fatores conhecidos por
// faixa, a calibracao tem de reencontra-los.
func TestCalibrateReencontraOsFatores(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	querido := []float64{1.60, 1.40, 1.25, 1.15}

	obs := frotaSintetica(r, 4000, querido, 0.10)
	cal, rep, err := Calibrate(obs)
	if err != nil {
		t.Fatal(err)
	}

	t.Log(rep)
	if len(cal.Bands) != len(querido) {
		t.Fatalf("esperava %d faixas, veio %d", len(querido), len(cal.Bands))
	}

	for i, b := range cal.Bands {
		if b.Samples < MinSamples {
			t.Errorf("faixa %d ficou com %d amostras, abaixo do minimo", i, b.Samples)
			continue
		}
		if e := relativo(b.Factor, querido[i]); e > 0.02 {
			t.Errorf("faixa ate %.0f m: fator %.4f, esperado %.4f (erro %.2f%%)",
				b.UpToMeters, b.Factor, querido[i], e*100)
		}
		if e := relativo(b.SpeedKmh, 40); e > 0.02 {
			t.Errorf("faixa ate %.0f m: velocidade %.2f km/h, esperada 40", b.UpToMeters, b.SpeedKmh)
		}
	}
}

// Um fator unico nao consegue servir a todas as faixas ao mesmo tempo. Este
// teste mede a diferenca em vez de afirma-la -- e a justificativa numerica
// para o pacote ter faixas.
func TestFaixasGanhamDeFatorUnico(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	querido := []float64{1.60, 1.40, 1.25, 1.15}
	obs := frotaSintetica(r, 4000, querido, 0.10)

	comFaixas, repFaixas, err := Calibrate(obs)
	if err != nil {
		t.Fatal(err)
	}

	// Uma faixa unica cobrindo tudo e o equivalente ao fator constante.
	umaFaixa, repUnico, err := Calibrate(obs, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("com faixas: erro mediano %.2f%%, p90 %.2f%%", repFaixas.MedianError*100, repFaixas.P90Error*100)
	t.Logf("fator unico: erro mediano %.2f%%, p90 %.2f%%", repUnico.MedianError*100, repUnico.P90Error*100)

	if repFaixas.MedianError >= repUnico.MedianError {
		t.Errorf("faixas (%.4f) nao ganharam do fator unico (%.4f); a complexidade nao se paga",
			repFaixas.MedianError, repUnico.MedianError)
	}
	if len(umaFaixa.Bands) != 1 || len(comFaixas.Bands) != 4 {
		t.Errorf("contagem de faixas inesperada: %d e %d", len(umaFaixa.Bands), len(comFaixas.Bands))
	}
}

// A mediana existe para nao ser arrastada por dado sujo. Historico real tem
// motorista que passou na oficina e hodometro digitado errado.
func TestMedianaAguentaDadoSujo(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	obs := frotaSintetica(r, 2000, []float64{1.5, 1.5, 1.5, 1.5}, 0.05)

	limpo, _, err := Calibrate(obs, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}

	// 5% das viagens com o hodometro tres vezes maior que o real -- volta
	// pela oficina, desvio pessoal, digitacao errada.
	for i := 0; i < len(obs)/20; i++ {
		obs[r.Intn(len(obs))].Meters *= 3
	}
	sujo, _, err := Calibrate(obs, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("fator limpo %.4f, com 5%% de lixo %.4f", limpo.Factor, sujo.Factor)
	if e := relativo(sujo.Factor, limpo.Factor); e > 0.02 {
		t.Errorf("o lixo moveu o fator em %.2f%%; a mediana deveria ignora-lo", e*100)
	}
}

func TestCalibrateDescartaObservacaoImpossivel(t *testing.T) {
	longe := geo.Destination(fortaleza, 45, 10_000)
	crow := geo.Haversine(fortaleza, longe)

	obs := []Observation{
		{From: fortaleza, To: longe, Meters: crow * 1.3}, // boa
		{From: fortaleza, To: longe, Meters: crow * 0.5}, // menor que a linha reta
		{From: fortaleza, To: longe, Meters: crow * 50},  // erro de digitacao
		{From: fortaleza, To: longe, Meters: 0},          // sem hodometro
		{From: fortaleza, To: longe, Meters: math.NaN()},
		{From: fortaleza, To: fortaleza, Meters: 100},                      // mesmo ponto
		{From: geo.Point{Lat: 200, Lon: 0}, To: longe, Meters: crow * 1.3}, // ponto invalido
	}

	cal, rep, err := Calibrate(obs, math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Used != 1 || rep.Discarded != 6 {
		t.Errorf("usou %d e descartou %d; esperava 1 e 6", rep.Used, rep.Discarded)
	}
	if e := relativo(cal.Factor, 1.3); e > 1e-9 {
		t.Errorf("fator = %.6f, esperado 1,3", cal.Factor)
	}
}

func TestCalibrateSemDadoUtil(t *testing.T) {
	_, _, err := Calibrate([]Observation{{From: fortaleza, To: fortaleza, Meters: 10}})
	if !errors.Is(err, ErrSemObservacoes) {
		t.Errorf("erro = %v, esperado ErrSemObservacoes", err)
	}
	if _, _, err := Calibrate(nil); !errors.Is(err, ErrSemObservacoes) {
		t.Errorf("erro = %v, esperado ErrSemObservacoes", err)
	}
}

func TestCalibrateRejeitaLimitesForaDeOrdem(t *testing.T) {
	if _, _, err := Calibrate(nil, 10_000, 3_000); err == nil {
		t.Error("esperava erro para limites de faixa em ordem decrescente")
	}
}

// Faixa com poucas amostras nao deve devolver um numero especifico sustentado
// por tres viagens: cai para o valor global.
func TestFaixaFracaCaiParaOGlobal(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	var obs []Observation

	// Muitas viagens curtas, pouquissimas longas.
	for i := 0; i < 500; i++ {
		d := geo.Destination(fortaleza, r.Float64()*360, 1_000+r.Float64()*1_000)
		obs = append(obs, Observation{From: fortaleza, To: d, Meters: geo.Haversine(fortaleza, d) * 1.6})
	}
	for i := 0; i < 5; i++ {
		d := geo.Destination(fortaleza, r.Float64()*360, 200_000)
		obs = append(obs, Observation{From: fortaleza, To: d, Meters: geo.Haversine(fortaleza, d) * 1.1})
	}

	cal, _, err := Calibrate(obs)
	if err != nil {
		t.Fatal(err)
	}

	ultima := cal.Bands[len(cal.Bands)-1]
	if ultima.Samples >= MinSamples {
		t.Fatalf("a faixa longa deveria estar fraca, veio com %d amostras", ultima.Samples)
	}

	fator, _ := cal.For(200_000)
	if fator != cal.Factor {
		t.Errorf("faixa fraca devolveu %.4f em vez do global %.4f", fator, cal.Factor)
	}
	if e := relativo(fator, 1.6); e > 0.02 {
		t.Errorf("o global deveria refletir a maioria (1,6), veio %.4f", fator)
	}
}

func TestDefaultCalibrationEstaMarcadaComoGenerica(t *testing.T) {
	d := DefaultCalibration()
	if !d.Generic() {
		t.Error("DefaultCalibration deveria se declarar generica")
	}
	if !d.Valid() {
		t.Error("DefaultCalibration deveria ser utilizavel")
	}

	r := rand.New(rand.NewSource(5))
	medida, _, err := Calibrate(frotaSintetica(r, 500, []float64{1.5, 1.4, 1.3, 1.2}, 0.1))
	if err != nil {
		t.Fatal(err)
	}
	if medida.Generic() {
		t.Error("calibracao medida nao pode se declarar generica")
	}
}

// O fator cai conforme a distancia cresce, no valor generico. E a afirmacao
// que justifica o pacote ter faixas -- se algum dia alguem editar os numeros
// para um valor plano, o teste avisa.
func TestDefaultCalibrationDecaiComADistancia(t *testing.T) {
	d := DefaultCalibration()
	anterior := math.Inf(1)
	for _, b := range d.Bands {
		if b.Factor >= anterior {
			t.Errorf("fator nao decresceu na faixa ate %.0f m: %.2f depois de %.2f",
				b.UpToMeters, b.Factor, anterior)
		}
		anterior = b.Factor
	}
}

func TestPercentil(t *testing.T) {
	if got := percentil(nil, 0.5); got != 0 {
		t.Errorf("percentil de fatia vazia = %v, esperado 0", got)
	}
	if got := percentil([]float64{7}, 0.5); got != 7 {
		t.Errorf("percentil de um elemento = %v, esperado 7", got)
	}
	// Numero par: a mediana e a media dos dois do meio.
	if got := mediana([]float64{1, 2, 3, 4}); got != 2.5 {
		t.Errorf("mediana de 1,2,3,4 = %v, esperado 2,5", got)
	}
	if got := mediana([]float64{5, 1, 3}); got != 3 {
		t.Errorf("mediana de 5,1,3 = %v, esperado 3", got)
	}
	// A entrada nao pode ser reordenada por baixo do chamador.
	v := []float64{3, 1, 2}
	mediana(v)
	if v[0] != 3 || v[1] != 1 || v[2] != 2 {
		t.Errorf("percentil embaralhou a fatia do chamador: %v", v)
	}
	if got := percentil([]float64{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, 0.90); got != 90 {
		t.Errorf("p90 = %v, esperado 90", got)
	}
}
