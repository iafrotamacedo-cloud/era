package dist

import (
	"bytes"
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

func TestCacheIdaEVoltaEmDisco(t *testing.T) {
	c, tras := cacheDeTeste(t)
	ctx := context.Background()
	origens, destinos := pontos(20, 30_000), pontos(15, 80_000)

	if _, err := c.Matrix(ctx, origens, destinos); err != nil {
		t.Fatal(err)
	}
	antes := c.Len()

	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}

	// O tamanho do arquivo e previsivel: cabecalho, contagem e entradas de
	// tamanho fixo. Se deixar de ser, o formato mudou sem ninguem avisar.
	if got, want := buf.Len(), 8+4+antes*tamanhoEntrada; got != want {
		t.Errorf("arquivo com %d bytes, esperado %d", got, want)
	}

	novoTras := &contador{inner: estimadorDeTeste(t)}
	lido, err := LoadCache(&buf, novoTras)
	if err != nil {
		t.Fatal(err)
	}
	if lido.Len() != antes {
		t.Fatalf("carregou %d entradas, gravou %d", lido.Len(), antes)
	}

	for i := range origens {
		for j := range destinos {
			querido, err := c.Distance(ctx, origens[i], destinos[j])
			if err != nil {
				t.Fatal(err)
			}
			got, err := lido.Distance(ctx, origens[i], destinos[j])
			if err != nil {
				t.Fatal(err)
			}
			if got != querido {
				t.Fatalf("(%d,%d): apos ida e volta %+v, era %+v", i, j, got, querido)
			}
		}
	}

	// Nada disso pode ter procurado o motor de tras: veio tudo do arquivo.
	if n := novoTras.distances.Load() + novoTras.matrices.Load(); n != 0 {
		t.Errorf("o cache carregado procurou o motor de tras %d vezes", n)
	}
	_ = tras
}

func TestCacheVazioIdaEVolta(t *testing.T) {
	c, _ := cacheDeTeste(t)
	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lido, err := LoadCache(&buf, nil)
	if err != nil {
		t.Fatal(err)
	}
	if lido.Len() != 0 {
		t.Errorf("cache vazio virou %d entradas", lido.Len())
	}
}

func TestCalibracaoIdaEVolta(t *testing.T) {
	r := rand.New(rand.NewSource(20))
	original, _, err := Calibrate(frotaSintetica(r, 2000, []float64{1.7, 1.5, 1.3, 1.2}, 0.1))
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := original.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lida, err := LoadCalibration(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if len(lida.Bands) != len(original.Bands) {
		t.Fatalf("carregou %d faixas, gravou %d", len(lida.Bands), len(original.Bands))
	}
	for i, b := range original.Bands {
		if lida.Bands[i] != b {
			t.Errorf("faixa %d: %+v, era %+v", i, lida.Bands[i], b)
		}
	}
	if lida.Factor != original.Factor || lida.SpeedKmh != original.SpeedKmh {
		t.Errorf("globais %v/%v, eram %v/%v",
			lida.Factor, lida.SpeedKmh, original.Factor, original.SpeedKmh)
	}
	if lida.Samples != original.Samples {
		t.Errorf("amostras %d, eram %d", lida.Samples, original.Samples)
	}
	if lida.Generic() {
		t.Error("calibracao medida voltou marcada como generica")
	}

	// A ultima faixa vai ate o infinito. Se o formato nao preservasse isso,
	// distancias longas cairiam fora de toda faixa.
	if !math.IsInf(lida.Bands[len(lida.Bands)-1].UpToMeters, 1) {
		t.Error("o infinito da ultima faixa nao sobreviveu ao arquivo")
	}
}

// A marca de "generica" viaja junto: um sistema que carrega a calibracao
// precisa poder avisar que ainda esta usando o andaime.
func TestGenericaSobreviveAoArquivo(t *testing.T) {
	var buf bytes.Buffer
	if err := DefaultCalibration().Save(&buf); err != nil {
		t.Fatal(err)
	}
	lida, err := LoadCalibration(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !lida.Generic() {
		t.Error("a calibracao generica voltou como se fosse medida")
	}

	// E continua se comportando igual depois da viagem.
	original := DefaultCalibration()
	for _, d := range []float64{500, 5_000, 30_000, 500_000} {
		f1, v1 := original.For(d)
		f2, v2 := lida.For(d)
		if f1 != f2 || v1 != v2 {
			t.Errorf("a %.0f m: %v/%v depois do arquivo, era %v/%v", d, f2, v2, f1, v1)
		}
	}
}

func TestFormatoRejeitaArquivoErrado(t *testing.T) {
	// Uma calibracao lida como cache, e vice-versa.
	var cal bytes.Buffer
	if err := DefaultCalibration().Save(&cal); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCache(bytes.NewReader(cal.Bytes()), nil); !errors.Is(err, ErrFormato) {
		t.Errorf("cache lendo calibracao: erro = %v, esperado ErrFormato", err)
	}

	c, _ := cacheDeTeste(t)
	var cache bytes.Buffer
	if err := c.Save(&cache); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCalibration(bytes.NewReader(cache.Bytes())); !errors.Is(err, ErrFormato) {
		t.Errorf("calibracao lendo cache: erro = %v, esperado ErrFormato", err)
	}

	for _, lixo := range [][]byte{nil, []byte("oi"), []byte("nao sou um arquivo da era")} {
		if _, err := LoadCache(bytes.NewReader(lixo), nil); !errors.Is(err, ErrFormato) {
			t.Errorf("lixo %q: erro = %v, esperado ErrFormato", lixo, err)
		}
	}
}

func TestFormatoRejeitaVersaoDesconhecida(t *testing.T) {
	var buf bytes.Buffer
	if err := DefaultCalibration().Save(&buf); err != nil {
		t.Fatal(err)
	}
	bs := buf.Bytes()
	bs[7] = versaoAtual + 1 // o byte de versao

	if _, err := LoadCalibration(bytes.NewReader(bs)); !errors.Is(err, ErrVersao) {
		t.Errorf("erro = %v, esperado ErrVersao", err)
	}
}

// Arquivo truncado -- disco cheio no meio da gravacao, cópia interrompida --
// tem de virar erro, nao cache pela metade fingindo estar completo.
func TestFormatoRejeitaArquivoTruncado(t *testing.T) {
	c, _ := cacheDeTeste(t)
	ctx := context.Background()
	if _, err := c.Matrix(ctx, pontos(10, 20_000), pontos(10, 50_000)); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	inteiro := buf.Bytes()

	for _, corte := range []int{9, 20, len(inteiro) / 2, len(inteiro) - 1} {
		if _, err := LoadCache(bytes.NewReader(inteiro[:corte]), nil); !errors.Is(err, ErrFormato) {
			t.Errorf("cortado em %d bytes: erro = %v, esperado ErrFormato", corte, err)
		}
	}

	var cal bytes.Buffer
	if err := DefaultCalibration().Save(&cal); err != nil {
		t.Fatal(err)
	}
	calBytes := cal.Bytes()
	for _, corte := range []int{9, 20, len(calBytes) - 1} {
		if _, err := LoadCalibration(bytes.NewReader(calBytes[:corte])); !errors.Is(err, ErrFormato) {
			t.Errorf("calibracao cortada em %d bytes: erro = %v, esperado ErrFormato", corte, err)
		}
	}
}

// O cache guarda em float32 para caber; isso custa precisao, e o custo tem de
// ser pequeno o bastante para nao importar.
func TestPrecisaoDoFloat32NoArquivo(t *testing.T) {
	c, _ := cacheDeTeste(t)
	ctx := context.Background()

	// Muitos destinos, ate 1.000 km. Um ponto so nao serve: uma distancia
	// pode calhar de ser exatamente representavel em float32, e o teste
	// passaria sem nunca exercitar o arredondamento.
	r := rand.New(rand.NewSource(21))
	destinos := make([]geo.Point, 2000)
	antes := make([]Leg, len(destinos))
	for i := range destinos {
		destinos[i] = geo.Destination(fortaleza, r.Float64()*360, 100+r.Float64()*1_000_000)
		leg, err := c.Distance(ctx, fortaleza, destinos[i])
		if err != nil {
			t.Fatal(err)
		}
		antes[i] = leg
	}

	var buf bytes.Buffer
	if err := c.Save(&buf); err != nil {
		t.Fatal(err)
	}
	lido, err := LoadCache(&buf, nil)
	if err != nil {
		t.Fatal(err)
	}

	var pior, piorEm float64
	for i, d := range destinos {
		depois, err := lido.Distance(ctx, fortaleza, d)
		if err != nil {
			t.Fatal(err)
		}
		if perda := math.Abs(depois.Meters - antes[i].Meters); perda > pior {
			pior, piorEm = perda, antes[i].Meters
		}
	}

	t.Logf("pior perda do float32 em %d distancias: %.4f m (a %.0f km)", len(destinos), pior, piorEm/1000)
	if pior == 0 {
		t.Error("nenhuma perda em 2000 distancias: o teste nao esta exercitando o arredondamento")
	}
	if pior > 0.5 {
		t.Errorf("perda de %.4f m acima do aceitavel", pior)
	}
}
