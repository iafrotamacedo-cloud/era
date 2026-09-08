package dist

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// contador embrulha um Distancer e conta quantas vezes foi realmente
// procurado. E como o teste distingue "o cache respondeu" de "o cache
// perguntou".
type contador struct {
	inner     Distancer
	distances atomic.Int64
	matrices  atomic.Int64
	erro      error
}

func (c *contador) Distance(ctx context.Context, from, to geo.Point) (Leg, error) {
	c.distances.Add(1)
	if c.erro != nil {
		return Leg{}, c.erro
	}
	return c.inner.Distance(ctx, from, to)
}

func (c *contador) Matrix(ctx context.Context, o, d []geo.Point) (*Matrix, error) {
	c.matrices.Add(1)
	if c.erro != nil {
		return nil, c.erro
	}
	return c.inner.Matrix(ctx, o, d)
}

func cacheDeTeste(t *testing.T) (*Cache, *contador) {
	t.Helper()
	e := estimadorDeTeste(t)
	c := &contador{inner: e}
	return NewCache(c), c
}

func pontos(n int, raioM float64) []geo.Point {
	p := make([]geo.Point, n)
	for i := range p {
		p[i] = geo.Destination(fortaleza, float64(i)*360/float64(n), raioM)
	}
	return p
}

func TestCachePerguntaUmaVezSo(t *testing.T) {
	c, tras := cacheDeTeste(t)
	ctx := context.Background()
	destino := geo.Destination(fortaleza, 45, 20_000)

	primeira, err := c.Distance(ctx, fortaleza, destino)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		got, err := c.Distance(ctx, fortaleza, destino)
		if err != nil {
			t.Fatal(err)
		}
		if got != primeira {
			t.Fatalf("resposta mudou entre chamadas: %+v depois %+v", primeira, got)
		}
	}

	if n := tras.distances.Load(); n != 1 {
		t.Errorf("o motor de tras foi procurado %d vezes, esperava 1", n)
	}
	if c.Hits() != 100 || c.Misses() != 1 {
		t.Errorf("acertos %d e faltas %d, esperava 100 e 1", c.Hits(), c.Misses())
	}
	if c.Len() != 1 {
		t.Errorf("cache com %d entradas, esperava 1", c.Len())
	}
}

// Coordenada em ponto flutuante nunca bate exatamente. Sem arredondar, o
// cache nunca acertaria nada -- e por isso que ele existe com precisao de
// cerca de um metro.
func TestCacheTrataPontosVizinhosComoOMesmoLugar(t *testing.T) {
	c, tras := cacheDeTeste(t)
	ctx := context.Background()
	destino := geo.Destination(fortaleza, 45, 20_000)

	// Deslocamento de um decimo da precisao: mesmo lugar.
	quase := geo.Point{Lat: fortaleza.Lat + CachePrecision/10, Lon: fortaleza.Lon}
	if _, err := c.Distance(ctx, fortaleza, destino); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Distance(ctx, quase, destino); err != nil {
		t.Fatal(err)
	}
	if n := tras.distances.Load(); n != 1 {
		t.Errorf("pontos a 11 cm foram tratados como lugares diferentes (%d consultas)", n)
	}

	// Dez vezes a precisao: lugares diferentes.
	longe := geo.Point{Lat: fortaleza.Lat + CachePrecision*10, Lon: fortaleza.Lon}
	if _, err := c.Distance(ctx, longe, destino); err != nil {
		t.Fatal(err)
	}
	if n := tras.distances.Load(); n != 2 {
		t.Errorf("pontos a 11 m deveriam ser lugares diferentes (%d consultas)", n)
	}
}

// A direcao importa: ir e voltar podem ter distancias diferentes assim que o
// motor de tras conhecer mao unica. O cache nao pode assumir simetria.
func TestCacheNaoAssumeSimetria(t *testing.T) {
	c, tras := cacheDeTeste(t)
	ctx := context.Background()
	destino := geo.Destination(fortaleza, 45, 20_000)

	if _, err := c.Distance(ctx, fortaleza, destino); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Distance(ctx, destino, fortaleza); err != nil {
		t.Fatal(err)
	}
	if n := tras.distances.Load(); n != 2 {
		t.Errorf("ida e volta foram tratadas como o mesmo par (%d consultas)", n)
	}
}

func TestCacheMatrizToraDaCacheNaoPergunta(t *testing.T) {
	c, tras := cacheDeTeste(t)
	ctx := context.Background()
	origens, destinos := pontos(8, 30_000), pontos(6, 70_000)

	primeira, err := c.Matrix(ctx, origens, destinos)
	if err != nil {
		t.Fatal(err)
	}
	// Tudo faltando: uma chamada de matriz ao motor de tras.
	if n := tras.matrices.Load(); n != 1 {
		t.Errorf("esperava 1 chamada de matriz, veio %d", n)
	}

	tras.matrices.Store(0)
	tras.distances.Store(0)

	segunda, err := c.Matrix(ctx, origens, destinos)
	if err != nil {
		t.Fatal(err)
	}
	if n := tras.matrices.Load() + tras.distances.Load(); n != 0 {
		t.Errorf("a segunda matriz procurou o motor de tras %d vezes", n)
	}

	for i := range origens {
		for j := range destinos {
			if primeira.At(i, j) != segunda.At(i, j) {
				t.Fatalf("(%d,%d) mudou entre as duas matrizes", i, j)
			}
		}
	}
}

// Falta pouca: vale mais perguntar par a par do que remedir a matriz inteira.
func TestCacheMatrizComPoucasFaltasPerguntaParAPar(t *testing.T) {
	c, tras := cacheDeTeste(t)
	ctx := context.Background()
	origens, destinos := pontos(10, 30_000), pontos(10, 70_000)

	if _, err := c.Matrix(ctx, origens, destinos); err != nil {
		t.Fatal(err)
	}
	tras.matrices.Store(0)
	tras.distances.Store(0)

	// Uma origem nova: 10 celulas faltando em 110, bem abaixo do limiar.
	origens = append(origens, geo.Destination(fortaleza, 123, 45_000))
	if _, err := c.Matrix(ctx, origens, destinos); err != nil {
		t.Fatal(err)
	}

	if n := tras.matrices.Load(); n != 0 {
		t.Errorf("delegou a matriz inteira (%d chamadas) com poucas faltas", n)
	}
	if n := tras.distances.Load(); n != int64(len(destinos)) {
		t.Errorf("perguntou %d pares, esperava %d", n, len(destinos))
	}
}

func TestCachePropagaErroDoMotorDeTras(t *testing.T) {
	c, tras := cacheDeTeste(t)
	ctx := context.Background()
	querido := errors.New("sem caminho")
	tras.erro = querido

	if _, err := c.Distance(ctx, fortaleza, geo.Destination(fortaleza, 0, 1000)); !errors.Is(err, querido) {
		t.Errorf("Distance: erro = %v, esperado %v", err, querido)
	}
	if _, err := c.Matrix(ctx, pontos(4, 10_000), pontos(4, 20_000)); !errors.Is(err, querido) {
		t.Errorf("Matrix: erro = %v, esperado %v", err, querido)
	}
	// Erro nao pode virar entrada guardada.
	if c.Len() != 0 {
		t.Errorf("o cache guardou %d entradas apesar do erro", c.Len())
	}
}

// O cache declara ser seguro para uso concorrente. O -race do CI e quem
// verifica; este teste e o que da a ele o que verificar.
func TestCacheConcorrente(t *testing.T) {
	c, _ := cacheDeTeste(t)
	ctx := context.Background()
	ps := pontos(50, 40_000)

	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range ps {
				for j := range ps {
					if _, err := c.Distance(ctx, ps[i], ps[j]); err != nil {
						t.Error(err)
						return
					}
				}
			}
			_, _ = c.Matrix(ctx, ps[:10], ps[:10])
			_ = c.Len()
			_ = c.Hits()
		}(w)
	}
	wg.Wait()

	if c.Len() != len(ps)*len(ps) {
		t.Errorf("cache com %d entradas, esperava %d", c.Len(), len(ps)*len(ps))
	}
	if c.Hits()+c.Misses() < int64(len(ps)*len(ps)) {
		t.Errorf("contabilidade perdeu eventos: %d acertos + %d faltas", c.Hits(), c.Misses())
	}
}
