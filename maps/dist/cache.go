package dist

import (
	"context"
	"math"
	"runtime"
	"sync"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// CachePrecision e o tamanho da celula, em graus, com que o cache trata dois
// pontos como o mesmo lugar. Cerca de 1,1 metro.
//
// Existe porque coordenada em ponto flutuante nunca bate exatamente: o mesmo
// endereco geocodificado duas vezes, ou lido de dois cadastros, difere no
// decimo digito. Sem arredondar, o cache nunca acertaria nada.
//
// Um metro e a escolha certa para logistica: a coordenada de um cliente nao
// tem, nem precisa ter, precisao melhor que isso. Quem entrega chega na
// portaria, nao numa coordenada.
const CachePrecision = 1e-5

// Cache guarda o que ja foi medido e so pergunta ao Distancer de tras o que
// ainda nao sabe.
//
// A ideia e obvia: a distancia entre dois lugares fixos nao muda, e uma
// operacao pergunta pelos mesmos pares todo dia. O que nao e obvio -- e os
// benchmarks deste pacote mostram -- e que guardar so compensa quando medir
// custa caro.
//
// Contra o Estimator, medido num i7-9750H:
//
//	consulta individual   ~100 ns cacheada contra ~130 ns calculada
//	matriz 500x500        ~17 ms cacheada contra ~10 ms calculada
//
// A matriz cacheada perde. O motivo e estrutural, nao um defeito a corrigir:
// meio milhao de buscas num mapa de 12 MB sao meio milhao de idas a memoria
// principal, e cada uma custa mais do que a multiplicacao que o Estimator
// faria no lugar. Nao ha cache que ganhe de uma conta de dezenas de
// nanossegundos.
//
// Entao: com o Estimator, use o cache para consultas avulsas e deixe as
// matrizes passarem direto. O cache foi escrito para a Fase 5, quando atras
// dele estiver uma busca no grafo rodoviario -- ai os 200 ns da busca no
// mapa somem diante do custo do que se evitou, e a conta inverte.
//
// Cache implementa Distancer, entao entra e sai da pilha sem que o sistema
// chamador saiba:
//
//	d := dist.NewCache(estimator)
//
// E seguro para uso concorrente.
type Cache struct {
	inner Distancer

	mu      sync.RWMutex
	entries map[chave]Leg
	hits    int64
	misses  int64
}

// chave e o par de pontos arredondado para a grade de CachePrecision.
//
// Quatro int32 em vez de quatro float64: cabe em 16 bytes, e comparavel por
// igualdade exata e serve de chave de mapa sem alocar.
type chave struct {
	deLat, deLon int32
	aLat, aLon   int32
}

func quantiza(p geo.Point) (int32, int32) {
	return int32(math.Round(p.Lat / CachePrecision)),
		int32(math.Round(geo.NormalizeLon(p.Lon) / CachePrecision))
}

func chaveDe(from, to geo.Point) chave {
	deLat, deLon := quantiza(from)
	aLat, aLon := quantiza(to)
	return chave{deLat: deLat, deLon: deLon, aLat: aLat, aLon: aLon}
}

// NewCache embrulha um Distancer.
func NewCache(inner Distancer) *Cache {
	return &Cache{inner: inner, entries: make(map[chave]Leg)}
}

// Len devolve quantos pares estao guardados.
func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// Hits e Misses contam acertos e faltas desde que o cache foi criado ou
// carregado. Servem para responder se ele esta valendo a pena.
func (c *Cache) Hits() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.hits
}

func (c *Cache) Misses() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.misses
}

func (c *Cache) busca(k chave) (Leg, bool) {
	c.mu.RLock()
	l, ok := c.entries[k]
	c.mu.RUnlock()
	return l, ok
}

func (c *Cache) guarda(k chave, l Leg, acertou bool) {
	c.mu.Lock()
	if !acertou {
		c.entries[k] = l
		c.misses++
	} else {
		c.hits++
	}
	c.mu.Unlock()
}

// Distance responde do que ja sabe, ou pergunta e guarda.
func (c *Cache) Distance(ctx context.Context, from, to geo.Point) (Leg, error) {
	k := chaveDe(from, to)
	if l, ok := c.busca(k); ok {
		c.guarda(k, l, true)
		return l, nil
	}

	l, err := c.inner.Distance(ctx, from, to)
	if err != nil {
		return Leg{}, err
	}
	c.guarda(k, l, false)
	return l, nil
}

// limiarDelegacao e a fracao de celulas faltando a partir da qual vale mais
// pedir a matriz inteira de uma vez do que par a par.
//
// Uma matriz cheia de faltas paga por remedir o pouco que ja sabia, mas ganha
// o processamento em lote de tras -- que no Estimator e o paralelismo por
// linha, e num roteador de verdade sera a busca muitos-para-muitos, onde uma
// chamada custa muito menos que N chamadas soltas.
const limiarDelegacao = 0.5

// Matrix responde a matriz combinando o que ja sabe com o que precisa
// perguntar.
func (c *Cache) Matrix(ctx context.Context, origins, destinations []geo.Point) (*Matrix, error) {
	m := NewMatrix(origins, destinations)
	total := len(origins) * len(destinations)
	if total == 0 {
		return m, nil
	}

	// Duas economias que so aparecem em escala de matriz, e que custaram uma
	// regressao de 8x para serem descobertas:
	//
	// Quantizar uma vez por ponto, nao uma por celula. Numa matriz 500x500
	// sao mil arredondamentos em vez de meio milhao.
	//
	// E trancar o mutex uma vez para a varredura inteira, nao por celula.
	// Bloquear e desbloquear 250 mil vezes custava mais do que simplesmente
	// recalcular tudo -- o cache ficava mais lento que nao ter cache.
	qo := make([][2]int32, len(origins))
	for i, p := range origins {
		lat, lon := quantiza(p)
		qo[i] = [2]int32{lat, lon}
	}
	qd := make([][2]int32, len(destinations))
	for j, p := range destinations {
		lat, lon := quantiza(p)
		qd[j] = [2]int32{lat, lon}
	}
	chaveEm := func(i, j int) chave {
		return chave{deLat: qo[i][0], deLon: qo[i][1], aLat: qd[j][0], aLon: qd[j][1]}
	}

	type falta struct{ i, j int }

	// A varredura e paralela por linha, como a do Estimator. Cada goroutine
	// so le do mapa e so escreve nas celulas da sua linha.
	//
	// Nao e microotimizacao: uma varredura serial contra um Estimator
	// paralelo compara coisas diferentes, e faz o cache parecer perder por
	// um fator que e so a contagem de nucleos.
	trabalhadores := runtime.GOMAXPROCS(0)
	if trabalhadores > len(origins) {
		trabalhadores = len(origins)
	}

	parciais := make([][]falta, trabalhadores)
	acertosPorW := make([]int64, trabalhadores)

	c.mu.RLock()
	var wg sync.WaitGroup
	for w := 0; w < trabalhadores; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := w; i < len(origins); i += trabalhadores {
				for j := range destinations {
					if l, ok := c.entries[chaveEm(i, j)]; ok {
						m.Set(i, j, l)
						acertosPorW[w]++
						continue
					}
					parciais[w] = append(parciais[w], falta{i, j})
				}
			}
		}(w)
	}
	wg.Wait()
	c.mu.RUnlock()

	var faltas []falta
	var acertos int64
	for w := 0; w < trabalhadores; w++ {
		faltas = append(faltas, parciais[w]...)
		acertos += acertosPorW[w]
	}

	if len(faltas) == 0 {
		c.mu.Lock()
		c.hits += acertos
		c.mu.Unlock()
		return m, nil
	}

	if float64(len(faltas))/float64(total) >= limiarDelegacao {
		// Falta demais: uma chamada so, e guarda tudo.
		completa, err := c.inner.Matrix(ctx, origins, destinations)
		if err != nil {
			return nil, err
		}
		c.mu.Lock()
		c.hits += acertos
		for i := range origins {
			for j := range destinations {
				k := chaveEm(i, j)
				if _, ja := c.entries[k]; !ja {
					c.entries[k] = completa.At(i, j)
					c.misses++
				}
			}
		}
		c.mu.Unlock()
		return completa, nil
	}

	// Poucas faltas: pergunta par a par, e grava tudo de uma vez no fim em
	// vez de trancar a cada resposta.
	medidas := make([]Leg, len(faltas))
	for n, f := range faltas {
		l, err := c.inner.Distance(ctx, origins[f.i], destinations[f.j])
		if err != nil {
			return nil, err
		}
		m.Set(f.i, f.j, l)
		medidas[n] = l
	}

	c.mu.Lock()
	c.hits += acertos
	for n, f := range faltas {
		c.entries[chaveEm(f.i, f.j)] = medidas[n]
		c.misses++
	}
	c.mu.Unlock()
	return m, nil
}
