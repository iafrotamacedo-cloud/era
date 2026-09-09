package ch

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/iafrotamacedo-cloud/era/maps/dist"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// ErrSemCaminho indica que nao ha rota entre os dois pontos.
var ErrSemCaminho = errors.New("ch: nao ha caminho entre os dois pontos")

// Router serve distancias rodoviarias a partir de uma hierarquia.
//
// Implementa dist.Distancer, a interface escrita na Fase 2. Trocar o motor e
// uma linha, e esta e a terceira implementacao a caber nela:
//
//	dist.NewEstimator(cal)          // Fase 2: haversine calibrado, erra ~10%
//	graph.NewRouter(g, graph.Time)  // Fase 4: rota de verdade, milissegundos
//	ch.NewRouter(c)                 // Fase 5: a mesma rota, microssegundos
//
// A metrica vem da hierarquia -- ela foi construida para uma so.
//
// # A matriz nao e um laco de consultas
//
// Distance responde um par. Matrix nao chama Distance N x M vezes: usa o
// algoritmo de baldes, em que cada metade da busca e feita uma vez so. E a
// diferenca entre roteirizar um dia de entregas em segundos ou em minutos, e
// o ganho cresce com o numero de destinos.
//
// # Sobre por o cache na frente
//
// A Fase 2 mediu que o dist.Cache perdia contra o Estimator: uma busca num
// mapa custa mais que a multiplicacao que ela evita. Contra a Fase 4 a conta
// se inverteu. Contra este Router ela volta a ficar apertada -- uma consulta
// na hierarquia ja custa microssegundos --, e vale medir antes de assumir.
type Router struct {
	c *CH

	livres chan *Query
	uma    sync.Once
	n      int
}

// NewRouter embrulha uma hierarquia como fonte de distancias.
func NewRouter(c *CH) *Router {
	return &Router{c: c, n: runtime.GOMAXPROCS(0)}
}

// Metric devolve a grandeza que a hierarquia por tras minimiza.
func (r *Router) Metric() graph.Metric { return r.c.m }

func (r *Router) consultas() chan *Query {
	r.uma.Do(func() {
		r.livres = make(chan *Query, r.n)
		for i := 0; i < r.n; i++ {
			r.livres <- r.c.NewQuery()
		}
	})
	return r.livres
}

// Distance mede um par de coordenadas.
//
// As pontas sao encaixadas no cruzamento mais proximo. O trecho entre a
// coordenada e esse cruzamento nao entra na conta -- para uma entrega e a
// diferenca entre o portao e a esquina.
func (r *Router) Distance(ctx context.Context, from, to geo.Point) (dist.Leg, error) {
	if err := ctx.Err(); err != nil {
		return dist.Leg{}, err
	}

	de, _, ok := r.c.Nearest(from)
	if !ok {
		return dist.Leg{}, fmt.Errorf("%w: nenhum cruzamento perto de %v", ErrSemCaminho, from)
	}
	para, _, ok := r.c.Nearest(to)
	if !ok {
		return dist.Leg{}, fmt.Errorf("%w: nenhum cruzamento perto de %v", ErrSemCaminho, to)
	}

	livres := r.consultas()
	var q *Query
	select {
	case q = <-livres:
	case <-ctx.Done():
		return dist.Leg{}, ctx.Err()
	}
	metros, segundos, achou := q.Leg(de, para)
	livres <- q

	if !achou {
		return dist.Leg{}, fmt.Errorf("%w: de %v ate %v", ErrSemCaminho, from, to)
	}
	return dist.Leg{Meters: metros, Seconds: segundos}, nil
}

// Matrix mede todos os pares de uma vez, pelo algoritmo de baldes.
//
// Par sem caminho fica zerado, como no resto do pacote dist. Um mapa real tem
// desses: ilha sem ponte, condominio mapeado sem ligacao com a rua.
func (r *Router) Matrix(ctx context.Context, origins, destinations []geo.Point) (*dist.Matrix, error) {
	m := dist.NewMatrix(origins, destinations)
	if len(origins) == 0 || len(destinations) == 0 {
		return m, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	nos := func(pontos []geo.Point) ([]graph.NodeID, error) {
		out := make([]graph.NodeID, len(pontos))
		for i, p := range pontos {
			n, _, ok := r.c.Nearest(p)
			if !ok {
				return nil, fmt.Errorf("%w: nenhum cruzamento perto de %v", ErrSemCaminho, p)
			}
			out[i] = n
		}
		return out, nil
	}

	de, err := nos(origins)
	if err != nil {
		return nil, err
	}
	para, err := nos(destinations)
	if err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	bruta := r.c.Matrix(de, para)

	for i := range origins {
		for j := range destinations {
			if !bruta.Alcancavel(i, j) {
				continue
			}
			metros, segundos := bruta.Leg(i, j)
			m.Set(i, j, dist.Leg{Meters: metros, Seconds: segundos})
		}
	}
	return m, nil
}
