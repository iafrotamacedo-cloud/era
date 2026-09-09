package graph

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/iafrotamacedo-cloud/era/maps/dist"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// ErrSemCaminho indica que nao ha rota entre os dois pontos.
//
// Acontece de verdade: uma ilha sem ponte, um condominio fechado mapeado sem
// ligacao com a rua, um pedaco de malha que ficou solto no recorte do
// extrato.
var ErrSemCaminho = errors.New("graph: nao ha caminho entre os dois pontos")

// Router mede distancia rodoviaria de verdade, percorrendo o grafo.
//
// Implementa dist.Distancer, a interface que a Fase 2 escreveu para este
// momento. Quem estava usando o dist.Estimator troca a construcao e nao muda
// mais nada:
//
//	d := dist.NewCache(graph.NewRouter(g, graph.Time))
//
// # O que muda em relacao ao Estimator
//
// A resposta deixa de ser uma estimativa com uns 10% de erro e passa a ser o
// caminho que existe. Em troca, uma consulta deixa de custar dezenas de
// nanossegundos e passa a custar centenas de microssegundos -- quatro ordens
// de grandeza.
//
// Isso inverte a conta do cache. Com o Estimator, guardar custava mais que
// recalcular, e o README do maps traz a tabela dizendo isso. Com um Router
// atras, os 200 ns da busca no mapa somem diante do que se evitou. Ponha o
// cache na frente.
//
// # O que ainda falta
//
// Uma matriz 500 x 500 sao 250 mil buscas. Mesmo com o Dijkstra bidirecional
// isso leva minutos, e e por isso que a Fase 5 existe: as Contraction
// Hierarchies transformam essa mesma matriz numa conta de segundos. Ate la,
// Matrix aqui serve para conferir a resposta, nao para rodar em producao.
type Router struct {
	g *Graph
	m Metric

	// Um buscador por goroutine. Cada um carrega vetores do tamanho do grafo,
	// entao eles circulam num conjunto em vez de serem criados por consulta.
	livres chan *Searcher
	uma    sync.Once
	n      int
}

// NewRouter embrulha um grafo como fonte de distancias.
func NewRouter(g *Graph, m Metric) *Router {
	return &Router{g: g, m: m, n: runtime.GOMAXPROCS(0)}
}

func (r *Router) buscadores() chan *Searcher {
	r.uma.Do(func() {
		r.livres = make(chan *Searcher, r.n)
		for i := 0; i < r.n; i++ {
			r.livres <- r.g.NewSearcher()
		}
	})
	return r.livres
}

// Distance mede um par de coordenadas.
//
// As pontas sao encaixadas no cruzamento mais proximo, porque o grafo so sabe
// falar de cruzamentos. O trecho a pe entre a coordenada e esse cruzamento
// nao entra na conta -- para uma entrega isso e a diferenca entre o portao e
// a esquina, alguns metros.
func (r *Router) Distance(ctx context.Context, from, to geo.Point) (dist.Leg, error) {
	if err := ctx.Err(); err != nil {
		return dist.Leg{}, err
	}

	de, _, ok := r.g.Nearest(from)
	if !ok {
		return dist.Leg{}, fmt.Errorf("%w: nenhum cruzamento perto de %v", ErrSemCaminho, from)
	}
	para, _, ok := r.g.Nearest(to)
	if !ok {
		return dist.Leg{}, fmt.Errorf("%w: nenhum cruzamento perto de %v", ErrSemCaminho, to)
	}

	livres := r.buscadores()
	var s *Searcher
	select {
	case s = <-livres:
	case <-ctx.Done():
		return dist.Leg{}, ctx.Err()
	}
	p, achou := s.Route(de, para, r.m)
	livres <- s

	if !achou {
		return dist.Leg{}, fmt.Errorf("%w: de %v ate %v", ErrSemCaminho, from, to)
	}
	return dist.Leg{Meters: p.Meters, Seconds: p.Seconds}, nil
}

// Matrix mede todos os pares, uma linha por goroutine.
//
// E correto e lento: cada celula e uma busca no grafo. Veja a nota sobre a
// Fase 5 na documentacao do Router.
func (r *Router) Matrix(ctx context.Context, origins, destinations []geo.Point) (*dist.Matrix, error) {
	m := dist.NewMatrix(origins, destinations)
	if len(origins) == 0 || len(destinations) == 0 {
		return m, nil
	}

	// Os destinos sao encaixados uma vez so, e nao uma vez por celula: numa
	// matriz 500 x 500 isso e a diferenca entre 500 e 250 mil consultas ao
	// indice espacial.
	alvos := make([]NodeID, len(destinations))
	for j, d := range destinations {
		n, _, ok := r.g.Nearest(d)
		if !ok {
			return nil, fmt.Errorf("%w: nenhum cruzamento perto de %v", ErrSemCaminho, d)
		}
		alvos[j] = n
	}

	livres := r.buscadores()
	var wg sync.WaitGroup
	linhas := make(chan int)
	erros := make(chan error, 1)
	parar := make(chan struct{})
	var umaVez sync.Once
	falhar := func(err error) {
		umaVez.Do(func() {
			select {
			case erros <- err:
			default:
			}
			close(parar)
		})
	}

	for w := 0; w < r.n; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := <-livres
			defer func() { livres <- s }()

			for i := range linhas {
				de, _, ok := r.g.Nearest(origins[i])
				if !ok {
					falhar(fmt.Errorf("%w: nenhum cruzamento perto de %v", ErrSemCaminho, origins[i]))
					return
				}
				for j, alvo := range alvos {
					if ctx.Err() != nil {
						falhar(ctx.Err())
						return
					}
					p, achou := s.Route(de, alvo, r.m)
					if !achou {
						continue // fica zerada: nao ha caminho entre esse par
					}
					m.Set(i, j, dist.Leg{Meters: p.Meters, Seconds: p.Seconds})
				}
			}
		}()
	}

	for i := range origins {
		select {
		case linhas <- i:
		case <-parar:
			close(linhas)
			wg.Wait()
			return nil, <-erros
		}
	}
	close(linhas)
	wg.Wait()

	select {
	case err := <-erros:
		return nil, err
	default:
		return m, nil
	}
}
