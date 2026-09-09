// Comando osrmcompare confere o motor de rotas da ERA contra o OSRM.
//
// # Por que existe
//
// Os testes da ERA verificam que as tres implementacoes de rota concordam
// entre si: a hierarquia bate com o Dijkstra bidirecional, que bate com o
// Dijkstra obvio. E uma cadeia solida, e ela nao prova o que mais importa.
//
// Se eu tiver interpretado errado uma etiqueta do OpenStreetMap -- mao unica
// implicita, restricao de acesso, o que conta como estrada --, as tres
// implementacoes erram juntas e em silencio, porque todas leem o mapa pelo
// mesmo perfil. So uma referencia de fora pega esse tipo de erro, e o OSRM e
// a referencia de fora que existe.
//
// # Como usar
//
// Baixe um extrato e suba um OSRM sobre ele:
//
//	curl -O https://download.geofabrik.de/south-america/brazil/nordeste/ceara-latest.osm.pbf
//
//	docker run -t -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-extract -p /opt/car.lua /data/ceara-latest.osm.pbf
//	docker run -t -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-partition /data/ceara-latest.osrm
//	docker run -t -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-customize /data/ceara-latest.osrm
//	docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-routed --algorithm mld /data/ceara-latest.osrm
//
// E compare, sobre o mesmo extrato:
//
//	go run ./maps/cmd/osrmcompare -mapa ceara-latest.osm.pbf -n 500
//
// # O que esperar
//
// Nao vai bater em um por cento de primeira, e o valor da primeira rodada e
// mostrar onde erra, nao dizer que esta certo. Erro parelho em todos os pares
// e perfil de velocidade; erro concentrado em rotas urbanas curtas e
// penalidade de conversao, que o OSRM aplica e a ERA ainda nao; erro
// esporadico e grande e etiqueta lida errado, que e o caso que este programa
// existe para achar.
//
// # Sobre a rede
//
// Este e o unico lugar da ERA que fala HTTP, e ele nao e parte do motor:
// nenhum pacote de maps importa net/http. A promessa de rodar dentro do
// processo de quem chama continua valendo.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/iafrotamacedo-cloud/era/maps/ch"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

func main() {
	mapa := flag.String("mapa", "", "arquivo .osm.pbf ou .eramap (obrigatorio)")
	base := flag.String("osrm", "http://localhost:5000", "endereco do OSRM")
	perfil := flag.String("perfil", "driving", "perfil do OSRM")
	n := flag.Int("n", 200, "quantos pares comparar")
	semente := flag.Int64("semente", 1, "semente do sorteio, para repetir a mesma amostra")
	minKm := flag.Float64("min-km", 1, "distancia minima em linha reta entre as pontas")
	maxKm := flag.Float64("max-km", 100, "distancia maxima em linha reta entre as pontas")
	encaixeMax := flag.Float64("encaixe-max", 30, "descarta o par se o OSRM encaixar a mais de tantos metros")
	paralelo := flag.Int("paralelo", 4, "consultas simultaneas ao OSRM")
	espera := flag.Duration("espera", 30*time.Second, "tempo limite de cada consulta")
	salvar := flag.String("salvar-eramap", "", "grava a hierarquia preparada neste caminho")
	flag.Parse()

	if *mapa == "" {
		fmt.Fprintln(os.Stderr, "falta -mapa; use -h para ver as opcoes")
		os.Exit(2)
	}

	if err := rodar(opcoes{
		mapa: *mapa, base: *base, perfil: *perfil,
		n: *n, semente: *semente,
		minM: *minKm * 1000, maxM: *maxKm * 1000,
		encaixeMax: *encaixeMax,
		paralelo:   *paralelo, espera: *espera,
		salvar: *salvar,
	}); err != nil {
		fmt.Fprintln(os.Stderr, "erro:", err)
		os.Exit(1)
	}
}

type opcoes struct {
	mapa, base, perfil string
	n                  int
	semente            int64
	minM, maxM         float64
	encaixeMax         float64
	paralelo           int
	espera             time.Duration
	salvar             string
}

func rodar(o opcoes) error {
	c, err := carregar(o.mapa, o.salvar)
	if err != nil {
		return err
	}
	fmt.Println(c.Stats())

	pares := sortearPares(c, o.n, o.semente, o.minM, o.maxM)
	if len(pares) == 0 {
		return fmt.Errorf("nao consegui sortear nenhum par entre %.0f e %.0f km; ajuste -min-km e -max-km",
			o.minM/1000, o.maxM/1000)
	}
	fmt.Printf("comparando %d pares contra %s\n\n", len(pares), o.base)

	rel, err := comparar(c, pares, o)
	if err != nil {
		return err
	}
	fmt.Print(rel)
	return nil
}

// carregar aceita os dois caminhos possiveis: o extrato cru ou a hierarquia
// ja preparada.
func carregar(caminho, salvar string) (*ch.CH, error) {
	if strings.EqualFold(filepath.Ext(caminho), ".eramap") {
		fmt.Printf("carregando %s\n", caminho)
		return ch.LoadFile(caminho)
	}

	fmt.Printf("montando o grafo de %s\n", caminho)
	inicio := time.Now()
	g, err := graph.BuildFile(caminho, graph.Car())
	if err != nil {
		return nil, err
	}
	fmt.Printf("%s, em %s\n", g.Stats(), time.Since(inicio).Round(time.Millisecond))

	fmt.Println("preparando a hierarquia (isto demora)")
	c, err := ch.Prepare(g, graph.Distance)
	if err != nil {
		return nil, err
	}

	if salvar != "" {
		if err := c.SaveFile(salvar); err != nil {
			return nil, fmt.Errorf("gravando o eramap: %w", err)
		}
		fmt.Printf("hierarquia gravada em %s\n", salvar)
	}
	return c, nil
}

type par struct{ de, para graph.NodeID }

// sortearPares escolhe pares de cruzamentos do proprio grafo.
//
// Sortear coordenadas soltas dentro do retangulo do extrato seria pior: cada
// motor encaixaria a sua maneira, e boa parte do erro medido seria a
// diferenca entre os dois encaixes, nao entre as duas rotas. Usando
// coordenadas que ja sao de cruzamentos, o OSRM encaixa praticamente no mesmo
// lugar -- e o quanto ele se afastou vem na resposta, para os pares em que
// isso nao valeu serem descartados.
func sortearPares(c *ch.CH, quantos int, semente int64, minM, maxM float64) []par {
	r := rand.New(rand.NewSource(semente))
	pares := make([]par, 0, quantos)

	// Um teto de tentativas: num extrato recortado pode simplesmente nao
	// haver pares na faixa pedida, e o programa precisa dizer isso em vez de
	// girar para sempre.
	for tentativas := 0; len(pares) < quantos && tentativas < quantos*200; tentativas++ {
		de := graph.NodeID(r.Intn(c.Len()))
		para := graph.NodeID(r.Intn(c.Len()))
		if de == para {
			continue
		}
		reta := geo.Haversine(c.Point(de), c.Point(para))
		if reta < minM || reta > maxM {
			continue
		}
		pares = append(pares, par{de, para})
	}
	return pares
}

func comparar(c *ch.CH, pares []par, o opcoes) (*Relatorio, error) {
	cliente := &Cliente{
		Base:   o.base,
		Perfil: o.perfil,
		HTTP:   &http.Client{Timeout: o.espera},
	}

	// Uma consulta antes de tudo, em serie, para falhar cedo e com mensagem
	// clara se o OSRM nao estiver de pe. Descobrir isso depois de 200
	// goroutines terem falhado daria uma pilha de erros iguais.
	ctx := context.Background()
	if _, err := cliente.Rota(ctx, c.Point(pares[0].de), c.Point(pares[0].para)); err != nil {
		var semRota ErrSemRota
		if !errors.As(err, &semRota) {
			return nil, fmt.Errorf("o osrm em %s nao respondeu: %w", o.base, err)
		}
	}

	rel := &Relatorio{Tentados: len(pares)}

	trabalho := make(chan par)
	var mu sync.Mutex
	var wg sync.WaitGroup

	if o.paralelo < 1 {
		o.paralelo = 1
	}
	for w := 0; w < o.paralelo; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := c.NewQuery()

			for p := range trabalho {
				de, para := c.Point(p.de), c.Point(p.para)

				metros, segundos, achou := q.Leg(p.de, p.para)
				resp, err := cliente.Rota(ctx, de, para)

				mu.Lock()
				switch {
				case !achou:
					rel.SemRotaNossa++
				case err != nil:
					var semRota ErrSemRota
					if errors.As(err, &semRota) {
						rel.SemRotaOsrm++
					} else {
						rel.Erros++
					}
				case max2(resp.Encaixe) > o.encaixeMax:
					rel.Descartados++
				default:
					rel.Casos = append(rel.Casos, Caso{
						De: de, Para: para,
						NossoMetros: metros, NossoSegs: segundos,
						OsrmMetros: resp.Metros, OsrmSegs: resp.Segundos,
						EncaixeMax: max2(resp.Encaixe),
					})
				}
				mu.Unlock()
			}
		}()
	}

	for _, p := range pares {
		trabalho <- p
	}
	close(trabalho)
	wg.Wait()

	return rel, nil
}

func max2(v [2]float64) float64 {
	if v[0] > v[1] {
		return v[0]
	}
	return v[1]
}
