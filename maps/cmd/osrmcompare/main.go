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
// Baixe um extrato e suba um OSRM sobre ele. A Geofabrik divide o Brasil em
// cinco regioes, e nao em estados -- o menor recorte que contem o Ceara e o
// nordeste, com uns 420 MB:
//
//	curl -L -O https://download.geofabrik.de/south-america/brazil/nordeste-latest.osm.pbf
//
//	docker run -t -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-extract -p /opt/car.lua /data/nordeste-latest.osm.pbf
//	docker run -t -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-partition /data/nordeste-latest.osrm
//	docker run -t -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-customize /data/nordeste-latest.osrm
//	docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend \
//	    osrm-routed --algorithm mld /data/nordeste-latest.osrm
//
// E compare, sobre o mesmo extrato:
//
//	go run ./maps/cmd/osrmcompare -mapa nordeste-latest.osm.pbf -n 500
//
// # Sem instalar nada
//
// Da para usar o servidor publico de demonstracao do OSRM em vez de subir um.
// Ele nao tem SLA e existe para testes, entao va devagar e com poucos pares:
//
//	go run ./maps/cmd/osrmcompare -mapa nordeste-latest.osm.pbf \
//	    -osrm https://router.project-osrm.org \
//	    -n 150 -paralelo 1 -pausa 300ms -sem-hierarquia
//
// A ressalva importante: o servidor publico roteia sobre o planeta inteiro, e
// o grafo daqui foi montado de um recorte. Perto da borda do recorte as duas
// respostas divergem por um motivo que nao e erro de ninguem -- a nossa malha
// acaba e a dele nao. Use -caixa para manter a amostra no miolo do extrato.
//
// # Sobre -sem-hierarquia
//
// Preparar a hierarquia de um extrato regional leva horas. Ela acelera a
// consulta e nao muda a resposta -- e os testes garantem que as duas
// concordam --, entao para conferir o motor contra o OSRM ela nao e
// necessaria. Com -sem-hierarquia a consulta usa o Dijkstra bidirecional da
// Fase 4, que nao precisa de preparo nenhum.
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
	"strconv"
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
	pausa := flag.Duration("pausa", 0, "espera entre consultas de cada trabalhador; use contra servidor publico")
	salvar := flag.String("salvar-eramap", "", "grava a hierarquia preparada neste caminho")
	semAcesso := flag.Bool("sem-acesso", false, "ignora as etiquetas de acesso; serve para comparar perfis")
	semCH := flag.Bool("sem-hierarquia", false, "consulta pelo Dijkstra da Fase 4, sem preparar a hierarquia")
	metrica := flag.String("metrica", "tempo", "o que minimizar aqui: tempo ou distancia")
	caixa := flag.String("caixa", "", "limita o sorteio a um retangulo: minLat,minLon,maxLat,maxLon")
	flag.Parse()

	if *mapa == "" {
		fmt.Fprintln(os.Stderr, "falta -mapa; use -h para ver as opcoes")
		os.Exit(2)
	}

	o := opcoes{
		mapa: *mapa, base: *base, perfil: *perfil,
		n: *n, semente: *semente,
		minM: *minKm * 1000, maxM: *maxKm * 1000,
		encaixeMax: *encaixeMax,
		paralelo:   *paralelo, espera: *espera, pausa: *pausa,
		salvar: *salvar, semCH: *semCH, semAcesso: *semAcesso,
	}

	// O OSRM responde a rota mais rapida; comparar a nossa mais curta com a
	// mais rapida dele mede a diferenca entre dois objetivos, nao entre dois
	// motores. Por isso o padrao aqui e tempo.
	switch *metrica {
	case "tempo":
		o.metrica = graph.Time
	case "distancia":
		o.metrica = graph.Distance
	default:
		fmt.Fprintln(os.Stderr, "erro: -metrica aceita tempo ou distancia")
		os.Exit(2)
	}

	if *caixa != "" {
		b, err := lerCaixa(*caixa)
		if err != nil {
			fmt.Fprintln(os.Stderr, "erro:", err)
			os.Exit(2)
		}
		o.caixa, o.temCaixa = b, true
	}

	if err := rodar(o); err != nil {
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
	pausa              time.Duration
	salvar             string
	semCH              bool
	semAcesso          bool
	caixa              geo.Box
	temCaixa           bool
	metrica            graph.Metric
}

// lerCaixa interpreta "minLat,minLon,maxLat,maxLon".
//
// Serve para restringir a amostra a uma regiao dentro do extrato -- so o
// Ceara dentro do nordeste, por exemplo. Importa quando a referencia e o
// servidor publico, que roteia sobre o planeta: perto da borda do recorte as
// duas respostas divergem porque a nossa malha acaba, e nao porque alguem
// errou.
func lerCaixa(s string) (geo.Box, error) {
	var b geo.Box
	partes := strings.Split(s, ",")
	if len(partes) != 4 {
		return b, fmt.Errorf("caixa precisa de quatro numeros: minLat,minLon,maxLat,maxLon")
	}

	campos := []*float64{&b.MinLat, &b.MinLon, &b.MaxLat, &b.MaxLon}
	for i, p := range partes {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return b, fmt.Errorf("caixa: %q nao e um numero", p)
		}
		*campos[i] = v
	}
	if b.MinLat >= b.MaxLat || b.MinLon >= b.MaxLon {
		return b, fmt.Errorf("caixa invertida: %+v", b)
	}
	return b, nil
}

func rodar(o opcoes) error {
	m, err := carregar(o)
	if err != nil {
		return err
	}

	pares := sortearPares(m, o)
	if len(pares) == 0 {
		return fmt.Errorf("nao consegui sortear nenhum par entre %.0f e %.0f km; ajuste -min-km, -max-km ou -caixa",
			o.minM/1000, o.maxM/1000)
	}
	fmt.Printf("comparando %d pares contra %s\n\n", len(pares), o.base)

	rel, err := comparar(m, pares, o)
	if err != nil {
		return err
	}
	fmt.Print(rel)
	return nil
}

// motor e o que a comparacao precisa de um mecanismo de rotas.
//
// Existe porque a hierarquia nao e obrigatoria para conferir o motor contra o
// OSRM: ela acelera a consulta, nao muda a resposta, e os testes garantem que
// as duas concordam. Num extrato regional preparar a hierarquia leva horas, e
// esperar por ela so para descobrir se uma etiqueta foi lida errado seria
// pagar caro por nada.
type motor interface {
	Len() int
	Point(graph.NodeID) geo.Point
	Nearest(geo.Point) (graph.NodeID, float64, bool)
	NovaConsulta() consulta
}

// consulta responde um par. Uma por goroutine.
type consulta interface {
	Leg(de, para graph.NodeID) (metros, segundos float64, ok bool)
}

// motorCH usa a hierarquia da Fase 5.
type motorCH struct{ c *ch.CH }

func (m motorCH) Len() int                       { return m.c.Len() }
func (m motorCH) Point(n graph.NodeID) geo.Point { return m.c.Point(n) }
func (m motorCH) NovaConsulta() consulta         { return m.c.NewQuery() }
func (m motorCH) Nearest(p geo.Point) (graph.NodeID, float64, bool) {
	return m.c.Nearest(p)
}

// motorGrafo usa o Dijkstra bidirecional da Fase 4.
type motorGrafo struct {
	g *graph.Graph
	m graph.Metric
}

func (m motorGrafo) Len() int                       { return m.g.Len() }
func (m motorGrafo) Point(n graph.NodeID) geo.Point { return m.g.Point(n) }
func (m motorGrafo) Nearest(p geo.Point) (graph.NodeID, float64, bool) {
	return m.g.Nearest(p)
}

func (m motorGrafo) NovaConsulta() consulta {
	return consultaGrafo{s: m.g.NewSearcher(), m: m.m}
}

type consultaGrafo struct {
	s *graph.Searcher
	m graph.Metric
}

func (c consultaGrafo) Leg(de, para graph.NodeID) (float64, float64, bool) {
	p, ok := c.s.Route(de, para, c.m)
	if !ok {
		return 0, 0, false
	}
	return p.Meters, p.Seconds, true
}

// carregar monta o motor a partir do que foi pedido.
func carregar(o opcoes) (motor, error) {
	if strings.EqualFold(filepath.Ext(o.mapa), ".eramap") {
		fmt.Printf("carregando %s\n", o.mapa)
		c, err := ch.LoadFile(o.mapa)
		if err != nil {
			return nil, err
		}
		fmt.Println(c.Stats())
		return motorCH{c}, nil
	}

	fmt.Printf("montando o grafo de %s\n", o.mapa)
	inicio := time.Now()
	perfil := graph.Car()
	if o.semAcesso {
		perfil.Acesso = nil
		fmt.Println("ignorando as etiquetas de acesso")
	}
	g, err := graph.BuildFile(o.mapa, perfil)
	if err != nil {
		return nil, err
	}
	fmt.Printf("%s, em %s\n", g.Stats(), time.Since(inicio).Round(time.Millisecond))

	if o.semCH {
		fmt.Printf("consultando pelo Dijkstra da Fase 4, minimizando %v\n", o.metrica)
		return motorGrafo{g: g, m: o.metrica}, nil
	}

	fmt.Println("preparando a hierarquia (isto demora; use -sem-hierarquia para pular)")
	c, err := ch.Prepare(g, o.metrica)
	if err != nil {
		return nil, err
	}
	fmt.Println(c.Stats())

	if o.salvar != "" {
		if err := c.SaveFile(o.salvar); err != nil {
			return nil, fmt.Errorf("gravando o eramap: %w", err)
		}
		fmt.Printf("hierarquia gravada em %s\n", o.salvar)
	}
	return motorCH{c}, nil
}

type par struct{ de, para graph.NodeID }

// sortearPares sorteia coordenadas no retangulo e as encaixa em cruzamentos.
//
// # Por que coordenadas, e nao indices de no
//
// A primeira versao sorteava indices do grafo direto. Funcionava para medir a
// qualidade de uma versao, e nao servia para comparar duas: mudar o perfil
// muda quantas vias entram, os indices deslocam, e a amostra vira outra. Uma
// mudanca de perfil aparecia misturada com uma troca de amostra, e nao havia
// como saber qual dos dois moveu o numero.
//
// Sorteando coordenadas e encaixando depois, a amostra e a mesma enquanto os
// cruzamentos continuarem existindo -- que e o caso para quase todos.
//
// # Por que ainda se encaixa antes de perguntar
//
// As coordenadas mandadas ao OSRM sao as dos cruzamentos, e nao as sorteadas.
// Cada motor encaixa a sua maneira, e mandar a coordenada crua faria boa parte
// do erro medido ser a diferenca entre os dois encaixes, e nao entre as duas
// rotas. Sobre um cruzamento os dois concordam, e o quanto o OSRM se afastou
// vem na resposta dele.
func sortearPares(m motor, o opcoes) []par {
	r := rand.New(rand.NewSource(o.semente))
	pares := make([]par, 0, o.n)

	caixa := o.caixa
	if !o.temCaixa {
		return nil // sem retangulo nao ha onde sortear coordenada
	}

	// Encaixe folgado: a coordenada cai no mato e o cruzamento mais proximo
	// pode estar a alguns quilometros. Longe demais e uma regiao sem estrada
	// mapeada, e o par nao diz nada sobre roteamento.
	const encaixeMaximo = 2000.0

	sortear := func() (graph.NodeID, bool) {
		p := geo.Point{
			Lat: caixa.MinLat + r.Float64()*(caixa.MaxLat-caixa.MinLat),
			Lon: caixa.MinLon + r.Float64()*(caixa.MaxLon-caixa.MinLon),
		}
		n, metros, ok := m.Nearest(p)
		if !ok || metros > encaixeMaximo {
			return graph.NoNode, false
		}
		return n, true
	}

	// Um teto de tentativas: num extrato recortado, ou com uma caixa
	// apertada, pode simplesmente nao haver pares na faixa pedida, e o
	// programa precisa dizer isso em vez de girar para sempre.
	for tentativas := 0; len(pares) < o.n && tentativas < o.n*500; tentativas++ {
		de, ok := sortear()
		if !ok {
			continue
		}
		para, ok := sortear()
		if !ok || de == para {
			continue
		}
		reta := geo.Haversine(m.Point(de), m.Point(para))
		if reta < o.minM || reta > o.maxM {
			continue
		}
		pares = append(pares, par{de, para})
	}
	return pares
}

func comparar(mot motor, pares []par, o opcoes) (*Relatorio, error) {
	cliente := &Cliente{
		Base:   o.base,
		Perfil: o.perfil,
		HTTP:   &http.Client{Timeout: o.espera},
	}

	// Uma consulta antes de tudo, em serie, para falhar cedo e com mensagem
	// clara se o OSRM nao estiver de pe. Descobrir isso depois de duzentas
	// goroutines terem falhado daria uma pilha de erros iguais.
	ctx := context.Background()
	if _, err := cliente.Rota(ctx, mot.Point(pares[0].de), mot.Point(pares[0].para)); err != nil {
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
			q := mot.NovaConsulta()

			for p := range trabalho {
				de, para := mot.Point(p.de), mot.Point(p.para)

				metros, segundos, achou := q.Leg(p.de, p.para)
				resp, err := cliente.Rota(ctx, de, para)
				if o.pausa > 0 {
					time.Sleep(o.pausa)
				}

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
