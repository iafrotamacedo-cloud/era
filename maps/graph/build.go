package graph

import (
	"fmt"
	"io"
	"os"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/osm"
)

// Stats conta o que a construcao viu e o quanto ela encolheu o mapa.
//
// Existe porque a contracao de nos de grau 2 e a maior decisao deste pacote,
// e uma decisao dessas precisa vir com o numero do lado.
type Stats struct {
	ViasLidas  int // vias no arquivo
	ViasUsadas int // as que o perfil aceitou como estrada
	NosLidos   int // nos no arquivo

	NosDeVia    int // nos referenciados por alguma estrada
	Cruzamentos int // os que sobraram como no do grafo
	Trechos     int // ligacoes entre cruzamentos

	NosSemCoordenada   int // referenciados por via, ausentes do extrato
	TrechosDegenerados int // laco de um no nele mesmo, descartado
}

// Encolhimento devolve a fracao de nos de via que a contracao eliminou.
func (s Stats) Encolhimento() float64 {
	if s.NosDeVia == 0 {
		return 0
	}
	return 1 - float64(s.Cruzamentos)/float64(s.NosDeVia)
}

func (s Stats) String() string {
	return fmt.Sprintf("grafo: %d de %d vias, %d cruzamentos de %d nos de via (%.1f%% contraidos), %d trechos",
		s.ViasUsadas, s.ViasLidas, s.Cruzamentos, s.NosDeVia, s.Encolhimento()*100, s.Trechos)
}

// Stats devolve o que a construcao contou.
func (g *Graph) Stats() Stats { return g.stats }

// BuildFile monta o grafo a partir de um arquivo .osm.pbf.
func BuildFile(caminho string, p Profile) (*Graph, error) {
	f, err := os.Open(caminho)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Build(f, p)
}

// Build monta o grafo a partir de um .osm.pbf.
//
// Recebe io.ReadSeeker, e nao io.Reader, porque a construcao percorre o
// arquivo tres vezes:
//
//  1. vias, para contar quantas estradas referenciam cada no. E isso que
//     distingue cruzamento de ponto de curva, sem que o grafo precise ser
//     montado em tamanho cheio para depois encolher.
//  2. nos, para pegar a coordenada dos que interessam. Sao necessarias as de
//     todos os nos de via, e nao so as dos cruzamentos: o comprimento de um
//     trecho e a soma dos pedacos, e os pedacos passam pelos pontos de curva.
//  3. vias de novo, agora montando os trechos entre cruzamentos.
//
// As passadas 1 e 3 nao decodificam no nenhum -- o campo Node do handler fica
// nil --, que e o caminho barato que a Fase 3 mediu.
//
// A memoria de pico e proporcional ao numero de nos de via, nao ao tamanho do
// arquivo: um mapa de identificador para coordenada, umas dezenas de bytes
// por no. Para um estado sao centenas de MB; para o Brasil inteiro isso
// aperta, e a saida e o formato .eramap da Fase 5, que guarda o grafo pronto.
func Build(rs io.ReadSeeker, p Profile) (*Graph, error) {
	if len(p.SpeedKmh) == 0 {
		return nil, fmt.Errorf("graph: perfil sem nenhum tipo de via; use graph.Car()")
	}

	nos := map[int64]*noInfo{}
	var st Stats

	// Passada 1: quem e cruzamento.
	if err := passada(rs, osm.Handler{
		Way: func(w osm.Way) error {
			st.ViasLidas++
			if _, ok := p.avaliar(w); !ok {
				return nil
			}
			st.ViasUsadas++
			for i, ref := range w.Refs {
				info := nos[ref]
				if info == nil {
					info = &noInfo{id: NoNode}
					nos[ref] = info
				}
				// As pontas de uma via sao sempre cruzamento: e onde ela
				// encontra a proxima, ou onde a rua simplesmente acaba.
				if i == 0 || i == len(w.Refs)-1 {
					info.contagem = 2
				} else if info.contagem < 2 {
					info.contagem++
				}
			}
			return nil
		},
	}); err != nil {
		return nil, err
	}
	st.NosDeVia = len(nos)

	// Passada 2: onde cada um esta.
	if err := passada(rs, osm.Handler{
		Node: func(n osm.Node) error {
			st.NosLidos++
			if info, ok := nos[n.ID]; ok {
				info.p = n.Point
				info.temPonto = true
			}
			return nil
		},
	}); err != nil {
		return nil, err
	}

	g := &Graph{}
	var trechos []trecho

	// Passada 3: os trechos entre cruzamentos.
	if err := passada(rs, osm.Handler{
		Way: func(w osm.Way) error {
			v, ok := p.avaliar(w)
			if !ok {
				return nil
			}
			trechos = g.montarVia(w, v, nos, trechos, &st)
			return nil
		},
	}); err != nil {
		return nil, err
	}

	// Qualquer no de via sem coordenada conta, e nao so os cruzamentos: um
	// ponto de curva ausente parte a rua em duas tanto quanto um cruzamento
	// ausente, porque nao da para medir a distancia atraves dele.
	for _, info := range nos {
		if !info.temPonto {
			st.NosSemCoordenada++
		}
	}
	st.Cruzamentos = len(g.pontos)
	st.Trechos = len(trechos)
	g.stats = st

	g.montarCSR(trechos)
	return g, nil
}

// noInfo e o que a construcao sabe de um no do OpenStreetMap enquanto monta.
type noInfo struct {
	p        geo.Point
	temPonto bool
	contagem uint8  // quantas estradas o citam, saturado em 2
	id       NodeID // o indice no grafo, atribuido na terceira passada
}

func (n *noInfo) cruzamento() bool { return n.contagem >= 2 }

// trecho e uma ligacao entre dois cruzamentos, antes de virar CSR.
type trecho struct {
	de, para NodeID
	metros   float32
	segundos float32
	frente   bool
	tras     bool
}

// montarVia percorre uma via, soma os pedacos entre cruzamentos e produz um
// trecho por par de cruzamentos consecutivos.
//
// E aqui que a contracao de grau 2 acontece: um no citado por uma estrada so,
// no meio dela, nunca vira no do grafo -- vira comprimento somado.
func (g *Graph) montarVia(w osm.Way, v via, nos map[int64]*noInfo, trechos []trecho, st *Stats) []trecho {
	de := -1 // indice, dentro de w.Refs, do ultimo cruzamento com coordenada
	var acumulado float64

	for i, ref := range w.Refs {
		info := nos[ref]

		// Um no sem coordenada e um no que o extrato nao trouxe -- via
		// cortada na fronteira do recorte. Nao da para medir a distancia
		// atraves dele, entao o trecho e interrompido e recomeca no proximo
		// cruzamento conhecido.
		if info == nil || !info.temPonto {
			de, acumulado = -1, 0
			continue
		}

		if de == -1 {
			if info.cruzamento() {
				de, acumulado = i, 0
			}
			continue
		}

		acumulado += geo.Haversine(nos[w.Refs[i-1]].p, info.p)

		if !info.cruzamento() {
			continue
		}

		origem := g.atribuir(nos[w.Refs[de]], w.Refs[de])
		destino := g.atribuir(info, ref)

		if origem == destino {
			// Um laco do no nele mesmo nao leva a lugar nenhum. Acontece em
			// rotatoria mapeada como via fechada com um cruzamento so.
			st.TrechosDegenerados++
		} else {
			trechos = append(trechos, trecho{
				de:       origem,
				para:     destino,
				metros:   float32(acumulado),
				segundos: float32(acumulado / (v.speedKmh / 3.6)),
				frente:   v.frente,
				tras:     v.tras,
			})
		}

		de, acumulado = i, 0
	}
	return trechos
}

// atribuir da a um cruzamento o seu indice no grafo, na primeira vez que ele
// e usado.
//
// A ordem de atribuicao segue a ordem do arquivo, e nao a iteracao do mapa,
// que em Go e deliberadamente aleatoria. E o que faz duas construcoes do
// mesmo arquivo produzirem exatamente o mesmo grafo.
func (g *Graph) atribuir(info *noInfo, osmID int64) NodeID {
	if info.id == NoNode {
		info.id = NodeID(len(g.pontos))
		g.pontos = append(g.pontos, info.p)
		g.osmIDs = append(g.osmIDs, osmID)
	}
	return info.id
}

// montarCSR transforma a lista de trechos na estrutura final.
//
// Cada trecho vira duas arestas, uma em cada ponta, com as bandeiras de
// sentido trocadas de lado -- e o que permite a busca para tras andar sem que
// exista um grafo invertido na memoria.
func (g *Graph) montarCSR(trechos []trecho) {
	n := len(g.pontos)
	g.offsets = make([]uint32, n+1)

	for _, t := range trechos {
		g.offsets[t.de+1]++
		g.offsets[t.para+1]++
	}
	for i := 1; i <= n; i++ {
		g.offsets[i] += g.offsets[i-1]
	}

	g.arestas = make([]Edge, len(trechos)*2)
	posicao := make([]uint32, n)
	copy(posicao, g.offsets[:n])

	for _, t := range trechos {
		g.arestas[posicao[t.de]] = Edge{
			To: t.para, Meters: t.metros, Seconds: t.segundos,
			Forward: t.frente, Backward: t.tras,
		}
		posicao[t.de]++

		// Da outra ponta os sentidos se invertem: sair de para rumo a de so e
		// possivel se o trecho admitia o sentido contrario.
		g.arestas[posicao[t.para]] = Edge{
			To: t.de, Meters: t.metros, Seconds: t.segundos,
			Forward: t.tras, Backward: t.frente,
		}
		posicao[t.para]++
	}
}

// passada roda uma varredura do arquivo desde o inicio.
func passada(rs io.ReadSeeker, h osm.Handler) error {
	if _, err := rs.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("graph: nao consegui voltar ao inicio do arquivo: %w", err)
	}
	return osm.Scan(rs, h)
}
