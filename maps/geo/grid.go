package geo

import (
	"fmt"
	"math"
	"slices"
)

// meiaVoltaAoMundo e a maior distancia possivel entre dois pontos sobre a
// esfera: metade da circunferencia.
const meiaVoltaAoMundo = math.Pi * EarthRadius

// Match e um ponto encontrado por uma busca, com a distancia ate o centro
// da busca ja calculada.
type Match struct {
	ID     int     // o identificador que o chamador registrou em Add
	Point  Point   // onde o ponto esta
	Meters float64 // distancia esferica ate o centro da consulta
}

// Grid indexa pontos numa grade regular de latitude e longitude para
// responder buscas por proximidade sem varrer a colecao inteira.
//
// O problema que resolve: "quais dos meus 5.000 clientes estao a menos de
// 10 km deste veiculo?". Sem indice, sao 5.000 calculos de distancia por
// consulta -- e uma roteirizacao faz isso milhares de vezes. A grade
// transforma a pergunta em "olhe apenas as celulas que o retangulo de busca
// toca", e o custo passa a depender de quantos pontos ha por perto, nao de
// quantos existem no total.
//
// Por que uma grade e nao uma k-d tree ou um R-tree: a grade e trivial de
// ler, aceita insercao a qualquer momento sem rebalancear, e ganha das
// arvores quando os pontos sao razoavelmente espalhados -- que e o caso de
// uma carteira de clientes. Ela perde quando ha concentracao extrema, com
// milhares de pontos numa celula so. Se isso aparecer no uso real, troca-se
// a estrutura por tras sem mexer nesta API.
//
// Grid nao e seguro para uso concorrente durante escrita. Depois de montado
// e so leitura, varias goroutines podem consultar em paralelo.
type Grid struct {
	cell  float64 // lado da celula, em graus
	cols  int32   // quantas colunas de longitude cabem na volta ao mundo
	rows  int32   // quantas linhas de latitude cabem do polo ao polo
	ids   []int
	pts   []Point
	cells map[cellKey][]int32
}

type cellKey struct{ x, y int32 }

// NewGrid cria uma grade com celulas quadradas de cellDegrees graus de lado.
//
// Prefira NewGridForRadius, que traduz a escolha para a unidade em que o
// problema e pensado: o raio tipico de busca.
func NewGrid(cellDegrees float64) *Grid {
	if !(cellDegrees > 0) || cellDegrees > 90 {
		panic(fmt.Sprintf("geo: lado de celula invalido: %v graus", cellDegrees))
	}
	return &Grid{
		cell:  cellDegrees,
		cols:  int32(math.Ceil(360 / cellDegrees)),
		rows:  int32(math.Ceil(180 / cellDegrees)),
		cells: make(map[cellKey][]int32),
	}
}

// NewGridForRadius cria uma grade dimensionada para buscas de raio tipico
// radiusM.
//
// A celula fica do tamanho do raio. Menor que isso e a busca visita muitas
// celulas quase vazias e paga o custo do mapa a toa; muito maior e cada
// celula devolve candidatos demais que serao descartados. Errar por um fator
// de dois ou tres nao muda nada -- o que doi e errar por cem.
func NewGridForRadius(radiusM float64) *Grid {
	graus := radiusM / metrosPorGrauLat
	if graus < 1e-4 {
		graus = 1e-4 // ~11 m: abaixo disso o mapa custa mais que a busca
	}
	if graus > 45 {
		graus = 45
	}
	return NewGrid(graus)
}

// Add registra um ponto sob o identificador dado.
//
// O identificador e do chamador: o codigo do cliente, o indice numa fatia,
// o que for. A grade nao o interpreta e nao verifica se ha repetidos --
// registrar o mesmo id duas vezes cria duas entradas.
func (g *Grid) Add(id int, p Point) {
	if !p.Valid() {
		panic(fmt.Sprintf("geo: ponto invalido em Add(%d, %v)", id, p))
	}
	i := int32(len(g.pts))
	g.pts = append(g.pts, p)
	g.ids = append(g.ids, id)
	k := g.key(p)
	g.cells[k] = append(g.cells[k], i)
}

// Len devolve quantos pontos foram registrados.
func (g *Grid) Len() int { return len(g.pts) }

// Cells devolve quantas celulas estao ocupadas. Util para diagnostico: uma
// grade com quase todos os pontos numa celula so nao esta ajudando em nada.
func (g *Grid) Cells() int { return len(g.cells) }

// Within devolve todos os pontos a no maximo radiusM metros de center,
// ordenados do mais proximo para o mais distante.
//
// A distancia e a esferica de Haversine, nao a rodoviaria: o resultado e um
// conjunto de candidatos, nao uma resposta de roteirizacao. Um cliente do
// outro lado de um rio aparece aqui como vizinho.
func (g *Grid) Within(center Point, radiusM float64) []Match {
	if radiusM <= 0 || len(g.pts) == 0 {
		return nil
	}

	var out []Match
	g.varrerCaixa(center, radiusM, func(m Match) {
		out = append(out, m)
	})

	// slices.SortFunc em vez de sort.Slice: a segunda troca elementos por
	// reflexao, e ordenar candidatos e o passo mais caro de uma consulta com
	// muitos resultados.
	slices.SortFunc(out, func(a, b Match) int {
		switch {
		case a.Meters < b.Meters:
			return -1
		case a.Meters > b.Meters:
			return 1
		default:
			return 0
		}
	})
	return out
}

// varrerCaixa chama visitar para cada ponto a no maximo radiusM de center.
//
// Separado de Within porque nem todo mundo quer a lista inteira. Quem procura
// os k mais proximos precisa olhar os mesmos candidatos, mas nao precisa
// junta-los num vetor nem ordena-los -- e numa grade densa os candidatos
// descartados sao a esmagadora maioria.
//
// A ordem de visita nao e definida.
func (g *Grid) varrerCaixa(center Point, radiusM float64, visitar func(Match)) {
	b := BoxAround(center, radiusM)
	y0 := g.row(b.MinLat)
	y1 := g.row(b.MaxLat)

	varrer := func(x int32) {
		for y := y0; y <= y1; y++ {
			for _, i := range g.cells[cellKey{x, y}] {
				p := g.pts[i]
				d := Haversine(center, p)
				if d <= radiusM {
					visitar(Match{ID: g.ids[i], Point: p, Meters: d})
				}
			}
		}
	}

	switch {
	case b.MinLon <= -180 && b.MaxLon >= 180:
		// O retangulo cobre todas as longitudes. Isso acontece perto dos
		// polos, onde um raio modesto ja da a volta no planeta, e em
		// qualquer busca de raio continental.
		//
		// Precisa ser um caso a parte porque col(+180) e col(-180) sao a
		// mesma coluna -- sao o mesmo lugar --, entao tratar este retangulo
		// como uma faixa comum varreria uma coluna so.
		for x := int32(0); x < g.cols; x++ {
			varrer(x)
		}

	case b.CrossesAntimeridian():
		// O retangulo da a volta pelo Pacifico: sao duas faixas de colunas,
		// uma ate o fim do mapa e outra do comeco.
		for x := g.col(b.MinLon); x < g.cols; x++ {
			varrer(x)
		}
		for x := int32(0); x <= g.col(b.MaxLon); x++ {
			varrer(x)
		}

	default:
		for x := g.col(b.MinLon); x <= g.col(b.MaxLon); x++ {
			varrer(x)
		}
	}
}

// Nearest devolve os k pontos mais proximos de center, do mais proximo para
// o mais distante. Devolve menos que k se a grade tiver menos pontos.
//
// Funciona buscando num raio pequeno e dobrando-o ate encontrar k pontos.
// Isso e exato, e nao apenas aproximado, por um motivo simples: qualquer
// ponto fora do raio esta mais longe que todos os que estao dentro. Entao,
// assim que uma busca devolve k ou mais, os k primeiros dela ja sao os k mais
// proximos do planeta inteiro.
//
// E essa propriedade que permite chutar o raio inicial pela densidade da
// grade sem arriscar a resposta: um chute ruim custa uma varredura a mais,
// nunca um ponto errado.
func (g *Grid) Nearest(center Point, k int) []Match {
	if k <= 0 || len(g.pts) == 0 {
		return nil
	}

	// Guarda so os k melhores, em vez de juntar todos os candidatos do raio e
	// ordenar no fim.
	//
	// A diferenca aparece porque a grade e densa: um raio de uma celula ja
	// devolve dezenas ou centenas de candidatos, e quase todos serao
	// descartados. Ordenar duzentos para entregar um e trabalho que nao
	// precisava existir -- e a alocacao do vetor de candidatos era, sozinha,
	// a maior parte do custo da consulta.
	//
	// Vetor ordenado, e nao heap: k costuma ser 1 ou 10, e nesse tamanho
	// inserir num vetor pequeno ganha do heap por ser previsivel para o
	// processador. O teste de descarte -- comparar com o pior guardado -- e
	// O(1) e resolve a esmagadora maioria dos candidatos.
	melhores := make([]Match, 0, k)

	raio := g.raioInicial(k)
	for {
		melhores = melhores[:0]
		g.varrerCaixa(center, raio, func(m Match) {
			if len(melhores) == k {
				if m.Meters >= melhores[k-1].Meters {
					return
				}
				melhores = melhores[:k-1]
			}
			i := len(melhores)
			for i > 0 && melhores[i-1].Meters > m.Meters {
				i--
			}
			melhores = append(melhores, Match{})
			copy(melhores[i+1:], melhores[i:])
			melhores[i] = m
		})

		if len(melhores) >= k {
			return melhores
		}
		if raio >= meiaVoltaAoMundo {
			return melhores // a grade inteira foi varrida; nao existem k pontos
		}
		// Dobra, e nao quadruplica. Com a estimativa inicial ja mirando alto,
		// repetir e raro; quando acontece, um passo menor desperdica menos.
		raio *= 2
		if raio > meiaVoltaAoMundo {
			raio = meiaVoltaAoMundo
		}
	}
}

// raioInicial estima o raio que ja deve conter k pontos.
//
// A versao antiga comecava por uma celula inteira. Numa grade densa isso e
// caro sem necessidade: uma celula dimensionada para buscas de 10 km pode
// guardar centenas de pontos, e varrer todos eles para devolver o mais
// proximo e o grosso do custo da consulta -- nao a alocacao, como parecia.
//
// Aqui o raio sai da densidade medida da propria grade: se ha p pontos por
// celula ocupada e a celula tem lado L, a area que em media contem k pontos
// tem raio da ordem de L * sqrt(k/p) / 2.
//
// Errar nao quebra nada, so custa uma iteracao a mais ou uma varredura maior
// que a necessaria: qualquer que seja o raio, achar k pontos dentro dele
// significa que sao os k mais proximos do planeta. Por isso o palpite pode
// ser agressivo.
func (g *Grid) raioInicial(k int) float64 {
	lado := g.cell * metrosPorGrauLat
	if len(g.cells) == 0 {
		return lado
	}

	porCelula := float64(len(g.pts)) / float64(len(g.cells))
	if porCelula <= 0 {
		return lado
	}

	// O alvo nao e conter k pontos, e sim uns quatro k. Mirar em k faria a
	// estimativa errar para baixo em metade das consultas, e cada erro custa
	// uma varredura descartada mais uma de area quatro vezes maior. Sai mais
	// barato varrer um pouco a mais de primeira do que repetir.
	raio := lado * math.Sqrt(float64(k)/porCelula)

	// Piso: abaixo de um dezesseis avos da celula a varredura ja olha uma
	// celula so, e encolher mais nao economiza -- so adianta iteracoes.
	if minimo := lado / 16; raio < minimo {
		raio = minimo
	}
	if raio > lado {
		raio = lado
	}
	return raio
}

// key, col e row traduzem coordenadas para indices de celula.
//
// Os indices sao sempre limitados a faixa valida: um ponto exatamente no
// polo ou no antimeridiano cairia uma celula fora da grade por
// arredondamento, e a ultima celula o acolhe.

func (g *Grid) key(p Point) cellKey {
	return cellKey{x: g.col(p.Lon), y: g.row(p.Lat)}
}

func (g *Grid) col(lon float64) int32 {
	x := int32(math.Floor((NormalizeLon(lon) + 180) / g.cell))
	if x < 0 {
		return 0
	}
	if x >= g.cols {
		return g.cols - 1
	}
	return x
}

func (g *Grid) row(lat float64) int32 {
	y := int32(math.Floor((lat + 90) / g.cell))
	if y < 0 {
		return 0
	}
	if y >= g.rows {
		return g.rows - 1
	}
	return y
}
