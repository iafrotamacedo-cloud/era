package graph

import (
	"strconv"
	"strings"

	"github.com/iafrotamacedo-cloud/era/maps/osm"
)

// Profile decide o que e estrada e quao rapido se anda nela.
//
// O OpenStreetMap nao tem tipos: uma via so vira rua porque alguem escreveu
// highway=residential nela. Decidir quais dessas etiquetas contam e uma
// politica, nao um fato, e ela muda com o veiculo -- uma bicicleta usa a
// ciclovia que o caminhao nao usa, e o caminhao respeita restricao de peso
// que o carro ignora.
//
// Por isso e um valor, e nao regra fixa no codigo.
type Profile struct {
	// SpeedKmh diz a velocidade de cada valor de highway=. Uma via cujo tipo
	// nao esta aqui nao entra no grafo -- e assim que a selecao acontece.
	SpeedKmh map[string]float64

	// MaxSpeedTag manda respeitar a etiqueta maxspeed quando ela existir e
	// for legivel.
	MaxSpeedTag bool

	// SentidoUnicoImplicito sao os pares chave=valor que implicam mao unica
	// mesmo sem oneway=yes. Rotatoria e via expressa sao os casos classicos.
	SentidoUnicoImplicito map[string]string

	// Bloqueadas sao pares chave=valor que excluem a via, mesmo que o tipo
	// dela esteja em SpeedKmh. Uma chave pode ter varios valores.
	Bloqueadas map[string][]string

	// Acesso lista as chaves de permissao, da mais especifica para a mais
	// geral.
	//
	// A primeira que existir na via decide, e as demais nem sao consultadas.
	// E assim que o OpenStreetMap resolve permissao, e sem isso o modelo erra
	// nos dois sentidos: uma servidao marcada access=private mas
	// motor_vehicle=yes ficaria de fora do grafo, e uma via liberada de modo
	// geral mas com motor_vehicle=no entraria.
	Acesso []string

	// AcessoNegado sao os valores dessas chaves que excluem a via.
	//
	// O que NAO esta aqui e permitido, e a lista do que fica de fora e uma
	// decisao sobre logistica, nao sobre o formato: destination, delivery e
	// customers descrevem exatamente o caminhao que vai entregar. Bloquea-los
	// faria a roteirizacao recusar o condominio, o posto e a fazenda -- os
	// lugares para onde a entrega vai.
	AcessoNegado map[string]bool

	// SurfaceMaxKmh limita a velocidade conforme a etiqueta surface.
	//
	// Teto, e nao multiplicador. A primeira versao multiplicava, e a
	// comparacao com o OSRM mostrou o erro: um unclassified de terra virava
	// 40 x 0,45 = 18 km/h, e a rota fugia do barro por um desvio de 77 km no
	// asfalto onde o OSRM fazia 52. Estrada de barro e lenta em termos
	// absolutos, nao em proporcao a classe da via -- um primary de terra nao
	// anda a 70% de 70, anda a velocidade de estrada de terra.
	//
	// Reduz a velocidade em vez de excluir a via, e isso e uma decisao sobre
	// logistica no Brasil, nao sobre o formato: no sertao a estrada de barro
	// as vezes e o unico caminho ate o cliente. Tirar essas vias do grafo
	// faria a roteirizacao dizer que nao ha rota para lugares onde os
	// caminhoes chegam todo dia.
	//
	// O efeito aparece no tempo, nao na distancia -- barro nao encurta nem
	// alonga a estrada. Mas e o tempo que a busca minimiza numa roteirizacao
	// de verdade, e com isso o desvio pelo asfalto ganha quando compensa, e so
	// quando compensa.
	//
	// Um valor de surface que nao esteja aqui, ou a ausencia da etiqueta, nao
	// muda nada.
	SurfaceMaxKmh map[string]float64
}

// Car devolve o perfil de carro de passeio.
//
// As velocidades sao genericas, tiradas dos limites usuais por tipo de via.
// Nao foram medidas em operacao nenhuma -- e sao a parte deste pacote com
// maior chance de estar errada para uma frota especifica.
//
// A fonte honesta delas e o hodometro: o dist.Calibrate da Fase 2 mede
// velocidade efetiva por faixa de distancia a partir do historico real, ja
// com transito e parada embutidos. Quem tiver esse historico deveria
// sobrescrever estes numeros com o que ele disser.
func Car() Profile {
	return Profile{
		SpeedKmh: map[string]float64{
			"motorway": 100, "motorway_link": 60,
			"trunk": 85, "trunk_link": 50,
			"primary": 70, "primary_link": 45,
			"secondary": 60, "secondary_link": 40,
			"tertiary": 50, "tertiary_link": 35,
			"unclassified":  40,
			"residential":   30,
			"living_street": 10,
			"service":       20,
			"road":          30,
		},
		MaxSpeedTag: true,
		SentidoUnicoImplicito: map[string]string{
			"junction": "roundabout",
			"highway":  "motorway",
		},
		Bloqueadas: map[string][]string{
			// Uma via marcada como area nao e caminho: e o desenho de uma
			// praca, de um patio, de um estacionamento visto de cima.
			"area": {"yes"},
		},

		// Da mais especifica para a mais geral. motor_vehicle fala do que
		// interessa a um carro; vehicle inclui a bicicleta junto; access vale
		// para todo mundo, inclusive o pedestre.
		Acesso: []string{"motor_vehicle", "vehicle", "access"},

		AcessoNegado: map[string]bool{
			"no":      true,
			"private": true,

			// So o veiculo daquele tipo passa, e nenhum deles e o nosso.
			"agricultural": true,
			"forestry":     true,
			"military":     true,
			"emergency":    true,

			// Fora daqui, e portanto permitidos: yes, permissive, designated,
			// destination, delivery, customers. Os tres ultimos sao a decisao
			// de logistica -- descrevem exatamente quem esta indo entregar.
		},

		SurfaceMaxKmh: superficiesCarro(),
	}
}

// superficiesCarro devolve o teto de velocidade por superficie.
//
// Vieram da comparacao com o OSRM sobre o Nordeste, em duas rodadas. A
// primeira mostrou a ERA cortando caminho por estradas de barro do sertao a
// 40 km/h, como se fossem asfalto. A segunda, ja com penalidade, mostrou o
// contrario -- fugindo do barro por desvios longos demais --, e foi ela que
// revelou que o modelo certo e teto, e nao multiplicador.
//
// Sao estimativas, e nao medicoes: dizem a ordem de grandeza, nao a velocidade
// da frota de ninguem. Como as velocidades base, a fonte honesta e o
// hodometro -- o dist.Calibrate da Fase 2.
//
// Duas coisas que isto nao resolve, e vale saber:
//
// A maioria das vias no interior simplesmente nao tem a etiqueta. Na rota que
// motivou este mapa, 26 de 49 vias vinham sem surface, e para essas nada muda.
//
// E paralelepipedo e barro nao sao a mesma coisa por motivos diferentes: um e
// firme e desconfortavel, o outro e liso e traicoeiro na chuva. O teto so
// captura "nao passa disso", nao "por que".
//
// Superficie boa -- asfalto, concreto -- nao aparece aqui: nao ha teto a impor,
// e a velocidade da classe da via vale inteira.
func superficiesCarro() map[string]float64 {
	return map[string]float64{
		// Pavimento ruim: firme, mas ninguem corre.
		"concrete:plates": 60,
		"paving_stones":   40,
		"sett":            30,
		"cobblestone":     30,

		// Sem pavimento, do melhor para o pior.
		"compacted":   50,
		"fine_gravel": 40,
		"gravel":      30,
		"pebblestone": 30,
		"unpaved":     30,
		"dirt":        25,
		"earth":       25,
		"ground":      25,
		"grass":       20,
		"sand":        15,
		"mud":         10,
	}
}

// via e o que o perfil extraiu de uma way: se entra no grafo, com que
// velocidade e em quais sentidos.
type via struct {
	speedKmh float64
	frente   bool
	tras     bool
}

// avaliar aplica o perfil a uma via do OpenStreetMap.
//
// Devolve ok=false quando a via nao e estrada para este perfil.
func (p Profile) avaliar(w osm.Way) (via, bool) {
	if len(w.Refs) < 2 {
		return via{}, false // uma via de um no so nao liga nada
	}

	tipo, ok := w.Tags.Get("highway")
	if !ok {
		return via{}, false
	}
	velocidade, ok := p.SpeedKmh[tipo]
	if !ok {
		return via{}, false
	}

	for chave, valores := range p.Bloqueadas {
		v, ok := w.Tags.Get(chave)
		if !ok {
			continue
		}
		for _, bloqueado := range valores {
			if v == bloqueado {
				return via{}, false
			}
		}
	}

	if !p.permite(w) {
		return via{}, false
	}

	if p.MaxSpeedTag {
		if v, ok := w.Tags.Get("maxspeed"); ok {
			if kmh, ok := lerMaxspeed(v); ok {
				velocidade = kmh
			}
		}
	}
	// A superficie entra depois do maxspeed, e de proposito: o limite legal e
	// o que a placa permite, a superficie e o que o chao entrega. Numa estrada
	// vicinal com maxspeed=60 e surface=dirt, quem manda e o barro.
	if teto, ok := p.SurfaceMaxKmh[superficieDe(w)]; ok && teto < velocidade {
		velocidade = teto
	}

	if velocidade <= 0 {
		return via{}, false
	}

	frente, tras := true, true
	switch v, _ := w.Tags.Get("oneway"); v {
	case "yes", "true", "1":
		tras = false
	case "-1", "reverse":
		frente = false
	case "no", "false", "0":
		// Explicito vence o implicito: uma rotatoria marcada oneway=no e de
		// mao dupla, por mais estranho que pareca. Quem mapeou viu a rua.
		return via{speedKmh: velocidade, frente: true, tras: true}, true
	default:
		for chave, valor := range p.SentidoUnicoImplicito {
			if v, ok := w.Tags.Get(chave); ok && v == valor {
				tras = false
				break
			}
		}
	}

	return via{speedKmh: velocidade, frente: frente, tras: tras}, true
}

// permite resolve as etiquetas de acesso da via.
//
// A regra do OpenStreetMap e de precedencia, e nao de acumulo: vale a chave
// mais especifica que existir, e as mais gerais nem sao olhadas. Uma servidao
// marcada
//
//	access=private
//	motor_vehicle=yes
//
// esta liberada para carro, apesar do access. E uma via com
//
//	access=yes
//	motor_vehicle=no
//
// esta fechada, apesar do access. Um modelo que so olhasse pares chave=valor
// erraria nos dois casos, e em sentidos opostos.
//
// Sem nenhuma das chaves, a via passa: no OpenStreetMap o silencio quer dizer
// permitido, e presumir o contrario apagaria a maior parte da malha.
func (p Profile) permite(w osm.Way) bool {
	for _, chave := range p.Acesso {
		v, ok := w.Tags.Get(chave)
		if !ok {
			continue
		}
		return !p.AcessoNegado[v]
	}
	return true
}

// superficieDe devolve a superficie da via.
//
// Prefere surface, e cai para tracktype quando ela falta. Sao etiquetas
// diferentes com a mesma pergunta por tras -- tracktype existe para estradas
// vicinais e vai de grade1, que e quase pavimento, a grade5, que e trilha de
// terra solta. Traduzir uma na outra aproveita a resposta de quem mapeou sem
// obrigar quem usa a conhecer as duas.
func superficieDe(w osm.Way) string {
	if s, ok := w.Tags.Get("surface"); ok {
		return s
	}
	switch t, _ := w.Tags.Get("tracktype"); t {
	case "grade1":
		return "compacted"
	case "grade2":
		return "gravel"
	case "grade3":
		return "unpaved"
	case "grade4":
		return "dirt"
	case "grade5":
		return "ground"
	}
	return ""
}

// lerMaxspeed interpreta a etiqueta maxspeed.
//
// O valor e texto livre e aparece de todo jeito: "60", "60 km/h", "50 mph",
// "BR:urban", "walk", "none". Le-se o numero quando ha um, converte-se mph, e
// desiste-se do resto -- desistir devolve a velocidade padrao do tipo da via,
// que e melhor que um chute sobre o que "BR:urban" significa numa rodovia.
func lerMaxspeed(s string) (float64, bool) {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return 0, false
	}

	mph := strings.HasSuffix(s, "mph")
	campos := strings.Fields(strings.TrimSuffix(strings.TrimSuffix(s, "mph"), "km/h"))
	if len(campos) == 0 {
		return 0, false
	}

	v, err := strconv.ParseFloat(campos[0], 64)
	if err != nil || v <= 0 || v > 300 {
		return 0, false
	}
	if mph {
		v *= 1.609344
	}
	return v, true
}
