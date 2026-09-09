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
	// dela esteja em SpeedKmh.
	Bloqueadas map[string]string
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
		Bloqueadas: map[string]string{
			"access":        "private",
			"motor_vehicle": "no",
			"area":          "yes",
		},
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

	for chave, valor := range p.Bloqueadas {
		if v, ok := w.Tags.Get(chave); ok && v == valor {
			return via{}, false
		}
	}

	if p.MaxSpeedTag {
		if v, ok := w.Tags.Get("maxspeed"); ok {
			if kmh, ok := lerMaxspeed(v); ok {
				velocidade = kmh
			}
		}
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
