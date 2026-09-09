package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Cliente conversa com um OSRM rodando em algum lugar.
//
// E o unico lugar da ERA que fala rede, e ele nao e parte do motor: nenhum
// pacote de maps importa net/http. A promessa de rodar dentro do processo de
// quem chama, sem servico e sem cota, continua valendo -- este programa
// existe para conferir o motor contra uma referencia externa, e roda quando
// alguem manda, nao quando uma rota e pedida.
type Cliente struct {
	Base   string
	HTTP   *http.Client
	Perfil string
}

// Resposta e o que interessa de uma consulta ao OSRM.
type Resposta struct {
	Metros   float64
	Segundos float64

	// Encaixe e a distancia, em metros, entre a coordenada perguntada e o
	// ponto onde o OSRM a colocou na malha. Importa mais do que parece: os
	// dois motores encaixam de jeitos diferentes -- o nosso no cruzamento
	// mais proximo, o OSRM em qualquer ponto de um trecho --, e um encaixe
	// distante quer dizer que os dois responderam sobre viagens diferentes.
	Encaixe [2]float64
}

// ErrSemRota indica que o OSRM nao achou caminho. Nao e falha do programa.
type ErrSemRota struct{ Codigo string }

func (e ErrSemRota) Error() string { return "osrm nao achou rota: " + e.Codigo }

// Rota pergunta ao OSRM o caminho entre dois pontos.
func (c *Cliente) Rota(ctx context.Context, de, para geo.Point) (Resposta, error) {
	url := c.montarURL(de, para)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Resposta{}, err
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Resposta{}, fmt.Errorf("chamando o osrm: %w", err)
	}
	defer resp.Body.Close()

	corpo, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return Resposta{}, fmt.Errorf("lendo a resposta do osrm: %w", err)
	}
	if resp.StatusCode != http.StatusOK && len(corpo) == 0 {
		return Resposta{}, fmt.Errorf("osrm respondeu %s", resp.Status)
	}

	return interpretar(corpo)
}

func (c *Cliente) montarURL(de, para geo.Point) string {
	perfil := c.Perfil
	if perfil == "" {
		perfil = "driving"
	}

	// Sete casas decimais: e a precisao com que o OpenStreetMap guarda
	// coordenada, e mandar mais seria fingir exatidao que o dado nao tem.
	g := func(v float64) string { return strconv.FormatFloat(v, 'f', 7, 64) }

	var b strings.Builder
	b.WriteString(strings.TrimSuffix(c.Base, "/"))
	b.WriteString("/route/v1/")
	b.WriteString(perfil)
	b.WriteString("/")
	b.WriteString(g(de.Lon))
	b.WriteString(",")
	b.WriteString(g(de.Lat))
	b.WriteString(";")
	b.WriteString(g(para.Lon))
	b.WriteString(",")
	b.WriteString(g(para.Lat))
	b.WriteString("?overview=false&alternatives=false&steps=false")
	return b.String()
}

// respostaOSRM e o pedaco do JSON que este programa le.
//
// Os campos que nao estao aqui sao ignorados de proposito: o OSRM manda
// geometria, instrucoes de virada e anotacoes que nao servem a comparacao, e
// pedir menos deixa a resposta menor e a leitura mais estavel entre versoes.
type respostaOSRM struct {
	Code   string `json:"code"`
	Routes []struct {
		Distance float64 `json:"distance"`
		Duration float64 `json:"duration"`
	} `json:"routes"`
	Waypoints []struct {
		Distance float64 `json:"distance"`
	} `json:"waypoints"`
}

func interpretar(corpo []byte) (Resposta, error) {
	var r respostaOSRM
	if err := json.Unmarshal(corpo, &r); err != nil {
		return Resposta{}, fmt.Errorf("resposta do osrm ilegivel: %w", err)
	}

	if r.Code != "Ok" {
		if r.Code == "" {
			return Resposta{}, fmt.Errorf("resposta do osrm sem codigo")
		}
		return Resposta{}, ErrSemRota{Codigo: r.Code}
	}
	if len(r.Routes) == 0 {
		return Resposta{}, ErrSemRota{Codigo: "sem rotas na resposta"}
	}

	out := Resposta{
		Metros:   r.Routes[0].Distance,
		Segundos: r.Routes[0].Duration,
	}
	for i := 0; i < len(r.Waypoints) && i < 2; i++ {
		out.Encaixe[i] = r.Waypoints[i].Distance
	}
	return out, nil
}
