package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Um OSRM falso, montado com httptest.
//
// E o que permite testar o programa inteiro -- montagem da URL, chamada,
// leitura da resposta, contabilidade do relatorio -- sem Docker, sem extrato
// de mapa e sem rede. O que fica de fora e se o OSRM de verdade responde o que
// este falso responde, e isso nenhum teste resolve: resolve-se rodando o
// programa contra o OSRM de verdade, que e para o que ele existe.

func servidorFalso(t *testing.T, responder func(w http.ResponseWriter, r *http.Request)) *Cliente {
	t.Helper()
	s := httptest.NewServer(http.HandlerFunc(responder))
	t.Cleanup(s.Close)
	return &Cliente{Base: s.URL, HTTP: s.Client()}
}

func TestMontarURL(t *testing.T) {
	c := &Cliente{Base: "http://localhost:5000/", Perfil: "driving"}
	got := c.montarURL(
		geo.Point{Lat: -3.7319, Lon: -38.5267},
		geo.Point{Lat: -3.8, Lon: -38.6},
	)

	// A barra do fim da base nao pode virar barra dupla, e a ordem tem de ser
	// longitude antes de latitude -- o OSRM usa a ordem do GeoJSON, ao
	// contrario de quase todo mundo. Trocar as duas devolve rotas do oceano
	// sem dar erro.
	querido := "http://localhost:5000/route/v1/driving/" +
		"-38.5267000,-3.7319000;-38.6000000,-3.8000000" +
		"?overview=false&alternatives=false&steps=false"
	if got != querido {
		t.Errorf("url =\n  %s\nesperada:\n  %s", got, querido)
	}

	if _, err := url.Parse(got); err != nil {
		t.Errorf("url invalida: %v", err)
	}
}

func TestPerfilPadrao(t *testing.T) {
	c := &Cliente{Base: "http://x"}
	if got := c.montarURL(geo.Point{}, geo.Point{}); !strings.Contains(got, "/route/v1/driving/") {
		t.Errorf("sem perfil deveria usar driving, veio %s", got)
	}
}

func TestRotaLeARespostaBoa(t *testing.T) {
	c := servidorFalso(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{
			"code": "Ok",
			"routes": [{"distance": 12345.6, "duration": 789.1, "weight": 800}],
			"waypoints": [{"distance": 3.5}, {"distance": 7.25}]
		}`))
	})

	resp, err := c.Rota(context.Background(), geo.Point{Lat: -3.7, Lon: -38.5}, geo.Point{Lat: -3.8, Lon: -38.6})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Metros != 12345.6 || resp.Segundos != 789.1 {
		t.Errorf("resposta = %+v", resp)
	}
	if resp.Encaixe[0] != 3.5 || resp.Encaixe[1] != 7.25 {
		t.Errorf("encaixes = %v", resp.Encaixe)
	}
}

// Sem rota nao e falha do programa: e informacao sobre o mapa, e o relatorio
// conta separado.
func TestRotaSemCaminho(t *testing.T) {
	c := servidorFalso(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"code":"NoRoute","message":"Impossible route"}`))
	})

	_, err := c.Rota(context.Background(), geo.Point{}, geo.Point{})
	var semRota ErrSemRota
	if !errors.As(err, &semRota) {
		t.Fatalf("erro = %v, esperado ErrSemRota", err)
	}
	if semRota.Codigo != "NoRoute" {
		t.Errorf("codigo = %q", semRota.Codigo)
	}
}

func TestRotaRespostaEstranha(t *testing.T) {
	casos := []struct {
		nome  string
		corpo string
	}{
		{"json quebrado", `{"code": "Ok",`},
		{"sem codigo", `{"routes":[{"distance":1}]}`},
		{"ok sem rotas", `{"code":"Ok","routes":[]}`},
		{"vazio", ``},
		{"html de proxy", `<html><body>502</body></html>`},
	}

	for _, caso := range casos {
		c := servidorFalso(t, func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(caso.corpo))
		})
		if _, err := c.Rota(context.Background(), geo.Point{}, geo.Point{}); err == nil {
			t.Errorf("%s: deveria dar erro", caso.nome)
		}
	}
}

func TestRotaServidorFora(t *testing.T) {
	c := &Cliente{
		Base: "http://127.0.0.1:1", // porta reservada, ninguem escuta
		HTTP: &http.Client{Timeout: 2 * time.Second},
	}
	_, err := c.Rota(context.Background(), geo.Point{}, geo.Point{})
	if err == nil {
		t.Fatal("servidor fora deveria dar erro")
	}
	var semRota ErrSemRota
	if errors.As(err, &semRota) {
		t.Error("servidor fora nao e o mesmo que nao haver rota")
	}
}

func TestRotaRespeitaCancelamento(t *testing.T) {
	c := servidorFalso(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})

	ctx, cancelar := context.WithCancel(context.Background())
	cancelar()
	if _, err := c.Rota(ctx, geo.Point{}, geo.Point{}); err == nil {
		t.Error("contexto cancelado deveria dar erro")
	}
}
