package dist

import (
	"context"
	"errors"
	"runtime"
	"sync"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// ErrCalibracaoInvalida indica que o Estimator foi construido sem numeros
// utilizaveis.
var ErrCalibracaoInvalida = errors.New("dist: calibracao sem fator ou velocidade")

// Estimator mede distancia rodoviaria multiplicando a linha reta pelo fator
// de desvio da calibracao.
//
// E a implementacao barata do Distancer: dezenas de nanossegundos por par,
// nenhum dado de mapa, nenhum pre-processamento. Em troca erra na casa de
// 10% -- quanto exatamente, o Report da calibracao diz.
//
// Serve bem para: agrupar clientes por regiao, decidir raio de atendimento,
// escolher qual deposito atende qual entrega, e cortar candidatos antes de
// gastar uma consulta cara. Nao serve para cotar frete nem prometer horario
// de entrega.
//
// Estimator e seguro para uso concorrente: nao guarda estado mutavel.
type Estimator struct {
	cal Calibration
}

// NewEstimator constroi o estimador a partir de uma calibracao.
func NewEstimator(cal Calibration) (*Estimator, error) {
	if !cal.Valid() {
		return nil, ErrCalibracaoInvalida
	}
	return &Estimator{cal: cal}, nil
}

// Calibration devolve a calibracao em uso, para quem precisar saber se ainda
// e a generica.
func (e *Estimator) Calibration() Calibration { return e.cal }

// Distance mede um par.
//
// O erro devolvido e sempre nil: uma estimativa geometrica nao tem como
// "nao achar caminho". A assinatura carrega o erro porque a interface e
// compartilhada com implementacoes que tem -- um ponto numa ilha, um trecho
// desconectado do grafo.
func (e *Estimator) Distance(ctx context.Context, from, to geo.Point) (Leg, error) {
	if err := ctx.Err(); err != nil {
		return Leg{}, err
	}
	return e.leg(from, to), nil
}

func (e *Estimator) leg(from, to geo.Point) Leg {
	crow := geo.Haversine(from, to)
	fator, velKmh := e.cal.For(crow)

	metros := crow * fator
	return Leg{
		Meters:  metros,
		Seconds: metros / (velKmh / 3.6),
	}
}

// Matrix mede todos os pares, dividindo as linhas entre goroutines.
//
// Usa Haversine, e nao o geo.Ruler, que seria umas nove vezes mais rapido. O
// motivo nao e desempenho: e que o mesmo par tem de dar o mesmo numero seja
// perguntado por Distance ou por Matrix. Uma matriz que discorda da consulta
// individual gera bug que ninguem acha. E o erro do Ruler (0,3% num estado)
// e irrelevante perto do erro do fator de desvio (uns 10%) -- otimizar a
// parcela pequena de um erro grande nao muda nada.
func (e *Estimator) Matrix(ctx context.Context, origins, destinations []geo.Point) (*Matrix, error) {
	m := NewMatrix(origins, destinations)
	if len(origins) == 0 || len(destinations) == 0 {
		return m, nil
	}

	trabalhadores := runtime.GOMAXPROCS(0)
	if trabalhadores > len(origins) {
		trabalhadores = len(origins)
	}

	var (
		wg       sync.WaitGroup
		proxima  = make(chan int)
		cancelou = make(chan struct{})
		umaVez   sync.Once
	)

	// Distribuicao por linha: cada goroutine escreve so nas celulas da linha
	// que pegou, entao nao ha duas escrevendo no mesmo lugar. E uma
	// afirmacao verificada pelo CI com -race, nao assumida.
	for w := 0; w < trabalhadores; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range proxima {
				if ctx.Err() != nil {
					umaVez.Do(func() { close(cancelou) })
					return
				}
				for j, destino := range destinations {
					m.Set(i, j, e.leg(origins[i], destino))
				}
			}
		}()
	}

	for i := range origins {
		select {
		case proxima <- i:
		case <-cancelou:
			close(proxima)
			wg.Wait()
			return nil, ctx.Err()
		}
	}
	close(proxima)
	wg.Wait()

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return m, nil
}
