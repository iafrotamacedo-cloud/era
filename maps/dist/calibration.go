package dist

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// Observation e uma viagem que ja aconteceu: de onde para onde, quanto o
// hodometro andou e quanto tempo levou.
//
// E a materia-prima da calibracao. Uma operacao logistica ja tem esse dado
// -- e a diferenca entre um fator de desvio chutado e um medido.
type Observation struct {
	From, To geo.Point
	Meters   float64 // o que o hodometro marcou
	Seconds  float64 // 0 quando o tempo nao foi registrado
}

// Band e o fator de desvio medido para uma faixa de distancia.
//
// Faixas existem porque o fator nao e constante, e nao e por um motivo
// geometrico: viagem curta paga desvio proporcionalmente maior. Ir a duas
// quadras pode custar seis por causa das maos unicas; atravessar o estado
// segue a rodovia, que ja foi tracada para ser curta. Um fator unico erra
// nas duas pontas ao mesmo tempo -- superestima o longo e subestima o curto.
type Band struct {
	UpToMeters float64 // limite superior da distancia em linha reta
	Factor     float64 // distancia rodoviaria = Factor x linha reta
	SpeedKmh   float64 // velocidade media efetiva, ja com paradas e transito
	Samples    int     // quantas observacoes sustentam estes numeros
}

// MinSamples e quantas observacoes uma faixa precisa para ser levada a serio.
//
// Abaixo disso a mediana da faixa e ruido, e a calibracao usa o valor global.
// Trinta e um numero convencional, escolhido por ser onde a mediana de uma
// amostra comeca a ficar estavel -- nao por nenhuma propriedade magica.
const MinSamples = 30

// DefaultBandLimits sao os limites de faixa usados quando nenhum outro e
// informado, em metros de linha reta.
//
// Correspondem a como uma operacao urbana realmente se divide: dentro do
// bairro, dentro da cidade, regiao metropolitana, e o resto.
var DefaultBandLimits = []float64{3_000, 15_000, 60_000}

// Calibration traduz distancia em linha reta em distancia e tempo
// rodoviarios.
//
// O valor zero nao serve para nada: use Calibrate com o historico da operacao,
// ou DefaultCalibration enquanto esse historico nao existe.
type Calibration struct {
	Bands []Band // ordenadas por UpToMeters; a ultima cobre ate o infinito

	// Valores globais, usados como rede: faixa com poucas amostras cai
	// para ca em vez de devolver ruido.
	Factor   float64
	SpeedKmh float64
	Samples  int

	generic bool // veio de DefaultCalibration, nao de medicao
}

// DefaultCalibration devolve uma calibracao generica, para quando ainda nao
// ha historico para medir.
//
// Os numeros vem da literatura de circuidade de malha viaria, que costuma
// situar o fator entre 1,2 e 1,4, e de velocidades urbanas efetivas tipicas.
// Nao foram medidos na operacao de ninguem.
//
// Isso e um andaime, nao um resultado. Rode Calibrate assim que houver
// hodometro: o proprio Report vai dizer quanto esta calibracao estava errada.
// Generic informa se o valor em uso ainda e este, para o sistema chamador
// poder avisar em vez de fingir precisao que nao tem.
func DefaultCalibration() Calibration {
	return Calibration{
		Bands: []Band{
			{UpToMeters: 3_000, Factor: 1.45, SpeedKmh: 18},
			{UpToMeters: 15_000, Factor: 1.35, SpeedKmh: 26},
			{UpToMeters: 60_000, Factor: 1.28, SpeedKmh: 45},
			{UpToMeters: math.Inf(1), Factor: 1.22, SpeedKmh: 65},
		},
		Factor:   1.30,
		SpeedKmh: 35,
		generic:  true,
	}
}

// Generic informa se a calibracao e o andaime generico de DefaultCalibration
// em vez de medicao da operacao.
func (c Calibration) Generic() bool { return c.generic }

// Valid informa se a calibracao tem os numeros minimos para ser usada.
func (c Calibration) Valid() bool {
	return c.Factor > 0 && c.SpeedKmh > 0
}

// For devolve o fator de desvio e a velocidade a aplicar numa distancia em
// linha reta.
//
// Faixa sem amostras suficientes cai para o valor global -- e melhor um
// numero medio honesto do que um numero especifico sustentado por tres
// viagens.
func (c Calibration) For(crowMeters float64) (factor, speedKmh float64) {
	for _, b := range c.Bands {
		if crowMeters <= b.UpToMeters {
			if b.Samples >= MinSamples || (c.generic && b.Factor > 0) {
				return b.Factor, b.SpeedKmh
			}
			break
		}
	}
	return c.Factor, c.SpeedKmh
}

// BandReport e a qualidade medida de uma faixa.
type BandReport struct {
	UpToMeters  float64
	Samples     int
	Factor      float64
	SpeedKmh    float64
	MedianError float64 // erro percentual mediano dentro da faixa
}

// Report conta o que a calibracao viu e quanto ela ainda erra.
//
// Existe porque um fator de desvio sem margem de erro e um chute com aparencia
// de medida. Quem for cotar frete com esses numeros precisa saber se metade
// das viagens erra 4% ou 25%.
type Report struct {
	Observations int // quantas entraram
	Used         int // quantas sobreviveram a validacao
	Discarded    int

	MedianError float64 // erro percentual mediano, sobre todas as usadas
	P90Error    float64 // nove em cada dez viagens erram menos que isto

	Bands []BandReport
}

// String resume o relatorio em uma linha legivel em log.
func (r Report) String() string {
	return fmt.Sprintf("calibracao: %d de %d observacoes usadas, erro mediano %.1f%%, p90 %.1f%%",
		r.Used, r.Observations, r.MedianError*100, r.P90Error*100)
}

// ErrSemObservacoes indica que nao sobrou dado utilizavel para calibrar.
var ErrSemObservacoes = errors.New("dist: nenhuma observacao valida para calibrar")

// Calibrate mede o fator de desvio e a velocidade a partir do historico da
// operacao.
//
// Os limites de faixa sao opcionais; sem eles usa DefaultBandLimits. Cada
// limite e a distancia em linha reta ate onde a faixa vale, em metros, em
// ordem crescente. Uma faixa final ate o infinito e acrescentada sozinha,
// a menos que o ultimo limite ja seja infinito -- passar math.Inf(1) sozinho
// e como pedir um fator unico para tudo.
//
// O fator de cada faixa e a *mediana* das razoes entre hodometro e linha
// reta, nao a media. Historico real de frota tem motorista que passou na
// oficina, entrega refeita e hodometro digitado errado -- a media persegue
// esses pontos, a mediana os ignora. E a razao mediana, e nao a razao entre
// as somas: o que se quer prever e uma viagem, nao o total da frota.
//
// O erro do Report e medido sobre as mesmas observacoes que geraram a
// calibracao, entao e otimista. Com um parametro por faixa e uma mediana o
// otimismo e pequeno, mas ele existe: trate os numeros como piso do erro
// real, nao como estimativa dele.
func Calibrate(obs []Observation, bandLimits ...float64) (Calibration, Report, error) {
	if len(bandLimits) == 0 {
		bandLimits = DefaultBandLimits
	}
	for i := 1; i < len(bandLimits); i++ {
		if bandLimits[i] <= bandLimits[i-1] {
			return Calibration{}, Report{}, fmt.Errorf(
				"dist: limites de faixa fora de ordem crescente: %v", bandLimits)
		}
	}

	rep := Report{Observations: len(obs)}

	type amostra struct {
		crow     float64
		ratio    float64
		speedKmh float64 // 0 quando o tempo nao foi registrado
		banda    int
	}

	// A faixa final ate o infinito e acrescentada sozinha, a menos que quem
	// chamou ja tenha informado uma -- caso contrario "uma faixa so, cobrindo
	// tudo" viraria duas, com a segunda vazia.
	limites := append([]float64(nil), bandLimits...)
	if !math.IsInf(limites[len(limites)-1], 1) {
		limites = append(limites, math.Inf(1))
	}
	amostras := make([]amostra, 0, len(obs))

	for _, o := range obs {
		if !valida(o) {
			continue
		}
		crow := geo.Haversine(o.From, o.To)
		ratio := o.Meters / crow
		if ratio < 0.95 || ratio > 10 {
			// Menor que a linha reta e impossivel; a folga de 5% cobre
			// ruido de GPS em viagem curta. Dez vezes e erro de digitacao,
			// nao rota.
			continue
		}

		a := amostra{crow: crow, ratio: ratio, banda: faixaDe(crow, limites)}
		if o.Seconds > 0 {
			a.speedKmh = o.Meters / o.Seconds * 3.6
			if a.speedKmh <= 0 || a.speedKmh > 130 {
				a.speedKmh = 0 // velocidade impossivel: descarta so o tempo
			}
		}
		amostras = append(amostras, a)
	}

	rep.Used = len(amostras)
	rep.Discarded = rep.Observations - rep.Used
	if rep.Used == 0 {
		return Calibration{}, rep, ErrSemObservacoes
	}

	cal := Calibration{Samples: rep.Used}

	// Valores globais primeiro: sao a rede em que as faixas fracas caem.
	todasRazoes := make([]float64, 0, len(amostras))
	todasVels := make([]float64, 0, len(amostras))
	for _, a := range amostras {
		todasRazoes = append(todasRazoes, a.ratio)
		if a.speedKmh > 0 {
			todasVels = append(todasVels, a.speedKmh)
		}
	}
	cal.Factor = mediana(todasRazoes)
	cal.SpeedKmh = mediana(todasVels)
	if cal.SpeedKmh == 0 {
		// Nenhuma observacao trouxe tempo. Sem inventar: a velocidade
		// generica entra e o Generic fica marcado para o chamador saber.
		cal.SpeedKmh = DefaultCalibration().SpeedKmh
	}

	// Depois cada faixa.
	for i, limite := range limites {
		var razoes, vels []float64
		for _, a := range amostras {
			if a.banda != i {
				continue
			}
			razoes = append(razoes, a.ratio)
			if a.speedKmh > 0 {
				vels = append(vels, a.speedKmh)
			}
		}

		b := Band{UpToMeters: limite, Samples: len(razoes)}
		if len(razoes) > 0 {
			b.Factor = mediana(razoes)
			b.SpeedKmh = mediana(vels)
		}
		if b.Factor == 0 {
			b.Factor = cal.Factor
		}
		if b.SpeedKmh == 0 {
			b.SpeedKmh = cal.SpeedKmh
		}
		cal.Bands = append(cal.Bands, b)
	}

	// Qualidade: quanto a calibracao recem-nascida erra sobre o proprio dado.
	erros := make([]float64, 0, len(amostras))
	errosPorFaixa := make([][]float64, len(limites))
	for _, a := range amostras {
		f, _ := cal.For(a.crow)
		e := math.Abs(f-a.ratio) / a.ratio
		erros = append(erros, e)
		errosPorFaixa[a.banda] = append(errosPorFaixa[a.banda], e)
	}
	rep.MedianError = mediana(erros)
	rep.P90Error = percentil(erros, 0.90)

	for i, b := range cal.Bands {
		rep.Bands = append(rep.Bands, BandReport{
			UpToMeters:  b.UpToMeters,
			Samples:     b.Samples,
			Factor:      b.Factor,
			SpeedKmh:    b.SpeedKmh,
			MedianError: mediana(errosPorFaixa[i]),
		})
	}

	return cal, rep, nil
}

// valida descarta a observacao que nao pode gerar razao nenhuma.
func valida(o Observation) bool {
	if !o.From.Valid() || !o.To.Valid() {
		return false
	}
	if o.Meters <= 0 || math.IsNaN(o.Meters) || math.IsInf(o.Meters, 0) {
		return false
	}
	// Pontos coincidentes nao tem razao definida: dividiria por zero.
	return geo.Haversine(o.From, o.To) >= 1
}

func faixaDe(crowMeters float64, limites []float64) int {
	for i, l := range limites {
		if crowMeters <= l {
			return i
		}
	}
	return len(limites) - 1
}

// mediana devolve 0 para fatia vazia -- e o sinal de "nao ha o que dizer",
// tratado por quem chama.
func mediana(v []float64) float64 { return percentil(v, 0.5) }

func percentil(v []float64, p float64) float64 {
	if len(v) == 0 {
		return 0
	}
	ordenado := append([]float64(nil), v...)
	sort.Float64s(ordenado)

	if len(ordenado) == 1 {
		return ordenado[0]
	}
	// Interpolacao linear entre os dois vizinhos, para que a mediana de um
	// numero par de amostras seja a media dos dois do meio.
	pos := p * float64(len(ordenado)-1)
	baixo := int(math.Floor(pos))
	alto := int(math.Ceil(pos))
	if baixo == alto {
		return ordenado[baixo]
	}
	peso := pos - float64(baixo)
	return ordenado[baixo]*(1-peso) + ordenado[alto]*peso
}
