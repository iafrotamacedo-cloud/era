package osm

import (
	"errors"
	"fmt"
	"io"
	"runtime"

	"github.com/iafrotamacedo-cloud/era/internal/protowire"
	"github.com/iafrotamacedo-cloud/era/maps/geo"
)

// recursosSuportados sao as capacidades do formato que este leitor entende.
//
// Um arquivo declara no cabecalho o que exige de quem for le-lo. Exigir algo
// fora desta lista e erro, nao aviso: ler um mapa pela metade em silencio
// vira rota errada, e rota errada em logistica vira caminhao no lugar errado.
//
// HistoricalInformation fica de fora de proposito. Arquivos com historico
// trazem versoes antigas e entidades apagadas junto das atuais; le-los como
// se fossem um retrato do presente produziria ruas que nao existem mais.
var recursosSuportados = map[string]bool{
	"OsmSchema-V0.6": true,
	"DenseNodes":     true,
}

// resultado e o que uma goroutine de decodificacao devolve: ou um cabecalho,
// ou um bloco de entidades, ou o erro que impediu os dois.
type resultado struct {
	bl  *bloco
	cab *Header
	err error
}

// Scan percorre um .osm.pbf e entrega as entidades ao handler.
//
// Nao carrega o arquivo na memoria: le bloco a bloco. Um extrato do Brasil
// tem cerca de 1,5 GB comprimido, e o pico de memoria de Scan e proporcional
// ao paralelismo, nao ao tamanho do arquivo.
//
// A decodificacao acontece em paralelo -- descomprimir e o gargalo, e os
// blocos sao independentes --, mas os callbacks sao chamados de uma goroutine
// so, na ordem do arquivo. O handler nao precisa se preocupar com
// concorrencia, e duas leituras do mesmo arquivo produzem a mesma sequencia.
//
// Um callback que devolve erro interrompe a leitura, e Scan devolve esse
// erro; ErrParar interrompe sem erro.
func Scan(r io.Reader, h Handler) error {
	return ScanN(r, h, runtime.GOMAXPROCS(0))
}

// ScanN e Scan com o paralelismo escolhido a mao. Util em teste e em maquina
// onde a leitura divide CPU com outra coisa.
func ScanN(r io.Reader, h Handler, paralelismo int) error {
	if paralelismo < 1 {
		paralelismo = 1
	}

	// Os blocos decodificados circulam num conjunto fixo em vez de serem
	// alocados a cada vez. Cada um carrega buffers grandes -- tabela de
	// strings, etiquetas, referencias, e a area de descompressao, que sozinha
	// pode ter dezenas de MB. Realoca-los milhares de vezes daria trabalho a
	// toa ao coletor de lixo.
	//
	// Um a mais que o numero de futuros em voo: sem essa folga o produtor
	// poderia ficar esperando um bloco livre que so seria devolvido depois
	// que ele mesmo enfileirasse o proximo futuro.
	const folga = 1
	livres := make(chan *bloco, paralelismo+folga)
	for i := 0; i < paralelismo+folga; i++ {
		livres <- &bloco{}
	}

	// Um canal de canais preserva a ordem do arquivo mesmo com a
	// decodificacao fora de ordem: o produtor enfileira uma promessa por
	// bloco, e o consumidor espera cada uma na sequencia em que foram
	// enfileiradas. A capacidade e o que limita quantos blocos ficam em voo.
	futuros := make(chan chan resultado, paralelismo)
	parar := make(chan struct{})

	go func() {
		defer close(futuros)
		for {
			bl, err := lerBlob(r)
			if errors.Is(err, io.EOF) {
				return
			}

			ch := make(chan resultado, 1)
			if err != nil {
				ch <- resultado{err: err}
				select {
				case futuros <- ch:
				case <-parar:
				}
				return
			}

			var b *bloco
			select {
			case b = <-livres:
			case <-parar:
				return
			}

			select {
			case futuros <- ch:
			case <-parar:
				return
			}

			go func(bl blob, b *bloco, ch chan resultado) {
				ch <- decodificar(bl, b, &h)
			}(bl, b, ch)
		}
	}()

	// Sair no meio exige desmontar: sem isto o produtor e as goroutines em
	// voo ficariam presos escrevendo em canais que ninguem mais le.
	desmontar := func(err error) error {
		close(parar)
		for ch := range futuros {
			<-ch
		}
		return err
	}

	viuCabecalho := false
	for ch := range futuros {
		res := <-ch
		if res.err != nil {
			return desmontar(res.err)
		}

		err := processar(res, &h, &viuCabecalho)

		// O bloco volta ao conjunto mesmo quando o handler falhou: quem sai
		// pela porta de erro tambem precisa deixar a casa arrumada.
		if res.bl != nil {
			livres <- res.bl
		}
		if err != nil {
			return desmontar(soltarParar(err))
		}
	}

	if !viuCabecalho {
		return fmt.Errorf("%w: arquivo sem bloco OSMHeader", ErrFormato)
	}
	return nil
}

// processar chama os callbacks para um resultado ja decodificado.
func processar(res resultado, h *Handler, viuCabecalho *bool) error {
	if res.cab != nil {
		*viuCabecalho = true
		if h.Header != nil {
			return h.Header(*res.cab)
		}
		return nil
	}
	return entregar(res.bl, h)
}

// soltarParar traduz ErrParar em sucesso.
func soltarParar(err error) error {
	if errors.Is(err, ErrParar) {
		return nil
	}
	return err
}

// decodificar transforma um blob cru no que ele contem: cabecalho ou
// entidades. Roda numa goroutine por bloco.
func decodificar(bl blob, b *bloco, h *Handler) resultado {
	res := resultado{bl: b}

	corpo, err := descomprimir(bl.corpo, &b.descomp)
	if err != nil {
		res.err = err
		return res
	}

	switch bl.tipo {
	case blocoCabecalho:
		cab, err := decodificarCabecalho(corpo)
		if err != nil {
			res.err = err
			return res
		}
		res.cab = &cab

	case blocoDados:
		if err := decodificarBloco(corpo, h, b); err != nil {
			res.err = err
			return res
		}

	default:
		// Tipo desconhecido e ignorado de proposito: o formato preve que
		// versoes futuras acrescentem tipos, e um leitor que quebra por isso
		// envelhece mal.
	}
	return res
}

// entregar chama os callbacks do handler para as entidades de um bloco.
func entregar(b *bloco, h *Handler) error {
	if b == nil {
		return nil
	}
	if h.Node != nil {
		for i := range b.nos {
			if err := h.Node(b.nos[i]); err != nil {
				return err
			}
		}
	}
	if h.Way != nil {
		for i := range b.vias {
			if err := h.Way(b.vias[i]); err != nil {
				return err
			}
		}
	}
	if h.Relation != nil {
		for i := range b.relacoes {
			if err := h.Relation(b.relacoes[i]); err != nil {
				return err
			}
		}
	}
	return nil
}

// decodificarCabecalho le o HeaderBlock e confere o que o arquivo exige.
func decodificarCabecalho(buf []byte) (Header, error) {
	var cab Header
	r := protowire.New(buf)

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return cab, fmt.Errorf("%w: cabecalho ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1: // bbox
			m, err := r.Message()
			if err != nil {
				return cab, fmt.Errorf("%w: retangulo do cabecalho ilegivel: %v", ErrFormato, err)
			}
			cab.BBox, err = decodificarBBox(m)
			if err != nil {
				return cab, err
			}
			cab.HasBBox = true
		case 4: // required_features
			s, err := r.String()
			if err != nil {
				return cab, fmt.Errorf("%w: cabecalho ilegivel: %v", ErrFormato, err)
			}
			cab.RequiredFeatures = append(cab.RequiredFeatures, s)
		case 5: // optional_features
			s, err := r.String()
			if err != nil {
				return cab, fmt.Errorf("%w: cabecalho ilegivel: %v", ErrFormato, err)
			}
			cab.OptionalFeatures = append(cab.OptionalFeatures, s)
		case 16:
			cab.WritingProgram, err = r.String()
		case 17:
			cab.Source, err = r.String()
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return cab, fmt.Errorf("%w: cabecalho ilegivel: %v", ErrFormato, err)
		}
	}

	for _, f := range cab.RequiredFeatures {
		if !recursosSuportados[f] {
			return cab, fmt.Errorf("%w: o arquivo exige %q, que este leitor nao implementa",
				ErrNaoSuportado, f)
		}
	}
	return cab, nil
}

// decodificarBBox le o retangulo do cabecalho.
//
// As coordenadas aqui vem em nanograus diretos, sem passar pela granularidade
// e pelos deslocamentos do PrimitiveBlock -- e uma mensagem a parte, com
// regra propria.
func decodificarBBox(r *protowire.Reader) (geo.Box, error) {
	var esq, dir, cima, baixo int64

	for !r.Done() {
		campo, typ, err := r.Tag()
		if err != nil {
			return geo.Box{}, fmt.Errorf("%w: retangulo ilegivel: %v", ErrFormato, err)
		}
		switch campo {
		case 1:
			esq, err = r.SInt64()
		case 2:
			dir, err = r.SInt64()
		case 3:
			cima, err = r.SInt64()
		case 4:
			baixo, err = r.SInt64()
		default:
			err = r.Skip(typ)
		}
		if err != nil {
			return geo.Box{}, fmt.Errorf("%w: retangulo ilegivel: %v", ErrFormato, err)
		}
	}

	return geo.Box{
		MinLat: 1e-9 * float64(baixo),
		MinLon: 1e-9 * float64(esq),
		MaxLat: 1e-9 * float64(cima),
		MaxLon: 1e-9 * float64(dir),
	}, nil
}
