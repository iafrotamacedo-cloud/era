package ch

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/iafrotamacedo-cloud/era/maps/geo"
	"github.com/iafrotamacedo-cloud/era/maps/graph"
)

// O formato .eramap: uma hierarquia pronta, gravada.
//
// Existe porque o preparo e caro e cresce mais que linearmente -- 1,26 s para
// 6.400 cruzamentos extrapola para dezenas de minutos num estado. Aceitavel
// uma vez; inaceitavel a cada partida do processo. Sem este arquivo, o motor
// nao sobe rapido, e um motor que demora dezenas de minutos para subir nao e
// usado.
//
// Escrito a mao, com cabecalho magico e byte de versao, como o formato do
// dist. Nao e gob: gob amarra o arquivo em disco a forma exata das structs
// Go, e renomear um campo passaria a quebrar a leitura de arquivos antigos
// sem que nada avisasse.
//
// O arquivo e autossuficiente: carrega-lo nao exige o .osm.pbf que o gerou.
// E o ponto inteiro.
//
// Disposicao:
//
//	cabecalho   "ERAMAPH" + versao + metrica            9 bytes
//	contagens   nos, arcos, arestas originais, atalhos  16 bytes
//	preparo     duracao em nanossegundos                8 bytes
//	nos         lat, lon, osmID, posicao                28 bytes cada
//	offsets     inicio dos arcos de cada no             4 bytes x (nos+1)
//	arcos       para, metros, segundos, meio, bandeiras 17 bytes cada
//
// Coordenada em float64, e nao em inteiro de nanograus, que economizaria
// metade. A precisao do OpenStreetMap caberia em int32 e a conversao seria
// exata para dado vindo de .osm.pbf -- mas so para esse. Um grafo montado de
// outra fonte perderia precisao em silencio, e a economia num estado e de uns
// poucos MB.

const (
	magicMapa   = "ERAMAPH"
	versaoAtual = 1

	tamanhoNo   = 8 + 8 + 8 + 4
	tamanhoArco = 4 + 4 + 4 + 4 + 1
)

// ErrFormato indica arquivo que nao e um .eramap valido, ou esta corrompido.
var ErrFormato = errors.New("ch: formato de arquivo invalido")

// ErrVersao indica arquivo de uma versao que este codigo nao sabe ler.
var ErrVersao = errors.New("ch: versao de formato nao suportada")

// SaveFile grava a hierarquia num arquivo.
func (c *CH) SaveFile(caminho string) error {
	f, err := os.Create(caminho)
	if err != nil {
		return err
	}
	if err := c.Save(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Save grava a hierarquia.
//
// Onde guardar e decisao de quem usa -- arquivo, memoria, dentro de um zip --,
// como no resto da ERA.
func (c *CH) Save(w io.Writer) error {
	bw := bufio.NewWriterSize(w, 1<<20)

	var cab [9]byte
	copy(cab[:7], magicMapa)
	cab[7] = versaoAtual
	cab[8] = byte(c.m)
	if _, err := bw.Write(cab[:]); err != nil {
		return err
	}

	var conta [24]byte
	binary.LittleEndian.PutUint32(conta[0:], uint32(len(c.pontos)))
	binary.LittleEndian.PutUint32(conta[4:], uint32(len(c.arcos)))
	binary.LittleEndian.PutUint32(conta[8:], uint32(c.stats.ArestasOriginais))
	binary.LittleEndian.PutUint32(conta[12:], uint32(c.stats.Atalhos))
	binary.LittleEndian.PutUint64(conta[16:], uint64(c.stats.Duracao))
	if _, err := bw.Write(conta[:]); err != nil {
		return err
	}

	var no [tamanhoNo]byte
	for i := range c.pontos {
		binary.LittleEndian.PutUint64(no[0:], math.Float64bits(c.pontos[i].Lat))
		binary.LittleEndian.PutUint64(no[8:], math.Float64bits(c.pontos[i].Lon))
		binary.LittleEndian.PutUint64(no[16:], uint64(c.osmIDs[i]))
		binary.LittleEndian.PutUint32(no[24:], uint32(c.posicao[i]))
		if _, err := bw.Write(no[:]); err != nil {
			return err
		}
	}

	var quatro [4]byte
	for _, o := range c.offsets {
		binary.LittleEndian.PutUint32(quatro[:], o)
		if _, err := bw.Write(quatro[:]); err != nil {
			return err
		}
	}

	var a [tamanhoArco]byte
	for i := range c.arcos {
		binary.LittleEndian.PutUint32(a[0:], uint32(c.arcos[i].Para))
		binary.LittleEndian.PutUint32(a[4:], math.Float32bits(c.arcos[i].Metros))
		binary.LittleEndian.PutUint32(a[8:], math.Float32bits(c.arcos[i].Segundos))
		binary.LittleEndian.PutUint32(a[12:], uint32(c.arcos[i].Meio))
		a[16] = 0
		if c.frente[i] {
			a[16] |= 1
		}
		if c.tras[i] {
			a[16] |= 2
		}
		if _, err := bw.Write(a[:]); err != nil {
			return err
		}
	}

	return bw.Flush()
}

// LoadFile le uma hierarquia de um arquivo.
func LoadFile(caminho string) (*CH, error) {
	f, err := os.Open(caminho)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f)
}

// Load le uma hierarquia gravada por Save.
//
// Confere as invariantes antes de devolver, e nao so os tamanhos. Um arquivo
// corrompido de um jeito que passasse pela checagem de tamanho poderia
// produzir uma consulta em laco infinito ou um indice fora da faixa no meio de
// uma rota -- muito pior de diagnosticar do que uma recusa na abertura.
func Load(r io.Reader) (*CH, error) {
	br := bufio.NewReaderSize(r, 1<<20)

	var cab [9]byte
	if _, err := io.ReadFull(br, cab[:]); err != nil {
		return nil, fmt.Errorf("%w: arquivo curto demais para ter cabecalho", ErrFormato)
	}
	if string(cab[:7]) != magicMapa {
		return nil, fmt.Errorf("%w: esperava %q, encontrei %q", ErrFormato, magicMapa, string(cab[:7]))
	}
	if cab[7] != versaoAtual {
		return nil, fmt.Errorf("%w: arquivo e versao %d, este codigo le a %d", ErrVersao, cab[7], versaoAtual)
	}
	if cab[8] > byte(graph.Time) {
		return nil, fmt.Errorf("%w: metrica %d desconhecida", ErrFormato, cab[8])
	}

	var conta [24]byte
	if _, err := io.ReadFull(br, conta[:]); err != nil {
		return nil, fmt.Errorf("%w: arquivo sem as contagens", ErrFormato)
	}
	nNos := int(binary.LittleEndian.Uint32(conta[0:]))
	nArcos := int(binary.LittleEndian.Uint32(conta[4:]))

	c := &CH{
		m:       graph.Metric(cab[8]),
		pontos:  make([]geo.Point, nNos),
		osmIDs:  make([]int64, nNos),
		posicao: make([]int32, nNos),
		offsets: make([]uint32, nNos+1),
		arcos:   make([]arco, nArcos),
		frente:  make([]bool, nArcos),
		tras:    make([]bool, nArcos),
		stats: Stats{
			Nos:              nNos,
			ArcosSubindo:     nArcos,
			ArestasOriginais: int(binary.LittleEndian.Uint32(conta[8:])),
			Atalhos:          int(binary.LittleEndian.Uint32(conta[12:])),
			Duracao:          time.Duration(binary.LittleEndian.Uint64(conta[16:])),
		},
	}

	var no [tamanhoNo]byte
	for i := 0; i < nNos; i++ {
		if _, err := io.ReadFull(br, no[:]); err != nil {
			return nil, fmt.Errorf("%w: arquivo diz ter %d nos, acabou no %d", ErrFormato, nNos, i)
		}
		c.pontos[i] = geo.Point{
			Lat: math.Float64frombits(binary.LittleEndian.Uint64(no[0:])),
			Lon: math.Float64frombits(binary.LittleEndian.Uint64(no[8:])),
		}
		c.osmIDs[i] = int64(binary.LittleEndian.Uint64(no[16:]))
		c.posicao[i] = int32(binary.LittleEndian.Uint32(no[24:]))
	}

	var quatro [4]byte
	for i := 0; i <= nNos; i++ {
		if _, err := io.ReadFull(br, quatro[:]); err != nil {
			return nil, fmt.Errorf("%w: arquivo acabou no meio dos indices", ErrFormato)
		}
		c.offsets[i] = binary.LittleEndian.Uint32(quatro[:])
	}

	var a [tamanhoArco]byte
	for i := 0; i < nArcos; i++ {
		if _, err := io.ReadFull(br, a[:]); err != nil {
			return nil, fmt.Errorf("%w: arquivo diz ter %d arcos, acabou no %d", ErrFormato, nArcos, i)
		}
		c.arcos[i] = arco{
			Para:     graph.NodeID(int32(binary.LittleEndian.Uint32(a[0:]))),
			Metros:   math.Float32frombits(binary.LittleEndian.Uint32(a[4:])),
			Segundos: math.Float32frombits(binary.LittleEndian.Uint32(a[8:])),
			Meio:     graph.NodeID(int32(binary.LittleEndian.Uint32(a[12:]))),
		}
		c.frente[i] = a[16]&1 != 0
		c.tras[i] = a[16]&2 != 0
	}

	if err := c.conferirInvariantes(); err != nil {
		return nil, err
	}
	return c, nil
}

// conferirInvariantes recusa arquivo que passaria pelos tamanhos mas quebraria
// a consulta.
func (c *CH) conferirInvariantes() error {
	n := len(c.pontos)

	// As posicoes tem de ser uma permutacao de 0 a n-1. Duas iguais fariam a
	// regra de "so sobe" perder o sentido, e a busca poderia andar em circulo.
	vistas := make([]bool, n)
	for i, p := range c.posicao {
		if p < 0 || int(p) >= n {
			return fmt.Errorf("%w: no %d tem posicao %d, fora de 0..%d", ErrFormato, i, p, n-1)
		}
		if vistas[p] {
			return fmt.Errorf("%w: posicao %d aparece em mais de um no", ErrFormato, p)
		}
		vistas[p] = true
	}

	if c.offsets[n] != uint32(len(c.arcos)) {
		return fmt.Errorf("%w: o ultimo indice e %d e ha %d arcos", ErrFormato, c.offsets[n], len(c.arcos))
	}
	for i := 1; i <= n; i++ {
		if c.offsets[i] < c.offsets[i-1] {
			return fmt.Errorf("%w: indices fora de ordem em %d", ErrFormato, i)
		}
	}

	for v := 0; v < n; v++ {
		for i := c.offsets[v]; i < c.offsets[v+1]; i++ {
			a := c.arcos[i]
			if a.Para < 0 || int(a.Para) >= n {
				return fmt.Errorf("%w: arco do no %d aponta para %d", ErrFormato, v, a.Para)
			}
			if a.Meio != graph.NoNode && (a.Meio < 0 || int(a.Meio) >= n) {
				return fmt.Errorf("%w: arco do no %d tem meio %d", ErrFormato, v, a.Meio)
			}
			// Todo arco tem de subir. Um que descesse nunca seria usado, e a
			// presenca dele denuncia arquivo corrompido.
			if c.posicao[a.Para] <= c.posicao[v] {
				return fmt.Errorf("%w: arco de %d para %d nao sobe a hierarquia", ErrFormato, v, a.Para)
			}
		}
	}
	return nil
}
