package dist

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Este arquivo guarda os dois formatos binarios do pacote: o do cache e o da
// calibracao.
//
// Sao escritos a mao, com cabecalho magico e numero de versao, em vez de
// encoding/gob. Gob seria menos linhas, mas amarra o formato em disco a forma
// exata das structs Go: renomear um campo quebra a leitura de arquivos
// antigos, e nada avisa. Um formato explicito custa mais para escrever e
// depois so muda quando alguem decide muda-lo -- e o byte de versao existe
// justamente para essa decisao ser tratavel.
//
// Tambem e ensaio: a Fase 5 precisa serializar um grafo rodoviario inteiro no
// formato .eramap, e e melhor errar aqui, com 24 bytes por registro.

const (
	magicCache       = "ERAMAPC"
	magicCalibration = "ERAMAPK"
	versaoAtual      = 1
)

// ErrFormato indica arquivo que nao e do tipo esperado ou esta corrompido.
var ErrFormato = errors.New("dist: formato de arquivo invalido")

// ErrVersao indica arquivo de uma versao que este codigo nao sabe ler.
var ErrVersao = errors.New("dist: versao de formato nao suportada")

func escreveCabecalho(w io.Writer, magic string) error {
	var b [8]byte
	copy(b[:7], magic)
	b[7] = versaoAtual
	_, err := w.Write(b[:])
	return err
}

func leCabecalho(r io.Reader, magic string) error {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return fmt.Errorf("%w: arquivo curto demais para ter cabecalho", ErrFormato)
		}
		return err
	}
	if string(b[:7]) != magic {
		return fmt.Errorf("%w: esperava %q, encontrei %q", ErrFormato, magic, string(b[:7]))
	}
	if b[7] != versaoAtual {
		return fmt.Errorf("%w: arquivo e versao %d, este codigo le a %d", ErrVersao, b[7], versaoAtual)
	}
	return nil
}

// tamanhoEntrada e o peso de um par no cache: quatro int32 de chave mais dois
// float32 de medicao.
const tamanhoEntrada = 4*4 + 2*4

// Save grava o cache.
//
// O pacote nao escolhe onde: recebe um io.Writer. Arquivo, memoria, rede,
// dentro de um zip -- e decisao de quem usa, como no resto da ERA.
func (c *Cache) Save(w io.Writer) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	bw := bufio.NewWriter(w)
	if err := escreveCabecalho(bw, magicCache); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, uint32(len(c.entries))); err != nil {
		return err
	}

	var b [tamanhoEntrada]byte
	for k, l := range c.entries {
		binary.LittleEndian.PutUint32(b[0:], uint32(k.deLat))
		binary.LittleEndian.PutUint32(b[4:], uint32(k.deLon))
		binary.LittleEndian.PutUint32(b[8:], uint32(k.aLat))
		binary.LittleEndian.PutUint32(b[12:], uint32(k.aLon))
		binary.LittleEndian.PutUint32(b[16:], math.Float32bits(float32(l.Meters)))
		binary.LittleEndian.PutUint32(b[20:], math.Float32bits(float32(l.Seconds)))
		if _, err := bw.Write(b[:]); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// LoadCache le um cache gravado por Save e o religa a um Distancer.
//
// O Distancer de tras nao e guardado no arquivo: qual motor responde as
// faltas e decisao de quem carrega, e pode mudar entre uma execucao e outra
// -- e exatamente isso que acontece quando a Fase 5 substituir o Estimator.
//
// Isso tem uma consequencia que precisa ser dita: um cache preenchido pelo
// Estimator e recarregado sob o motor de grafo devolve os numeros velhos, com
// erro de 10%, sem avisar. Ao trocar de motor, descarte o cache.
func LoadCache(r io.Reader, inner Distancer) (*Cache, error) {
	br := bufio.NewReader(r)
	if err := leCabecalho(br, magicCache); err != nil {
		return nil, err
	}

	var n uint32
	if err := binary.Read(br, binary.LittleEndian, &n); err != nil {
		return nil, fmt.Errorf("%w: nao consegui ler a contagem de entradas", ErrFormato)
	}

	c := &Cache{inner: inner, entries: make(map[chave]Leg, n)}
	var b [tamanhoEntrada]byte
	for i := uint32(0); i < n; i++ {
		if _, err := io.ReadFull(br, b[:]); err != nil {
			return nil, fmt.Errorf("%w: arquivo diz ter %d entradas, acabou na %d", ErrFormato, n, i)
		}
		k := chave{
			deLat: int32(binary.LittleEndian.Uint32(b[0:])),
			deLon: int32(binary.LittleEndian.Uint32(b[4:])),
			aLat:  int32(binary.LittleEndian.Uint32(b[8:])),
			aLon:  int32(binary.LittleEndian.Uint32(b[12:])),
		}
		c.entries[k] = Leg{
			Meters:  float64(math.Float32frombits(binary.LittleEndian.Uint32(b[16:]))),
			Seconds: float64(math.Float32frombits(binary.LittleEndian.Uint32(b[20:]))),
		}
	}
	return c, nil
}

// Save grava a calibracao.
func (c Calibration) Save(w io.Writer) error {
	bw := bufio.NewWriter(w)
	if err := escreveCabecalho(bw, magicCalibration); err != nil {
		return err
	}

	escreveF := func(v float64) error {
		return binary.Write(bw, binary.LittleEndian, math.Float64bits(v))
	}
	escreveU := func(v uint32) error {
		return binary.Write(bw, binary.LittleEndian, v)
	}

	if err := escreveU(uint32(len(c.Bands))); err != nil {
		return err
	}
	for _, b := range c.Bands {
		for _, v := range []float64{b.UpToMeters, b.Factor, b.SpeedKmh} {
			if err := escreveF(v); err != nil {
				return err
			}
		}
		if err := escreveU(uint32(b.Samples)); err != nil {
			return err
		}
	}

	for _, v := range []float64{c.Factor, c.SpeedKmh} {
		if err := escreveF(v); err != nil {
			return err
		}
	}
	if err := escreveU(uint32(c.Samples)); err != nil {
		return err
	}

	// A marca de "generica" viaja junto: um sistema que carrega uma
	// calibracao precisa poder avisar que ainda esta usando o andaime.
	var g byte
	if c.generic {
		g = 1
	}
	if err := bw.WriteByte(g); err != nil {
		return err
	}
	return bw.Flush()
}

// LoadCalibration le uma calibracao gravada por Save.
func LoadCalibration(r io.Reader) (Calibration, error) {
	br := bufio.NewReader(r)
	if err := leCabecalho(br, magicCalibration); err != nil {
		return Calibration{}, err
	}

	leF := func() (float64, error) {
		var bits uint64
		err := binary.Read(br, binary.LittleEndian, &bits)
		return math.Float64frombits(bits), err
	}
	leU := func() (uint32, error) {
		var v uint32
		err := binary.Read(br, binary.LittleEndian, &v)
		return v, err
	}

	curto := func(err error) error {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return fmt.Errorf("%w: arquivo terminou no meio da calibracao", ErrFormato)
		}
		return err
	}

	n, err := leU()
	if err != nil {
		return Calibration{}, curto(err)
	}

	var c Calibration
	for i := uint32(0); i < n; i++ {
		var b Band
		for _, campo := range []*float64{&b.UpToMeters, &b.Factor, &b.SpeedKmh} {
			v, err := leF()
			if err != nil {
				return Calibration{}, curto(err)
			}
			*campo = v
		}
		s, err := leU()
		if err != nil {
			return Calibration{}, curto(err)
		}
		b.Samples = int(s)
		c.Bands = append(c.Bands, b)
	}

	for _, campo := range []*float64{&c.Factor, &c.SpeedKmh} {
		v, err := leF()
		if err != nil {
			return Calibration{}, curto(err)
		}
		*campo = v
	}
	s, err := leU()
	if err != nil {
		return Calibration{}, curto(err)
	}
	c.Samples = int(s)

	g, err := br.ReadByte()
	if err != nil {
		return Calibration{}, curto(err)
	}
	c.generic = g == 1

	return c, nil
}
