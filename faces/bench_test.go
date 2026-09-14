package faces

import (
	"testing"

	"github.com/iafrotamacedo-cloud/era/faces/detect"
)

func BenchmarkEmbed(b *testing.B) {
	eng := carregarEngine(b)
	img := imagemDeTeste(640, 640)

	if _, err := eng.Embed(img); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.Embed(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDetect(b *testing.B) {
	eng := carregarEngine(b)
	img := imagemDeTeste(640, 640)

	if _, err := eng.Detect(img); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.Detect(img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkOpen(b *testing.B) {
	yunet, sface := caminhosModelo()
	cfg := Config{
		YuNet:  yunet,
		SFace:  sface,
		Detect: detect.Options{Score: limiarDoTeste, NMS: nmsDoTeste},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eng, err := Open(cfg)
		if err != nil {
			b.Fatal(err)
		}
		_ = eng
	}
}
