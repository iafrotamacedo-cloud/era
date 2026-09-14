// Package geom fornece o vocabulario geometrico do ERA READ: ponto,
// poligono, homografia e ajuste de curva.
//
// Nada aqui toca em pixel -- e conta pura, sem imagem nenhuma envolvida.
// Quem toca em pixel e o imgproc; a ponte entre os dois e Remap, que usa a
// amostragem bilinear do imgproc para reamostrar uma imagem segundo um
// mapeamento geometrico definido aqui.
//
// E o material dos niveis de retificacao N1 e N2: N1 e uma homografia por
// quatro cantos, N2 e um ajuste de curva por linha de texto.
package geom
