// Copyright 2017 The Universality Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	Size = 1000
)

func TestEigen() {
	a := mat.NewDense(2, 2, []float64{2, 1, 1, 2})
	fmt.Println(mat.Formatted(a))
	eigen := &mat.Eigen{}
	ok := eigen.Factorize(a, true, true)
	if !ok {
		panic("failed to factorize matrix")
	}
	fmt.Println(mat.Formatted(eigen.LeftVectors()))
	fmt.Println(mat.Formatted(eigen.Vectors()))
	fmt.Println(eigen.Values(nil))
	v := mat.NewDense(2, 1, []float64{-0.7071067811865475, 0.7071067811865475})
	b := &mat.Dense{}
	b.Mul(a, v)
	fmt.Println(mat.Formatted(b))
}

func main() {
	data := make([]float64, Size*Size)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	a := mat.NewDense(Size, Size, data)
	b := mat.NewDense(Size, Size, nil)
	b.Add(a, a.T())
	a.Apply(func(i, j int, v float64) float64 {
		return v / 2
	}, b)
	eigen := &mat.Eigen{}
	ok := eigen.Factorize(a, false, false)
	if !ok {
		panic("failed to factorize matrix")
	}
	c := math.Sqrt(float64(Size) / 2)
	values := make(plotter.Values, Size)
	for i, j := range eigen.Values(nil) {
		values[i] = real(j) / c
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Semicircle Law"

	h, err := plotter.NewHist(values, 20)
	if err != nil {
		panic(err)
	}
	h.Normalize(1)
	p.Add(h)

	circle := plotter.NewFunction(func(a float64) float64 {
		return math.Sqrt(4-math.Pow(a, 2)) / (2 * math.Pi)
	})
	circle.Color = color.RGBA{R: 255, A: 255}
	circle.Width = vg.Points(2)
	p.Add(circle)

	if err := p.Save(8*vg.Inch, 8*vg.Inch, "semicircle_law.png"); err != nil {
		panic(err)
	}
}
