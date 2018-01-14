// Copyright 2017 The Universality Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"hash/fnv"
	"image/color"
	"math"
	"math/rand"
	"os"
	"time"
	"unicode"

	"github.com/bugra/kmeans"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const (
	Size       = 1000
	vectorSize = 1024
	bufferSize = 17
)

// CircularBuffer is a circular buffer of size bufferSize
type CircularBuffer struct {
	Buffer          []string
	Index, Previous int
}

// NewCircularBuffer creates a new circular buffer of size bufferSize
func NewCircularBuffer() *CircularBuffer {
	return &CircularBuffer{
		Buffer: make([]string, bufferSize),
	}
}

// Push adds a new string to the end of the buffer
func (c *CircularBuffer) Push(a string) {
	c.Buffer[c.Index] = a
	c.Index, c.Previous = (c.Index+1)%bufferSize, c.Index
}

// Item returns the string at index relative to the beginning of the buffer
func (c *CircularBuffer) Item(index int) string {
	return c.Buffer[(c.Index+index)%bufferSize]
}

// GetPrevious gets the string just inserted into the buffer
func (c *CircularBuffer) GetPrevious() string {
	return c.Buffer[c.Previous]
}

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

func hash(a string) uint64 {
	h := fnv.New64()
	h.Write([]byte(a))
	return h.Sum64()
}

func BookAdjacencyMatrix(book string) {
	in, err := os.Open(book)
	if err != nil {
		panic(err)
	}

	count, word, reader, words, buffer, cache :=
		0, "", bufio.NewReader(in), make(map[string][]int64), NewCircularBuffer(), make(map[uint64][]int8)

	lookup := func(a string) []int8 {
		h := hash(a)
		transform, found := cache[h]
		if found {
			return transform
		}
		transform = make([]int8, vectorSize)
		rnd := rand.New(rand.NewSource(int64(h)))
		for i := range transform {
			// https://en.wikipedia.org/wiki/Random_projection#More_computationally_efficient_random_projections
			// make below distribution function of vector element index
			switch rnd.Intn(6) {
			case 0:
				transform[i] = 1
			case 1:
				transform[i] = -1
			}
		}
		cache[h] = transform
		return transform
	}

	for {
		r, _, err := reader.ReadRune()
		if err != nil {
			break
		}
		if unicode.IsLetter(r) || r == '\'' {
			word += string(unicode.ToLower(r))
		} else if word != "" {
			center := buffer.Item(bufferSize / 2)
			wordVector := words[center]
			if wordVector == nil {
				wordVector = make([]int64, vectorSize)
				words[center] = wordVector
			}

			last := buffer.Item(0)
			for i := 1; i < bufferSize; i++ {
				current := buffer.Item(i)
				if current == center {
					continue
				}
				transform := lookup(last + current)
				for i, t := range transform {
					wordVector[i] += int64(t)
				}
				last = current
			}

			count++
			buffer.Push(word)
			word = ""
		}
	}

	i, wordList, data := 0, make([]string, len(words)), make([][]float64, len(words))
	for word, vector := range words {
		wordList[i] = word
		a := make([]float64, vectorSize)
		for j := range a {
			a[j] = float64(vector[j])
		}
		data[i] = a
		i++
	}

	means, err := kmeans.Kmeans(data, 1000, kmeans.ManhattanDistance, 1)
	if err != nil {
		panic(err)
	}
	_ = means

	fmt.Printf("count=%v\n", count)
}

func main() {
	start := time.Now()
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "semicircle_law.png")
	if err != nil {
		panic(err)
	}
	fmt.Println(time.Now().Sub(start).String())

	BookAdjacencyMatrix("data/244-0.txt")
}
