package lsh

import (
	"math"
	"math/rand/v2"
)

// 计算点积
func DotProduct[T PrecisionType](vecA, vecB []T) float64 {
	sum := float64(0)
	for i := range vecA {
		sum += float64(vecA[i]) * float64(vecB[i])
	}
	return sum
}

func toFloat32Slice(slice []float64) []float32 {
	result := make([]float32, len(slice))
	for i, v := range slice {
		result[i] = float32(v)
	}
	return result
}

// 生成随机向量（范围：[-1, 1)）
func generateRandomVector(size int) []float64 {
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = rand.Float64()*2 - 1
	}
	return vector
}

// 生成单位向量
func randomUnitVector(dimension int) []float64 {
	vec := make([]float64, dimension)
	var norm float64
	for i := range vec {
		vec[i] = rand.Float64()*2 - 1 // 生成 [-1,1] 之间的随机数
		norm += vec[i] * vec[i]
	}
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

// L2Normalize 归一化
func L2Normalize(vec []float64) []float64 {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}
	sum = math.Sqrt(sum)
	for i := range vec {
		vec[i] /= sum
	}
	return vec
}
