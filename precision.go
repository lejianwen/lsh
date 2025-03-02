package lsh

const (
	PrecisionFloat32 string = "float32"
	PrecisionFloat64 string = "float64"
	PrecisionInt8    string = "int8"
	PrecisionInt16   string = "int16"
)

const (
	Int8Scale  float64 = 126
	Int16Scale float64 = 32766
)

// 计算缩放因子
// 归一化后向量取值范围是[-1,1]，因此最大值为 (1*s^2)*len, 得出 (1*s^2)*len<=max , s^2<=max/len, s<=sqrt(max/len)
// 这样算似乎不对，算sum时,单个乘积会转int64，应该是 s<sqrt(max)
// 如果算sum时,单个乘数转int64，应该是 s<max
/*func calcScale(to string) float64 {
	switch to {
	case "int8":
		return math.Floor(math.Sqrt(float64(math.MaxInt8)))
	case "int16":
		return math.Floor(math.Sqrt(float64(math.MaxInt16)))
	case "int32":
		return math.Floor(math.Sqrt(float64(math.MaxInt32)))
	default:
		return 1
	}
}*/

type PrecisionType interface {
	~float32 | ~float64 | ~int8 | ~int16
}
type PrecisionHandler interface {
	Type() string
	ConvertVector([]float64) interface{}
	DotProduct(interface{}, interface{}) float64
}

// 高精度处理器（float64）
type Float64Handler struct{}

func (h *Float64Handler) ConvertVector(vec []float64) interface{} {
	return vec
}

func (h *Float64Handler) DotProduct(a, b interface{}) float64 {
	return DotProduct(a.([]float64), b.([]float64))
}
func (h *Float64Handler) Type() string {
	return PrecisionFloat64
}

type Float32Handler struct{}

func (h *Float32Handler) ConvertVector(vec []float64) interface{} {
	return toFloat32Slice(vec)
}

func (h *Float32Handler) DotProduct(a, b interface{}) float64 {
	return DotProduct(a.([]float32), b.([]float32))
}
func (h *Float32Handler) Type() string {
	return PrecisionFloat32
}

type Int16Handler struct{}

func (h *Int16Handler) ConvertVector(vec []float64) interface{} {
	result := make([]int16, len(vec))
	for i, v := range vec {
		result[i] = int16(v * Int16Scale)
	}
	return result
}

func (h *Int16Handler) DotProduct(a, b interface{}) float64 {
	return DotProduct(a.([]int16), b.([]int16)) / Int16Scale / Int16Scale
}
func (h *Int16Handler) Type() string {
	return PrecisionInt16
}

type Int8Handler struct{}

func (h *Int8Handler) ConvertVector(vec []float64) interface{} {
	result := make([]int8, len(vec))
	for i, v := range vec {
		result[i] = int8(v * Int8Scale)
	}
	return result
}

func (h *Int8Handler) DotProduct(a, b interface{}) float64 {
	return DotProduct(a.([]int8), b.([]int8)) / Int8Scale / Int8Scale
}
func (h *Int8Handler) Type() string {
	return PrecisionInt8
}
func NewPrecisionHandler(precision string) PrecisionHandler {
	switch precision {
	case PrecisionFloat32:
		return &Float32Handler{}
	case PrecisionFloat64:
		return &Float64Handler{}
	case PrecisionInt8:
		return &Int8Handler{}
	case PrecisionInt16:
		return &Int16Handler{}
	default:
		return &Float64Handler{}
	}
}
