package lsh

import (
	"fmt"
	"github.com/bytedance/sonic"
	"os"
	"sort"
	"strings"
	"sync"
)

var json = sonic.ConfigStd

// LSH 结构体
type LSH struct {
	numTables  int                   // 哈希表数量 ： 越大，召回率高，但内存占用大。越 小，查询速度快，但召回率低。一般取 10~100 之间
	numHashes  int                   // 每个表的哈希函数数量 ： 越大，召回率低，但误检率低（更精确）。 越小，召回率高，但误检率高（可能返回不相关的向量）。一般取 4~16 之间
	vectorSize int                   // 向量维度
	randomVecs [][][]float64         // 随机向量集合 [table][hash][dim]
	hashTables []map[string][]string // 哈希表存储 [table][hash]
	tableLocks []sync.RWMutex        // 每个哈希表的读写锁
	useCache   bool                  // 是否使用缓存 如果不用缓存计算点积，基本得不到想要的结果
	cacheMap   sync.Map              // 缓存
	filePath   string                // 保存文件路径
	//精度处理
	precisionHandler PrecisionHandler
}
type scoreItem struct {
	Id    string
	Score float64
}

// 初始化 LSH
func NewLSH(numTables, numHashes, vectorSize int) *LSH {
	// 生成随机向量（三维数组）
	randomVecs := make([][][]float64, numTables)
	for t := 0; t < numTables; t++ {
		randomVecs[t] = make([][]float64, numHashes)
		for h := 0; h < numHashes; h++ {
			randomVecs[t][h] = randomUnitVector(vectorSize)
		}
	}

	// 初始化哈希表和锁
	hashTables := make([]map[string][]string, numTables)
	tableLocks := make([]sync.RWMutex, numTables)
	for t := 0; t < numTables; t++ {
		hashTables[t] = make(map[string][]string)
	}

	return &LSH{
		numTables:        numTables,
		numHashes:        numHashes,
		vectorSize:       vectorSize,
		randomVecs:       randomVecs,
		hashTables:       hashTables,
		tableLocks:       tableLocks,
		useCache:         true,
		cacheMap:         sync.Map{},
		precisionHandler: NewPrecisionHandler(PrecisionFloat64),
	}
}

func NewNoCacheLSH(numTables, numHashes, vectorSize int) *LSH {
	l := NewLSH(numTables, numHashes, vectorSize)
	l.useCache = false
	return l
}

func (l *LSH) DisableCache() {
	l.useCache = false
}
func (l *LSH) EnableCache() {
	l.useCache = true
}
func (l *LSH) SetFilePath(filePath string) {
	l.filePath = strings.TrimRight(filePath, "/")
}
func (l *LSH) SetPrecisionHandler(precisionHandler PrecisionHandler) {
	l.precisionHandler = precisionHandler
}

// 计算单个哈希表的哈希值（二进制字符串）
func (l *LSH) computeHash(vector []float64, tableIdx int) string {
	var sb strings.Builder
	sb.Grow(l.numHashes) // 预分配空间

	for h := 0; h < l.numHashes; h++ {
		dot := DotProduct(vector, l.randomVecs[tableIdx][h])
		if dot >= 0 {
			sb.WriteByte('1')
		} else {
			sb.WriteByte('0')
		}
	}
	return sb.String()
}

func (l *LSH) addCacheItem(id string, vec []float64) {
	item := l.precisionHandler.ConvertVector(vec)
	l.cacheMap.Store(id, item)
}

func (l *LSH) loadCacheItem(id string) (any, bool) {
	return l.cacheMap.Load(id)
}

// 添加向量
func (l *LSH) AddVector(id string, vector []float64) {
	if l.useCache {
		l.addCacheItem(id, vector)
	}

	var wg sync.WaitGroup
	wg.Add(l.numTables)

	// 并行处理所有哈希表
	for t := 0; t < l.numTables; t++ {
		go func(tableIdx int) {
			defer wg.Done()

			// 计算哈希值
			hash := l.computeHash(vector, tableIdx)

			// 获取写锁
			l.tableLocks[tableIdx].Lock()
			defer l.tableLocks[tableIdx].Unlock()

			// 更新哈希表
			l.hashTables[tableIdx][hash] = append(l.hashTables[tableIdx][hash], id)
		}(t)
	}
	wg.Wait()
}

// 批量添加向量
func (l *LSH) AddVectors(vectors map[string][]float64) {
	var wg sync.WaitGroup
	wg.Add(len(vectors))
	for id, vec := range vectors {
		go func(id string, vec []float64) {
			defer wg.Done()
			l.AddVector(id, vec)
		}(id, vec)
	}
	wg.Wait()
}

/*func _getSimilarity[T PrecisionType](vector []float64, candidates *sync.Map, l *LSH, strategy DotProductStrategy[T], k int) []string {
	similarities := make([]scoreItem, 0, k)
	vLow := strategy.ConvertVector(vector)
	candidates.Range(func(key, value any) bool {
		id := key.(string)
		v, ok := l.loadCacheItem(id)
		if ok {
			s := strategy.Calculate(vLow, v.([]T))
			similarities = append(similarities, scoreItem{Id: id, Score: s})
		}
		return true
	})
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].Score > similarities[j].Score
	})
	if len(similarities) > k {
		similarities = similarities[:k]
	}
	result := make([]string, 0, k)
	for _, v := range similarities {
		result = append(result, v.Id)
	}
	return result
}*/

// 查询近似最近邻
func (l *LSH) Query(vector []float64, k int) []string {
	var candidates sync.Map
	var wg sync.WaitGroup
	wg.Add(l.numTables)

	// 并行查询所有哈希表
	for t := 0; t < l.numTables; t++ {
		go func(tableIdx int) {
			defer wg.Done()

			// 计算哈希值
			hash := l.computeHash(vector, tableIdx)
			// 获取读锁
			l.tableLocks[tableIdx].RLock()
			bucket, exists := l.hashTables[tableIdx][hash]
			l.tableLocks[tableIdx].RUnlock()

			if exists {
				for _, id := range bucket {
					candidates.Store(id, struct{}{})
				}
			}
		}(t)
	}
	wg.Wait()

	if l.useCache {
		vectorLow := l.precisionHandler.ConvertVector(vector)
		sortedSimilarities := make([]scoreItem, 0, k)
		candidates.Range(func(key, value any) bool {
			id := key.(string)
			v, ok := l.loadCacheItem(id)
			if ok {
				s := l.precisionHandler.DotProduct(vectorLow, v)
				sortedSimilarities = append(sortedSimilarities, scoreItem{Id: id, Score: s})
			}
			return true
		})

		sort.Slice(sortedSimilarities, func(i, j int) bool {
			return sortedSimilarities[i].Score > sortedSimilarities[j].Score
		})
		if len(sortedSimilarities) > k {
			sortedSimilarities = sortedSimilarities[:k]
		}
		result := make([]string, 0, k)
		for _, v := range sortedSimilarities {
			result = append(result, v.Id)
		}
		return result

	} else {
		result := make([]string, 0, k)
		candidates.Range(func(key, value any) bool {
			id := key.(string)
			result = append(result, id)
			return len(result) < k
		})
		return result
	}
}

// 批量查询
func (l *LSH) BatchQuery(vectors map[string][]float64, k int) map[string][]string {
	results := make(map[string][]string)
	var wg sync.WaitGroup
	wg.Add(len(vectors))

	var mu sync.Mutex

	for id, vec := range vectors {
		go func(id string, vec []float64) {
			defer wg.Done()
			res := l.Query(vec, k)

			mu.Lock()
			results[id] = res
			mu.Unlock()
		}(id, vec)
	}
	wg.Wait()
	return results
}

// 评估 LSH 的召回率和精度
func (l *LSH) EvaluateLSH(testQueries map[string][]float64, groundTruth map[string][]string, k int) (float64, float64, float64) {
	totalRecall, totalPrecision := 0.0, 0.0
	numQueries := len(testQueries)

	results := l.BatchQuery(testQueries, k)

	for id, retrieved := range results {
		trueNeighbors, exists := groundTruth[id]
		if !exists {
			continue
		}

		// 计算召回率 Recall = |正确召回的数量| / |真实近邻数量|
		truePositives := 0
		for _, r := range retrieved {
			for _, t := range trueNeighbors {
				if r == t {
					truePositives++
					break
				}
			}
		}
		recall := float64(truePositives) / float64(len(trueNeighbors))

		// 计算精度 Precision = |正确召回的数量| / |召回数量 k|
		precision := float64(truePositives) / float64(len(retrieved))

		totalRecall += recall
		totalPrecision += precision
	}
	f1 := 2 * totalRecall * totalPrecision / (totalRecall + totalPrecision)
	// 计算平均值
	return totalRecall / float64(numQueries), totalPrecision / float64(numQueries), f1
}

// 将hashTable保存到文件
func (l *LSH) saveTables() error {
	tp := fmt.Sprintf("%v/tables_%d_%d_%d.txt", l.filePath, l.numTables, l.numHashes, l.vectorSize)
	f, err := os.Create(tp)
	if err != nil {
		return err
	}
	defer f.Close()
	//转成json
	encoder := json.NewEncoder(f)
	for _, hashTable := range l.hashTables {
		if err := encoder.Encode(hashTable); err != nil {
			return err
		}
	}
	return nil
}
func (l *LSH) loadTables() error {
	tp := fmt.Sprintf("%v/tables_%d_%d_%d.txt", l.filePath, l.numTables, l.numHashes, l.vectorSize)
	f, err := os.Open(tp)
	if err != nil {
		return err
	}
	defer f.Close()
	//读取json
	var hashTables []map[string][]string
	var hashTable map[string][]string
	decoder := json.NewDecoder(f)
	for decoder.More() {
		if err := decoder.Decode(&hashTable); err != nil {
			return err
		}
		hashTables = append(hashTables, hashTable)
		hashTable = map[string][]string{}
	}
	l.hashTables = hashTables
	return nil
}

// SaveRandomVecs 保存随机向量
func (l *LSH) SaveRandomVecs() error {
	tp := fmt.Sprintf("%v/randomVecs_%d_%d_%d.txt", l.filePath, l.numTables, l.numHashes, l.vectorSize)
	f, err := os.Create(tp)
	if err != nil {
		return err
	}
	defer f.Close()
	//转成json
	jsonStr, err := json.Marshal(l.randomVecs)
	if err != nil {
		return err
	}
	_, err = f.Write(jsonStr)
	if err != nil {
		return err
	}
	return nil
}
func (l *LSH) LoadRandomVecs() error {
	tp := fmt.Sprintf("%v/randomVecs_%d_%d_%d.txt", l.filePath, l.numTables, l.numHashes, l.vectorSize)
	f, err := os.Open(tp)
	if err != nil {
		return err
	}
	defer f.Close()
	//读取json
	var randomVecs [][][]float64
	err = json.NewDecoder(f).Decode(&randomVecs)
	if err != nil {
		return err
	}
	l.randomVecs = randomVecs
	return nil
}

type kv[T int8 | int16 | float32 | float64] struct {
	K string `json:"k"`
	V []T    `json:"v"`
}

func _saveCacheMap[T int8 | int16 | float32 | float64](l *LSH) error {
	tp := fmt.Sprintf("%v/cacheMap.txt", l.filePath)
	f, err := os.Create(tp)
	if err != nil {
		return err
	}
	defer f.Close()

	item := kv[T]{
		K: "",
		V: nil,
	}
	// 创建 JSON 编码器
	encoder := json.NewEncoder(f)
	// 逐个写入 key-value
	l.cacheMap.Range(func(key, value any) bool {
		item = kv[T]{
			K: key.(string),
			V: value.([]T),
		}
		if err := encoder.Encode(item); err != nil {
			return false
		}
		return true
	})
	return nil
}
func (l *LSH) saveCacheMap() error {
	switch l.precisionHandler.Type() {
	case PrecisionInt8:
		return _saveCacheMap[int8](l)
	case PrecisionInt16:
		return _saveCacheMap[int16](l)
	case PrecisionFloat32:
		return _saveCacheMap[float32](l)
	case PrecisionFloat64:
		return _saveCacheMap[float64](l)
	default:
		return fmt.Errorf("unsupported precision type")
	}
}
func _loadCacheMap[T int8 | int16 | float32 | float64](l *LSH) error {
	tp := fmt.Sprintf("%v/cacheMap.txt", l.filePath)
	f, err := os.Open(tp)
	if err != nil {
		return err
	}
	defer f.Close()

	// 创建 JSON 解码器
	decoder := json.NewDecoder(f)
	item := kv[T]{}
	for decoder.More() {
		if err := decoder.Decode(&item); err != nil {
			return err
		}
		l.cacheMap.Store(item.K, item.V)
		// 显式释放引用
		item = kv[T]{}
	}

	return nil
}
func (l *LSH) loadCacheMap() error {
	switch l.precisionHandler.Type() {
	case PrecisionInt8:
		return _loadCacheMap[int8](l)
	case PrecisionInt16:
		return _loadCacheMap[int16](l)
	case PrecisionFloat32:
		return _loadCacheMap[float32](l)
	case PrecisionFloat64:
		return _loadCacheMap[float64](l)
	default:
		return fmt.Errorf("unsupported precision type")
	}
}

func (l *LSH) SaveToFile() error {
	if l.filePath == "" {
		return fmt.Errorf("file path is empty")
	}
	if err := l.SaveRandomVecs(); err != nil {
		return err
	}
	if err := l.saveTables(); err != nil {
		return err
	}
	if l.useCache {
		if err := l.saveCacheMap(); err != nil {
			return err
		}
	}
	return nil
}
func (l *LSH) LoadFromFile() error {
	if l.filePath == "" {
		return fmt.Errorf("file path is empty")
	}
	if err := l.LoadRandomVecs(); err != nil {
		return err
	}
	if err := l.loadTables(); err != nil {
		return err
	}
	if l.useCache {
		if err := l.loadCacheMap(); err != nil {
			return err
		}
	}

	return nil
}

func (l *LSH) Migrate(numTables, numHashes int) (error, *LSH) {
	if !l.useCache {
		return fmt.Errorf("cache is disabled"), nil
	}
	nl := NewLSH(numTables, numHashes, l.vectorSize)
	nl.useCache = l.useCache
	nl.filePath = l.filePath
	l.cacheMap.Range(func(key, value any) bool {
		nl.AddVector(key.(string), value.([]float64))
		return true
	})
	return nil, nl
}
