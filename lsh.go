package lsh

import (
	"fmt"
	"github.com/bytedance/sonic"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

var json = sonic.ConfigStd

// LSH 结构体（并发安全版本）
type LSH struct {
	numTables  int                   // 哈希表数量 ： 越大，召回率高，但内存占用大。越 小，查询速度快，但召回率低。一般取 10~100 之间
	numHashes  int                   // 每个表的哈希函数数量 ： 越大，召回率低，但误检率低（更精确）。 越小，召回率高，但误检率高（可能返回不相关的向量）。一般取 4~16 之间
	vectorSize int                   // 向量维度
	randomVecs [][][]float64         // 随机向量集合 [table][hash][dim]
	hashTables []map[string][]string // 哈希表存储 [table][hash]
	tableLocks []sync.RWMutex        // 每个哈希表的读写锁
	useCache   bool                  // 是否使用缓存
	cacheMap   sync.Map              // 缓存
	filePath   string                // 缓存文件路径
}
type scoreItem struct {
	Id    string
	Score float64
}
type kv struct {
	K string    `json:"k"`
	V []float64 `json:"v"`
}

// 初始化 LSH
func NewLSH(numTables, numHashes, vectorSize int, filePath string) *LSH {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// 生成随机向量（三维数组）
	randomVecs := make([][][]float64, numTables)
	for t := 0; t < numTables; t++ {
		randomVecs[t] = make([][]float64, numHashes)
		for h := 0; h < numHashes; h++ {
			randomVecs[t][h] = randomUnitVector(vectorSize, r)
		}
	}

	// 初始化哈希表和锁
	hashTables := make([]map[string][]string, numTables)
	tableLocks := make([]sync.RWMutex, numTables)
	for t := 0; t < numTables; t++ {
		hashTables[t] = make(map[string][]string)
	}

	return &LSH{
		numTables:  numTables,
		numHashes:  numHashes,
		vectorSize: vectorSize,
		randomVecs: randomVecs,
		hashTables: hashTables,
		tableLocks: tableLocks,
		useCache:   true,
		cacheMap:   sync.Map{},
		filePath:   strings.TrimRight(filePath, "/"),
	}
}

func NewNoCacheLSH(numTables, numHashes, vectorSize int) *LSH {
	l := NewLSH(numTables, numHashes, vectorSize, "")
	l.useCache = false
	return l
}

// 生成随机向量（范围：[-1, 1)）
func generateRandomVector(size int, r *rand.Rand) []float64 {
	vector := make([]float64, size)
	for i := range vector {
		vector[i] = r.Float64()*2 - 1
	}
	return vector
}

// 生成单位向量
func randomUnitVector(dimension int, r *rand.Rand) []float64 {
	vec := make([]float64, dimension)
	var norm float64
	for i := range vec {
		vec[i] = r.Float64()*2 - 1 // 生成 [-1,1] 之间的随机数
		norm += vec[i] * vec[i]
	}
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

// 计算点积
func (l *LSH) dotProduct(vecA, vecB []float64) float64 {
	sum := 0.0
	for i := range vecA {
		sum += vecA[i] * vecB[i]
	}
	return sum
}

// 计算单个哈希表的哈希值（二进制字符串）
func (l *LSH) computeHash(vector []float64, tableIdx int) string {
	var sb strings.Builder
	sb.Grow(l.numHashes) // 预分配空间

	for h := 0; h < l.numHashes; h++ {
		dot := l.dotProduct(vector, l.randomVecs[tableIdx][h])
		if dot >= 0 {
			sb.WriteByte('1')
		} else {
			sb.WriteByte('0')
		}
	}
	return sb.String()
}

// 添加向量（并行版）
func (l *LSH) AddVector(id string, vector []float64) {
	if l.useCache {
		l.cacheMap.Store(id, vector)
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

// 查询近似最近邻（并行版）
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
		sortedSimilarities := make([]scoreItem, 0, k)
		candidates.Range(func(key, value any) bool {
			id := key.(string)
			v, ok := l.cacheMap.Load(id)
			if ok {
				s := l.dotProduct(vector, v.([]float64))
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

func (l *LSH) saveCacheMap() error {
	tp := fmt.Sprintf("%v/cacheMap.txt", l.filePath)
	f, err := os.Create(tp)
	if err != nil {
		return err
	}
	defer f.Close()

	// 创建 JSON 编码器
	encoder := json.NewEncoder(f)
	item := kv{
		K: "",
		V: nil,
	}
	// 逐个写入 key-value
	l.cacheMap.Range(func(key, value any) bool {
		item = kv{
			K: key.(string),
			V: value.([]float64),
		}
		if err := encoder.Encode(item); err != nil {
			return false
		}
		return true
	})
	return nil
}

func (l *LSH) loadCacheMap() error {
	tp := fmt.Sprintf("%v/cacheMap.txt", l.filePath)
	f, err := os.Open(tp)
	if err != nil {
		return err
	}
	defer f.Close()

	// 创建 JSON 解码器
	decoder := json.NewDecoder(f)
	item := kv{
		K: "",
		V: nil,
	}
	for decoder.More() {
		if err := decoder.Decode(&item); err != nil {
			return err
		}
		l.cacheMap.Store(item.K, item.V)
		// 显式释放引用
		item = kv{}
	}

	return nil
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
	if err := l.saveCacheMap(); err != nil {
		return err
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
	if err := l.loadCacheMap(); err != nil {
		return err
	}
	return nil
}

func (l *LSH) Migrate(numTables, numHashes int) (error, *LSH) {
	if !l.useCache {
		return fmt.Errorf("cache is disabled"), nil
	}
	nl := NewLSH(numTables, numHashes, l.vectorSize, l.filePath)
	l.cacheMap.Range(func(key, value any) bool {
		nl.AddVector(key.(string), value.([]float64))
		return true
	})
	return nil, nl
}
