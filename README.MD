## 简介

该项目实现了一个局部敏感哈希（LSH）算法，用于高效地进行相似性搜索和向量查询。

## 功能

- 添加向量
- 查询相似向量
- 批量查询相似向量
- 迁移哈希表
- 评估LSH算法的精度、召回率和F1值
- 保存和加载哈希表
- 设置不同精度处理器

## 安装

克隆

```sh
git clone https://github.com/lejianwen/lsh
cd lsh
```

## 使用

```go
go get -u github.com/lejianwen/lsh
```

### 添加向量

```go
var vectors = map[string][]float64{
"item1": {-0.010979787, 0.0208240951, -0.006406612, 0.0070472737, ...},
"item2": {-0.0402136976, 0.0058495451, -0.0306575614, ...},
"item3": {-0.0333253096, -0.0037028126, -0.0085615029, ...},
"item4": {-0.01552303, 0.0351111111, -0.0183980546, ...},
"item5": {0.0097557013, 0.0152862035, 0.0014691215,...},
"item6": {-0.0180807306, 0.0101798173, -0.0130499444,...},
...
}

l := NewLSH(10, 10, 2048)
l.AddVector("item1", vectors["item1"])

l.AddVectors(vectors)

vec := []float64{3,4,...} //非L2归一化的向量
//先归一化
lsh.L2Normalize(vec)
l.AddVector("item1", vec)

```



### 查询相似向量

```go
neighbors := lsh.Query(vectors["item1"], 2)
fmt.Printf("Query result: %v\n", neighbors)
```

### 批量查询相似向量

```go
neighbors := lsh.BatchQuery(vectors, 2)
fmt.Printf("Query result: %+v\n", neighbors)
```

### 迁移哈希表

```go
err, _ := lsh.Migrate(12, 12)
if err != nil {
log.Fatalf("Migrate error: %v", err)
}
```

### 评估LSH算法

```go
precision, recall, f1 := lsh.EvaluateLSH(qvectors, groundTruth, 2)
fmt.Printf("Evaluate result: precision=%v, recall=%v, f1=%v\n", precision, recall, f1)
```

### 保存和加载到文件

```go
lsh.SetFilePath("./")
err := lsh.SaveToFile()
if err != nil {
log.Fatalf("SaveToFile error: %v", err)
}

err = lsh.LoadFromFile()
if err != nil {
log.Fatalf("LoadFromFile error: %v", err)
}
```

### 设置精度处理器

支持`int8`、`int16`、`float32`和`float64`精度
精度越低内存占用越小，但是精度越低，查询结果的准确性也会降低

```go
lsh.SetPrecisionHandler(NewPrecisionHandler(PrecisionInt16))
```

## 许可证

此项目使用MIT许可证。详见`LICENSE`文件。
