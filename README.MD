# Locality-Sensitive Hashing (LSH) Implementation

## Introduction

This project implements a Locality-Sensitive Hashing (LSH) algorithm in Go. LSH is a method for performing probabilistic dimension reduction of high-dimensional data. It is used for tasks such as nearest neighbor search in high-dimensional spaces, which is common in applications like image retrieval, document clustering, and recommendation systems.

## Features

- **Add Vectors**: Add individual or batch vectors to the LSH tables.
- **Query**: Perform approximate nearest neighbor search for a given vector.
- **Batch Query**: Perform approximate nearest neighbor search for multiple vectors.
- **Evaluate**: Evaluate the precision, recall, and F1 score of the LSH implementation.
- **Save and Load**: Save and load the LSH tables and vectors to/from files.
- **Migrate**: Migrate the LSH tables to a new configuration with different parameters.

## Installation

Clone the repository:

```sh
git clone https://github.com/yourusername/lsh.git
cd lsh
```

Install the required dependencies:

```sh
go get -u github.com/lejianwen/lsh
```

## Usage

### Adding Vectors

To add vectors to the LSH tables, use the `AddVector` or `AddVectors` methods:

```go
lsh := NewLSH(10, 10, 2048, "")
lsh.AddVector("item1", []float64{...})
lsh.AddVectors(map[string][]float64{
    "item1": {...},
    "item2": {...},
})
```

### Querying

To perform a nearest neighbor search, use the `Query` or `BatchQuery` methods:

```go
neighbors := lsh.Query([]float64{...}, 2)
batchNeighbors := lsh.BatchQuery(map[string][]float64{
    "item1": {...},
    "item2": {...},
}, 2)
```

### Evaluation

To evaluate the LSH implementation, use the `EvaluateLSH` method:

```go
precision, recall, f1 := lsh.EvaluateLSH(testQueries, groundTruth, 2)
```

### Saving and Loading

To save and load the LSH tables and vectors, use the `SaveToFile` and `LoadFromFile` methods:

```go
err := lsh.SaveToFile()
if err != nil {
    log.Fatalf("SaveToFile error: %v", err)
}

err = lsh.LoadFromFile()
if err != nil {
    log.Fatalf("LoadFromFile error: %v", err)
}
```

### Migration

To migrate the LSH tables to a new configuration, use the `Migrate` method:

```go
err, newLsh := lsh.Migrate(12, 12)
if err != nil {
    log.Fatalf("Migrate error: %v", err)
}
```

### Benchmarking

To benchmark the performance of the query operations, use the `BenchmarkLSH_Query` and `BenchmarkLSH_BatchQuery` methods:

```go
func BenchmarkLSH_Query(b *testing.B) {
    lsh := NewLSH(10, 10, 2048, "")
    lsh.AddVectors(vectors)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        lsh.Query(vectors["item1"], 2)
    }
}

func BenchmarkLSH_BatchQuery(b *testing.B) {
    lsh := NewLSH(10, 10, 2048, "")
    lsh.AddVectors(vectors)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        lsh.BatchQuery(vectors, 2)
    }
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.