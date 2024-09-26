package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
)

const (
    numTrees          = 5
    maxDepth          = 6
    minSamplesLeaf    = 20
    trainTestSplit    = 0.8
    numFeatures       = 5
    numClasses        = 2
    bootstrapRatio    = 0.05
    featureSplitRatio = 0.5
    maxGoroutines     = 4
)

type Sample struct {
    features []float64
    label    int
}

type Node struct {
    feature    int
    threshold  float64
    left       *Node
    right      *Node
    prediction int
}

type RandomForest struct {
    trees []*Node
}

func main() {
    totalStart := time.Now()

    fmt.Println("Loading and preprocessing data...")
    samples := loadData("data-clean.csv")

    trainSamples, testSamples := splitData(samples)

    fmt.Printf("Loaded %d training samples and %d testing samples.\n", len(trainSamples), len(testSamples))

    fmt.Println("Training Random Forest...")
    forest := trainRandomForest(trainSamples)
    fmt.Println("Training completed.")

    fmt.Println("Evaluating the model...")
    accuracy := evaluateModel(forest, testSamples)
    fmt.Printf("Model accuracy: %.2f%%\n", accuracy*100)

    fmt.Println("Making predictions on provided data points:")
    testCustomDataPoints(forest)

    fmt.Printf("Total execution time: %v\n", time.Since(totalStart))
}

func loadData(filename string) []Sample {
    file, err := os.Open(filename)
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        log.Fatal(err)
    }

    var samples []Sample
    samples = make([]Sample, 0, len(records)-1)
    for i, record := range records {
        if i == 0 {
            continue
        }
        if len(record) < numFeatures+1 {
            log.Printf("Skipping incomplete row %d\n", i+1)
            continue
        }
        sample := Sample{features: make([]float64, numFeatures)}
        for j := 0; j < numFeatures; j++ {
            val, err := strconv.ParseFloat(record[j], 64)
            if err != nil {
                log.Fatalf("Error parsing feature at row %d, column %d: %v", i+1, j+1, err)
            }
            sample.features[j] = val
        }
        label, err := strconv.Atoi(record[numFeatures])
        if err != nil {
            log.Fatalf("Error parsing label at row %d: %v", i+1, err)
        }
        sample.label = label
        samples = append(samples, sample)
    }

    fmt.Printf("Loaded %d samples\n", len(samples))
    return samples
}

func splitData(samples []Sample) ([]Sample, []Sample) {
    rand.Seed(time.Now().UnixNano())
    rand.Shuffle(len(samples), func(i, j int) {
        samples[i], samples[j] = samples[j], samples[i]
    })
    splitIndex := int(float64(len(samples)) * trainTestSplit)
    trainSamples := make([]Sample, splitIndex)
    testSamples := make([]Sample, len(samples)-splitIndex)
    copy(trainSamples, samples[:splitIndex])
    copy(testSamples, samples[splitIndex:])
    return trainSamples, testSamples
}

func trainRandomForest(samples []Sample) RandomForest {
    forest := RandomForest{trees: make([]*Node, numTrees)}

    var wg sync.WaitGroup
    semaphore := make(chan struct{}, maxGoroutines)

    for i := 0; i < numTrees; i++ {
        wg.Add(1)
        semaphore <- struct{}{}
        go func(i int) {
            defer wg.Done()
            defer func() { <-semaphore }()

            bootstrapSamples := bootstrapSamples(samples)
            forest.trees[i] = buildTree(bootstrapSamples, 0)
            fmt.Printf("Built tree %d/%d\n", i+1, numTrees)
        }(i)
    }

    wg.Wait()
    return forest
}

func bootstrapSamples(samples []Sample) []Sample {
    numBootstrapSamples := int(float64(len(samples)) * bootstrapRatio)
    bootstrapSamples := make([]Sample, numBootstrapSamples)
    for i := 0; i < numBootstrapSamples; i++ {
        index := rand.Intn(len(samples))
        bootstrapSamples[i] = samples[index]
    }
    return bootstrapSamples
}

func buildTree(samples []Sample, depth int) *Node {
    if len(samples) <= minSamplesLeaf || depth >= maxDepth {
        prediction := majorityVote(samples)
        return &Node{prediction: prediction}
    }

    bestFeature, bestThreshold := findBestSplit(samples)
    if bestFeature == -1 {
        prediction := majorityVote(samples)
        return &Node{prediction: prediction}
    }

    leftSamples, rightSamples := splitSamples(samples, bestFeature, bestThreshold)
    if len(leftSamples) == 0 || len(rightSamples) == 0 {
        prediction := majorityVote(samples)
        return &Node{prediction: prediction}
    }

    node := &Node{
        feature:   bestFeature,
        threshold: bestThreshold,
    }

    node.left = buildTree(leftSamples, depth+1)
    node.right = buildTree(rightSamples, depth+1)

    return node
}

func findBestSplit(samples []Sample) (int, float64) {
    bestGini := math.Inf(1)
    bestFeature := -1
    bestThreshold := 0.0

    numFeaturesToTry := int(math.Round(float64(numFeatures) * featureSplitRatio))
    featuresToTry := rand.Perm(numFeatures)[:numFeaturesToTry]

    for _, feature := range featuresToTry {
        thresholds := getUniqueValues(samples, feature)
        for _, threshold := range thresholds {
            leftSamples, rightSamples := splitSamples(samples, feature, threshold)
            gini := calculateGini(leftSamples, rightSamples)

            if gini < bestGini {
                bestGini = gini
                bestFeature = feature
                bestThreshold = threshold
            }
        }
    }

    return bestFeature, bestThreshold
}

func getUniqueValues(samples []Sample, feature int) []float64 {
    uniqueMap := make(map[float64]struct{})
    for _, sample := range samples {
        uniqueMap[sample.features[feature]] = struct{}{}
    }
    values := make([]float64, 0, len(uniqueMap))
    for value := range uniqueMap {
        values = append(values, value)
    }
    return values
}

func splitSamples(samples []Sample, feature int, threshold float64) ([]Sample, []Sample) {
    left := make([]Sample, 0, len(samples)/2)
    right := make([]Sample, 0, len(samples)/2)
    for _, sample := range samples {
        if sample.features[feature] <= threshold {
            left = append(left, sample)
        } else {
            right = append(right, sample)
        }
    }
    return left, right
}

func calculateGini(leftSamples, rightSamples []Sample) float64 {
    total := len(leftSamples) + len(rightSamples)
    leftGini := giniImpurity(leftSamples)
    rightGini := giniImpurity(rightSamples)
    return (float64(len(leftSamples))*leftGini + float64(len(rightSamples))*rightGini) / float64(total)
}

func giniImpurity(samples []Sample) float64 {
    if len(samples) == 0 {
        return 0.0
    }
    classCounts := make([]int, numClasses)
    for _, sample := range samples {
        classCounts[sample.label]++
    }
    impurity := 1.0
    for _, count := range classCounts {
        prob := float64(count) / float64(len(samples))
        impurity -= prob * prob
    }
    return impurity
}

func majorityVote(samples []Sample) int {
    classCounts := make([]int, numClasses)
    for _, sample := range samples {
        classCounts[sample.label]++
    }
    maxCount := 0
    majorityClass := 0
    for class, count := range classCounts {
        if count > maxCount {
            maxCount = count
            majorityClass = class
        }
    }
    return majorityClass
}

func evaluateModel(forest RandomForest, testSamples []Sample) float64 {
    correct := 0
    total := len(testSamples)
    var wg sync.WaitGroup
    semaphore := make(chan struct{}, maxGoroutines)
    var mu sync.Mutex

    for _, sample := range testSamples {
        wg.Add(1)
        semaphore <- struct{}{}
        go func(sample Sample) {
            defer wg.Done()
            defer func() { <-semaphore }()

            prediction := predict(forest, sample)
            mu.Lock()
            if prediction == sample.label {
                correct++
            }
            mu.Unlock()
        }(sample)
    }
    wg.Wait()

    accuracy := float64(correct) / float64(total)
    return accuracy
}

func predict(forest RandomForest, sample Sample) int {
    votes := make([]int, numClasses)
    var wg sync.WaitGroup
    var mu sync.Mutex

    for _, tree := range forest.trees {
        wg.Add(1)
        go func(tree *Node) {
            defer wg.Done()
            prediction := predictTree(tree, sample)
            mu.Lock()
            votes[prediction]++
            mu.Unlock()
        }(tree)
    }
    wg.Wait()
    finalPrediction := argmax(votes)
    return finalPrediction
}

func predictTree(node *Node, sample Sample) int {
    if node.left == nil && node.right == nil {
        return node.prediction
    }
    if sample.features[node.feature] <= node.threshold {
        return predictTree(node.left, sample)
    }
    return predictTree(node.right, sample)
}

func argmax(slice []int) int {
    maxIndex := 0
    maxValue := slice[0]
    for i, value := range slice {
        if value > maxValue {
            maxValue = value
            maxIndex = i
        }
    }
    return maxIndex
}

func testCustomDataPoints(forest RandomForest) {
	dataPoints := [][]float64{
		{54, 0, 20210916, 1, 1},
		{26, 0, 20210525, 2, 1},
		{59, 0, 20210310, 3, 2},
		{54, 0, 20210212, 1, 1},
		{34, 0, 20210114, 3, 1},
		{65, 1, 20210928, 2, 1},
		{54, 0, 20211224, 3, 1},
		{63, 1, 20211115, 3, 1},
		{61, 1, 20211127, 2, 1},
		{31, 1, 20210712, 2, 1},
		{20, 0, 20210225, 2, 2},
		{15, 1, 20210101, 1, 2},
	}

    for i, features := range dataPoints {
        sample := Sample{features: features}
        prediction := predict(forest, sample)
        fmt.Printf("Data point #%d: Features = %v, Predicted REFUERZO = %d\n", i+1, features, prediction)
    }
}

func init() {
    rand.Seed(time.Now().UnixNano())
    runtime.GOMAXPROCS(runtime.NumCPU())
}
