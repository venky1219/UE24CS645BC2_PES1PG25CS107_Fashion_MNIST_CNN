# UE24CS645BC2_PES1PG25CS107_Fashion_MNIST_CNN

A Convolutional Neural Network built **from first principles** using only NumPy — no PyTorch, no TensorFlow, no Keras.  
Trained and evaluated on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

---
## Name
M VENKATESH

## USN
PES1PG25CS107

---

# Submission Details

## Architecture

```
Input       (N, 1, 28, 28)   — greyscale images
Conv1       8 filters, 3×3   → (N, 8, 26, 26)  + ReLU
MaxPool1    2×2              → (N, 8, 13, 13)
Conv2       16 filters, 3×3  → (N, 16, 11, 11) + ReLU
MaxPool2    2×2              → (N, 16, 5, 5)
Flatten                      → (N, 400)
FC1         400 → 128        + ReLU
FC2         128 → 10         + Softmax
Loss        Categorical Cross-Entropy
Optimiser   Mini-batch SGD
```

---

## Project Structure

```
.
├── cnn_fashion_mnist.py   # All layer implementations + training + evaluation
└── README.md
```

### Key Components in `cnn_fashion_mnist.py`

| Section | Class / Function | Description |
|---------|-----------------|-------------|
| §1 | `load_fashion_mnist()` | Downloads raw IDX files, normalises to [0,1] |
| §2 | `ConvLayer` | 2-D convolution — forward & backward, SGD weight update |
| §3 | `MaxPoolLayer` | Non-overlapping 2×2 max-pool — forward & backward (mask-based) |
| §4 | `FlattenLayer` | Reshape (N,C,H,W) ↔ (N,C·H·W) |
| §5 | `FCLayer` | Dense layer with ReLU or Softmax activation — forward & backward |
| §6 | `cross_entropy_loss()` | Categorical cross-entropy + combined Softmax gradient |
| §7 | `CNN` | Assembles all layers; `forward()` and `backward()` |
| §8 | `train()` | Mini-batch SGD loop with epoch-level metrics |
| §9 | `evaluate()` | Overall and per-class accuracy report |
| §10 | `confusion_matrix_text()` | 10×10 confusion matrix printed to stdout |

---

## How Backpropagation Works (brief)

### Convolution Layer
The gradient of the loss w.r.t. the filter weights is the cross-correlation of the input patch with the upstream gradient.  
The gradient w.r.t. the input is the full convolution of the upstream gradient with the flipped filters.

```
dW[f,c,i,j] = Σ_{n,h,w} dZ[n,f,h,w] · X[n,c,h+i,w+j]
dX[n,c,h,w] = Σ_{f,i,j} dZ[n,f,h-i,w-j] · W[f,c,i,j]   (full conv)
```

### MaxPool Layer
Gradients flow only through the positions that held the maximum value (stored as a boolean mask during the forward pass).

### FC Layer (Softmax + Cross-Entropy)
The combined gradient simplifies elegantly:  
`dZ = ŷ − one_hot(y)` — probability vector minus the one-hot true label.

---

## Requirements

```
python >= 3.8
numpy
```

No other dependencies. The dataset is downloaded automatically on first run.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/venky1219/UE24CS645BC2_PES1PG25CS107_Fashion_MNIST_CNN


# Install numpy if needed
pip install numpy

# Run (quick mode: 5 000 train samples, 5 epochs)
python cnn_fashion_mnist.py
```

### Quick mode vs Full training

Inside `cnn_fashion_mnist.py`, find the `QUICK` flag near the bottom:

```python
QUICK = True    # 5 000 samples — runs in ~10 minutes on CPU
QUICK = False   # all 60 000 samples — expect 1-2 hrs on CPU
```

Set `QUICK = False` for a full training run (~85–88% test accuracy expected after 10 epochs).

---

## Sample Output (Quick mode, 5 epochs)

```
Loading Fashion-MNIST …
  (Quick mode: 5 000 train / 1 000 test samples)
  Train: (5000, 1, 28, 28)  Test: (1000, 1, 28, 28)

══════════════════════════════════════════
   CNN Architecture Summary
══════════════════════════════════════════
  ConvLayer(filters=8, kernel=3×3, in_channels=1)
  MaxPoolLayer(pool_size=2)
  ConvLayer(filters=16, kernel=3×3, in_channels=8)
  MaxPoolLayer(pool_size=2)
  FlattenLayer()
  FCLayer(in=400, out=128, act=relu)
  FCLayer(in=128, out=10, act=softmax)
══════════════════════════════════════════

Epoch  1/5  loss=2.1983  train_acc=21.34%  val_acc=28.10%  (87.3s)
Epoch  2/5  loss=1.8721  train_acc=38.52%  val_acc=42.30%  (86.1s)
...
```

---

## Fashion-MNIST Classes

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---
