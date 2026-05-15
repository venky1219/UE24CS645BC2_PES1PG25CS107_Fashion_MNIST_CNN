"""
CNN from Scratch – Fashion MNIST
=================================
Implements Convolution, MaxPool, Flatten, and Fully-Connected layers
with forward AND backward passes using only NumPy.
"""

import numpy as np
import struct
import gzip
import os
import urllib.request
import time

# ─────────────────────────────────────────────
# 1.  DATA LOADING  (raw IDX files via urllib)
# ─────────────────────────────────────────────
BASE_URL = "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def download_data(data_dir="fashion_mnist_data"):
    os.makedirs(data_dir, exist_ok=True)
    for name, fname in FILES.items():
        dest = os.path.join(data_dir, fname)
        if not os.path.exists(dest):
            print(f"  Downloading {fname} …")
            urllib.request.urlretrieve(BASE_URL + fname, dest)
    return data_dir


def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int32)


def load_fashion_mnist(data_dir="fashion_mnist_data"):
    data_dir = download_data(data_dir)
    X_train = load_images(os.path.join(data_dir, FILES["train_images"]))
    y_train = load_labels(os.path.join(data_dir, FILES["train_labels"]))
    X_test  = load_images(os.path.join(data_dir, FILES["test_images"]))
    y_test  = load_labels(os.path.join(data_dir, FILES["test_labels"]))
    # Add channel dim: (N, H, W) → (N, 1, H, W)
    return (X_train[:, np.newaxis], y_train,
            X_test[:, np.newaxis],  y_test)


# ─────────────────────────────────────────────
# 2.  CONVOLUTION LAYER
# ─────────────────────────────────────────────
class ConvLayer:
    """
    2-D Convolution layer (no padding, stride = 1).

    Shapes
    ------
    Input  : (N, C_in,  H,    W)
    Filters: (F, C_in,  KH,   KW)   – F filters of size KH×KW
    Output : (N, F,     H_out, W_out)
        H_out = H - KH + 1
        W_out = W - KW + 1
    """

    def __init__(self, n_filters: int, filter_size: int,
                 n_channels: int = 1, lr: float = 1e-3):
        self.F  = n_filters
        self.KH = self.KW = filter_size
        self.lr = lr

        # He initialisation
        fan_in = n_channels * filter_size * filter_size
        self.W = np.random.randn(n_filters, n_channels,
                                 filter_size, filter_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((n_filters, 1))

        # Cache for backward pass
        self._cache = {}

    # ── Forward ──────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X : (N, C_in, H, W)
        Returns Z : (N, F, H_out, W_out)
        """
        N, C, H, W = X.shape
        H_out = H - self.KH + 1
        W_out = W - self.KW + 1

        Z = np.zeros((N, self.F, H_out, W_out), dtype=np.float32)

        for i in range(H_out):
            for j in range(W_out):
                patch = X[:, :, i:i+self.KH, j:j+self.KW]   # (N, C, KH, KW)
                # (N, C, KH, KW) × (F, C, KH, KW) → sum → (N, F)
                Z[:, :, i, j] = np.tensordot(patch, self.W,
                                              axes=([1, 2, 3], [1, 2, 3])) + self.b.T

        self._cache = {"X": X, "Z": Z}
        return Z

    # ── ReLU activation ──────────────────────
    def relu(self, Z: np.ndarray) -> np.ndarray:
        A = np.maximum(0, Z)
        self._cache["A"] = A
        return A

    # ── Backward ─────────────────────────────
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        dA : gradient w.r.t. the ReLU output  (N, F, H_out, W_out)
        Returns dX : gradient w.r.t. input    (N, C_in, H, W)
        """
        X  = self._cache["X"]
        Z  = self._cache["Z"]

        # Gradient through ReLU
        dZ = dA * (Z > 0)                               # (N, F, H_out, W_out)

        N, C, H, W = X.shape
        H_out, W_out = dZ.shape[2], dZ.shape[3]

        dW = np.zeros_like(self.W)                      # (F, C, KH, KW)
        db = np.zeros_like(self.b)                      # (F, 1)
        dX = np.zeros_like(X)                           # (N, C, H, W)

        for i in range(H_out):
            for j in range(W_out):
                patch = X[:, :, i:i+self.KH, j:j+self.KW]   # (N, C, KH, KW)
                dz_ij = dZ[:, :, i, j]                        # (N, F)

                # dW += sum over batch: patch^T × dz
                dW += np.tensordot(dz_ij, patch, axes=([0], [0]))   # (F,C,KH,KW)

                # dX patch region
                dX[:, :, i:i+self.KH, j:j+self.KW] += \
                    np.tensordot(dz_ij, self.W, axes=([1], [0]))     # (N,C,KH,KW)

            db += dZ[:, :, i, :].sum(axis=(0, 2)).reshape(self.F, 1)

        # SGD update
        self.W -= self.lr * dW / N
        self.b -= self.lr * db / N

        return dX

    def __repr__(self):
        return (f"ConvLayer(filters={self.F}, kernel={self.KH}×{self.KW}, "
                f"in_channels={self.W.shape[1]})")


# ─────────────────────────────────────────────
# 3.  MAX-POOL LAYER
# ─────────────────────────────────────────────
class MaxPoolLayer:
    """
    2-D Max-Pooling (non-overlapping windows, stride = pool_size).

    Input  : (N, C, H,    W)
    Output : (N, C, H//p, W//p)   where p = pool_size
    """

    def __init__(self, pool_size: int = 2):
        self.p = pool_size
        self._cache = {}

    # ── Forward ──────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        N, C, H, W = X.shape
        p = self.p
        H_out, W_out = H // p, W // p

        out   = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        masks = np.zeros_like(X, dtype=bool)              # where the max was

        for i in range(H_out):
            for j in range(W_out):
                region = X[:, :, i*p:(i+1)*p, j*p:(j+1)*p]   # (N,C,p,p)
                max_val = region.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j] = max_val[:, :, 0, 0]
                masks[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = (region == max_val)

        self._cache = {"masks": masks, "input_shape": X.shape}
        return out

    # ── Backward ─────────────────────────────
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout : (N, C, H_out, W_out)
        Returns dX : (N, C, H, W)
        """
        masks = self._cache["masks"]
        N, C, H, W = self._cache["input_shape"]
        p = self.p
        H_out, W_out = H // p, W // p

        dX = np.zeros((N, C, H, W), dtype=np.float32)

        for i in range(H_out):
            for j in range(W_out):
                d = dout[:, :, i, j][:, :, np.newaxis, np.newaxis]  # (N,C,1,1)
                dX[:, :, i*p:(i+1)*p, j*p:(j+1)*p] += \
                    masks[:, :, i*p:(i+1)*p, j*p:(j+1)*p] * d

        return dX

    def __repr__(self):
        return f"MaxPoolLayer(pool_size={self.p})"


# ─────────────────────────────────────────────
# 4.  FLATTEN LAYER
# ─────────────────────────────────────────────
class FlattenLayer:
    """Reshapes (N, C, H, W) → (N, C*H*W)."""

    def __init__(self):
        self._cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._cache["shape"] = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._cache["shape"])

    def __repr__(self):
        return "FlattenLayer()"


# ─────────────────────────────────────────────
# 5.  FULLY-CONNECTED LAYER  (with ReLU or Softmax)
# ─────────────────────────────────────────────
class FCLayer:
    """
    Dense layer.
    activation : 'relu' for hidden layers, 'softmax' for the output layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", lr: float = 1e-3):
        self.activation = activation
        self.lr = lr

        # He init for ReLU, Xavier for Softmax
        if activation == "relu":
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = np.sqrt(1.0 / in_features)

        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self._cache = {}

    # ── Activations ──────────────────────────
    @staticmethod
    def _relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def _softmax(Z):
        Z_shifted = Z - Z.max(axis=1, keepdims=True)   # numerical stability
        exp_Z = np.exp(Z_shifted)
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    # ── Forward ──────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """X : (N, in_features)  →  A : (N, out_features)"""
        Z = X @ self.W + self.b
        if self.activation == "relu":
            A = self._relu(Z)
        else:
            A = self._softmax(Z)

        self._cache = {"X": X, "Z": Z, "A": A}
        return A

    # ── Backward ─────────────────────────────
    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        For the output (softmax) layer, dA is already dZ (passed from loss).
        For hidden (relu) layers, dA is the gradient w.r.t. A.
        Returns dX : gradient w.r.t. input.
        """
        X = self._cache["X"]
        Z = self._cache["Z"]
        N = X.shape[0]

        if self.activation == "relu":
            dZ = dA * (Z > 0)
        else:
            dZ = dA           # cross-entropy + softmax combined gradient

        dW = X.T @ dZ / N
        db = dZ.sum(axis=0, keepdims=True) / N
        dX = dZ @ self.W.T

        # SGD update
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dX

    def __repr__(self):
        return (f"FCLayer(in={self.W.shape[0]}, out={self.W.shape[1]}, "
                f"act={self.activation})")


# ─────────────────────────────────────────────
# 6.  LOSS  (Categorical Cross-Entropy)
# ─────────────────────────────────────────────
def cross_entropy_loss(probs: np.ndarray, y: np.ndarray):
    """
    probs : (N, C)  softmax output
    y     : (N,)    integer class labels
    Returns (scalar loss, dZ for backward)
    """
    N = probs.shape[0]
    log_likelihood = -np.log(probs[np.arange(N), y] + 1e-12)
    loss = log_likelihood.mean()

    # Combined softmax + cross-entropy gradient: p - one_hot(y)
    dZ = probs.copy()
    dZ[np.arange(N), y] -= 1.0

    return loss, dZ


# ─────────────────────────────────────────────
# 7.  FULL CNN MODEL
# ─────────────────────────────────────────────
class CNN:
    """
    Architecture
    ─────────────────────────────────────
    Input      : (N, 1, 28, 28)
    Conv1      : 8 filters, 3×3  → (N, 8, 26, 26)   + ReLU
    MaxPool1   : 2×2             → (N, 8, 13, 13)
    Conv2      : 16 filters, 3×3 → (N, 16, 11, 11)   + ReLU
    MaxPool2   : 2×2             → (N, 16, 5, 5)     (floor)
    Flatten    :                 → (N, 400)
    FC1        : 400 → 128       + ReLU
    FC2        : 128 → 10        + Softmax
    ─────────────────────────────────────
    """

    def __init__(self, lr: float = 5e-3):
        self.conv1  = ConvLayer(n_filters=8,  filter_size=3, n_channels=1,  lr=lr)
        self.pool1  = MaxPoolLayer(pool_size=2)
        self.conv2  = ConvLayer(n_filters=16, filter_size=3, n_channels=8,  lr=lr)
        self.pool2  = MaxPoolLayer(pool_size=2)
        self.flat   = FlattenLayer()
        self.fc1    = FCLayer(16 * 5 * 5, 128, activation="relu",    lr=lr)
        self.fc2    = FCLayer(128,         10,  activation="softmax", lr=lr)

    # ── Forward ──────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        out = self.conv1.relu(self.conv1.forward(X))
        out = self.pool1.forward(out)
        out = self.conv2.relu(self.conv2.forward(out))
        out = self.pool2.forward(out)
        out = self.flat.forward(out)
        out = self.fc1.forward(out)
        out = self.fc2.forward(out)
        return out

    # ── Backward ─────────────────────────────
    def backward(self, dZ_out: np.ndarray) -> None:
        d = self.fc2.backward(dZ_out)
        d = self.fc1.backward(d)
        d = self.flat.backward(d)
        d = self.pool2.backward(d)
        d = self.conv2.backward(d)
        d = self.pool1.backward(d)
        d = self.conv1.backward(d)

    # ── Predict ──────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def summary(self):
        print("═" * 42)
        print("   CNN Architecture Summary")
        print("═" * 42)
        for layer in [self.conv1, self.pool1,
                      self.conv2, self.pool2,
                      self.flat,  self.fc1,  self.fc2]:
            print(f"  {layer}")
        print("═" * 42)


# ─────────────────────────────────────────────
# 8.  TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(model: CNN,
          X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray, y_val:   np.ndarray,
          epochs: int = 5, batch_size: int = 64) -> dict:
    """
    Mini-batch SGD training loop.

    Parameters
    ----------
    model      : CNN instance
    X_train    : (N, 1, 28, 28) float32
    y_train    : (N,) int32
    X_val      : validation images
    y_val      : validation labels
    epochs     : number of full passes over the training set
    batch_size : number of samples per mini-batch

    Returns
    -------
    history : dict with lists 'train_loss', 'train_acc', 'val_acc'
    """
    N = X_train.shape[0]
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Shuffle training data
        idx = np.random.permutation(N)
        X_s, y_s = X_train[idx], y_train[idx]

        epoch_loss, correct = 0.0, 0

        for start in range(0, N, batch_size):
            Xb = X_s[start:start + batch_size]
            yb = y_s[start:start + batch_size]

            # Forward
            probs = model.forward(Xb)

            # Loss
            loss, dZ = cross_entropy_loss(probs, yb)
            epoch_loss += loss * Xb.shape[0]
            correct    += (np.argmax(probs, axis=1) == yb).sum()

            # Backward
            model.backward(dZ)

        # ── Validation accuracy ──
        val_preds = []
        for start in range(0, X_val.shape[0], batch_size):
            Xb = X_val[start:start + batch_size]
            val_preds.append(model.predict(Xb))
        val_preds = np.concatenate(val_preds)
        val_acc   = (val_preds == y_val).mean()

        avg_loss  = epoch_loss / N
        train_acc = correct / N

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs}  "
              f"loss={avg_loss:.4f}  "
              f"train_acc={train_acc*100:.2f}%  "
              f"val_acc={val_acc*100:.2f}%  "
              f"({elapsed:.1f}s)")

    return history


# ─────────────────────────────────────────────
# 9.  EVALUATION FUNCTION
# ─────────────────────────────────────────────
def evaluate(model: CNN,
             X_test: np.ndarray, y_test: np.ndarray,
             batch_size: int = 64) -> None:
    """Print overall accuracy and a per-class breakdown."""
    preds = []
    for start in range(0, X_test.shape[0], batch_size):
        Xb = X_test[start:start + batch_size]
        preds.append(model.predict(Xb))
    preds = np.concatenate(preds)

    overall_acc = (preds == y_test).mean()

    print("\n" + "═" * 42)
    print("         Evaluation Report")
    print("═" * 42)
    print(f"  Overall Accuracy : {overall_acc * 100:.2f}%")
    print(f"  Correct          : {(preds == y_test).sum()} / {len(y_test)}")
    print("─" * 42)
    print(f"  {'Class':<16}  {'Acc':>6}  {'N':>5}")
    print("─" * 42)
    for c, name in enumerate(CLASS_NAMES):
        mask = y_test == c
        acc  = (preds[mask] == c).mean()
        print(f"  {name:<16}  {acc*100:>5.1f}%  {mask.sum():>5}")
    print("═" * 42)


# ─────────────────────────────────────────────
# 10. CONFUSION MATRIX (text)
# ─────────────────────────────────────────────
def confusion_matrix_text(model: CNN,
                           X_test: np.ndarray, y_test: np.ndarray,
                           batch_size: int = 64) -> None:
    """Print a 10×10 confusion matrix to stdout."""
    preds = []
    for start in range(0, X_test.shape[0], batch_size):
        preds.append(model.predict(X_test[start:start + batch_size]))
    preds = np.concatenate(preds)

    C = 10
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1

    header = "Pred→   " + "  ".join(f"{i:4d}" for i in range(C))
    print("\nConfusion Matrix (rows = True, cols = Predicted):")
    print(header)
    for i, row in enumerate(cm):
        print(f"True {i:2d}  " + "  ".join(f"{v:4d}" for v in row))


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    print("Loading Fashion-MNIST …")
    X_train, y_train, X_test, y_test = load_fashion_mnist()

    # Use 5 000 training images for a fast demo; remove slice for full run
    QUICK = True
    if QUICK:
        X_train, y_train = X_train[:5000], y_train[:5000]
        X_test,  y_test  = X_test[:1000],  y_test[:1000]
        print("  (Quick mode: 5 000 train / 1 000 test samples)")

    print(f"  Train: {X_train.shape}  Test: {X_test.shape}\n")

    model = CNN(lr=5e-3)
    model.summary()

    print("\nStarting training …\n")
    history = train(
        model,
        X_train, y_train,
        X_test,  y_test,     # using test as val for simplicity
        epochs=5,
        batch_size=32,
    )

    evaluate(model, X_test, y_test)
    confusion_matrix_text(model, X_test, y_test)
