# Coin Problem: Hyper-Optimized Coin Change Solver

![Rust](https://img.shields.io/badge/Language-Rust-orange.svg)
![Python](https://img.shields.io/badge/Python-PyO3-blue.svg)
![WASM](https://img.shields.io/badge/Web-WASM-yellow.svg)

**Coin-Problem-RS** is a specialized high-performance computing kernel designed to solve the "Coin Change Problem with Cardinality Constraints" (finding the number of ways to pay $N$ using at most $M$ coins).

This project demonstrates how to break the performance limits of Python (even with Numba/SIMD) by leveraging **Rust**, **Unsafe Pointer Arithmetic**, **Transposed Memory Layout**, and **Modulo Parallelism**.

## 🚀 Performance

| Implementation | Time (N=100k, M=1k) | Speedup | Note |
|:---|:---:|:---:|:---|
| **Python (Numpy)** | > 100.0s | 1x | Memory bound & Overhead heavy |
| **Numba (SIMD)** | 1.25s | ~80x | Highly optimized SIMD slicing |
| **Coin-Problem-RS** | **0.76s** | **~130x** | **Modulo Parallelism + Transposed Layout** |

## 🧠 Technical Deep Dive

The core algorithm overcomes the **Memory Wall** and **Data Dependency** issues inherent in dynamic programming.

### 1. The Challenge: Unbounded Knapsack Dependency
The standard DP transition for the coin change problem (unbounded knapsack) is:
$$dp[v] = dp[v] + dp[v-c]$$
This creates a strict data dependency: calculating the value for amount $v$ requires the result of $v-c$ (using the *same* coin). This dependency chain usually prevents parallelization across the amount $v$.

### 2. Solution A: Transposed Memory Layout
Standard DP tables are often shaped `[coins][amount]`. However, we are interested in the cardinality constraint (max coins $M$).
LimitBreakRS uses a **Transposed Layout**: `[amount][max_coins]`.
* **Structure:** `dp[v][k]` stores the ways to make amount $v$ with exactly $k$ coins.
* **Benefit:** The inner loop iterates over $k$ (`1..M`). In the transposed layout, `dp[v][1..M]` occupies contiguous memory addresses. This maximizes **L1 Cache Hits** and allows the CPU to use SIMD instructions efficiently for the vector addition.

### 3. Solution B: Modulo Parallelism (The "Limit Break")
While $dp[v]$ depends on $dp[v-c]$, it does **not** depend on $dp[v-1]$.
The dependencies form $c$ independent chains based on the remainder modulo $c$:
* Chain 0: $0 \to c \to 2c \dots$
* Chain 1: $1 \to 1+c \to 1+2c \dots$

**LimitBreakRS leverages this by parallelizing across remainders.**
Using `Rayon`, we launch threads to handle each remainder group independently. This allows us to utilize all CPU cores without any locks or synchronization, breaking the single-core speed limit of SIMD.

### 4. Unsafe Optimization
To squeeze out the last millisecond, we use `unsafe` Rust to bypass array bounds checking in the hot loop.
```rust
// Hot loop inside the kernel
unsafe {
    for k in 1..=max_coins {
        let prev_val = *dp_ptr.add(prev_base + k - 1);
        *dp_ptr.add(current_base + k) += prev_val;
    }
}

```

## 🛠 Usage

### Python (High Performance Analysis)

Requires `maturin`.

```bash
# Install
pip install maturin
maturin develop --release --features python

# Run
python benchmark.py

```

```python
import coin_problem_rs
import numpy as np

target = 100000
max_coins = 1000
coins = np.array([10, 50, 100, 500], dtype=np.int64)

# Returns result (int)
result = coin_problem_rs.solve(target, max_coins, coins)

```

### WebAssembly (Browser Demo)

Runs in the browser via `wasm-bindgen`. Note that the WASM version runs in **Sequential Mode** (single-threaded) to ensure compatibility with standard browser security headers, but still benefits from the Transposed Layout cache optimization.

```bash
# Build
wasm-pack build --target web --out-dir www/pkg --no-default-features --features wasm

# Serve
cd www
python3 -m http.server

```

## 📂 Project Structure

* `src/lib.rs`: The dual-target Rust kernel.
* `tests/`: Python functional tests.
* `benchmark.py`: Performance comparison script.
* `www/`: Web frontend.

## 📜 License

MIT
