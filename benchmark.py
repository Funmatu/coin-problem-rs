# benchmark.py
import time
import numpy as np
import coin_problem_rs
from numba import njit


# --- The Challenger: Numba (Optimized SIMD) ---
@njit(fastmath=True, cache=True)
def numba_simd(target, max_coins, coins):
    # dp[k][v]
    dp = np.zeros((max_coins + 1, target + 1), dtype=np.int64)
    dp[0, 0] = 1
    for coin in coins:
        for k in range(1, max_coins + 1):
            # SIMD Optimized Slicing
            dp[k, coin:] += dp[k - 1, : target + 1 - coin]

    total = 0
    for k in range(1, max_coins + 1):
        total += dp[k, target]
    return total


def run_bench():
    print("=== Coin-Problem-RS vs Numba Benchmark ===")

    # Test Parameters
    TARGET = 100_000
    MAX_COINS = 1_000
    COINS = np.array([10, 50, 100, 500], dtype=np.int64)

    print(f"Target: {TARGET:,}, MaxCoins: {MAX_COINS:,}, Coins: {COINS}")

    # 1. Numba Warmup & Run
    print("Running Numba...", end="", flush=True)
    numba_simd(100, 10, COINS)  # Warmup
    start = time.time()
    res_numba = numba_simd(TARGET, MAX_COINS, COINS)
    dur_numba = time.time() - start
    print(f" Done. ({dur_numba:.4f}s)")

    # 2. Rust Warmup & Run
    print("Running Coin-Problem-RS (Rust)...", end="", flush=True)
    coin_problem_rs.solve(100, 10, COINS)  # Warmup
    start = time.time()
    res_rust = coin_problem_rs.solve(TARGET, MAX_COINS, COINS)
    dur_rust = time.time() - start
    print(f" Done. ({dur_rust:.4f}s)")

    # Results
    print("\n--- Results ---")
    print(f"Numba: {dur_numba:.6f} sec")
    print(f"Rust : {dur_rust:.6f} sec")

    speedup = dur_numba / dur_rust
    print(f"Speedup: {speedup:.2f}x")

    if res_numba == res_rust:
        print("Consistency: ✅ OK")
    else:
        print(f"Consistency: ❌ FAIL (Numba={res_numba}, Rust={res_rust})")


if __name__ == "__main__":
    run_bench()
