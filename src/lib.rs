// src/lib.rs

// -----------------------------------------------------------------------------
// Core Logic: Transposed Layout + Modulo Parallelism (The "Limit Break" Algo)
// -----------------------------------------------------------------------------

/// コアアルゴリズム: Transposed DP with Unsafe Pointer Arithmetic
///
/// メモリレイアウト: [v][k] (Transposed)
/// - v: 金額 (0..=target)
/// - k: 枚数 (0..=max_coins)
///
/// これにより、内側ループ(k)でのメモリアクセスが連続になり、
/// L1キャッシュヒット率が最大化される。
///
/// Python版では Rayon を用いて「剰余類(v % c)」ごとに並列化する。
#[cfg(not(target_family = "wasm"))]
use rayon::prelude::*;

#[cfg(not(target_family = "wasm"))]
fn solve_transposed_par_core(target: usize, max_coins: usize, coins: &[i64]) -> u64 {
    let stride = max_coins + 1;
    // ゼロ初期化された巨大な配列 (DPテーブル)
    let mut dp = vec![0u64; (target + 1) * stride];
    
    // 初期化: 0円を0枚で作る = 1通り
    dp[0] = 1;

    for &coin in coins {
        let c = coin as usize;
        let dp_ptr = dp.as_mut_ptr() as usize; // Send対策でusize化して渡す

        // 剰余類並列化 (Modulo Parallelism)
        // v と v-c は依存関係があるが、v と v-1 は独立している。
        // したがって、v % c の余りが異なるグループは完全に並列実行可能。
        (0..c).into_par_iter().for_each(|rem| {
            let dp_ptr = dp_ptr as *mut u64;
            let mut v = rem;
            
            // v-c < 0 の領域はスキップ。最初の有効な v は c 以上。
            if v < c { v += c; }

            while v <= target {
                let current_base = v * stride;
                let prev_base = (v - c) * stride;
                
                unsafe {
                    // kループ: メモリ連続アクセス (L1 Cache Resident)
                    for k in 1..=max_coins {
                        let prev_val = *dp_ptr.add(prev_base + k - 1);
                        *dp_ptr.add(current_base + k) += prev_val;
                    }
                }
                v += c;
            }
        });
    }

    // 集計
    let mut total = 0;
    let target_base = target * stride;
    for k in 0..=max_coins {
        total += dp[target_base + k];
    }
    total
}

/// WASM用: シングルスレッド版 Transposed DP
/// 並列化オーバーヘッドがないため、WASM環境ではこちらが最適かつ安全。
fn solve_transposed_seq_core(target: usize, max_coins: usize, coins: &[i64]) -> u64 {
    let stride = max_coins + 1;
    let mut dp = vec![0u64; (target + 1) * stride];
    dp[0] = 1;

    for &coin in coins {
        let c = coin as usize;
        let dp_ptr = dp.as_mut_ptr();

        // 依存関係順(昇順)に処理
        for v in c..=target {
            let current_base = v * stride;
            let prev_base = (v - c) * stride;

            unsafe {
                for k in 1..=max_coins {
                    let prev_val = *dp_ptr.add(prev_base + k - 1);
                    *dp_ptr.add(current_base + k) += prev_val;
                }
            }
        }
    }

    let mut total = 0;
    let target_base = target * stride;
    for k in 0..=max_coins {
        total += dp[target_base + k];
    }
    total
}


// -----------------------------------------------------------------------------
// Module: Python Interface (PyO3)
// -----------------------------------------------------------------------------
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::PyReadonlyArray1;

#[cfg(feature = "python")]
#[pyfunction]
fn solve(target: usize, max_coins: usize, coins: PyReadonlyArray1<i64>) -> PyResult<u64> {
    let coins_slice = coins.as_slice()?;
    
    #[cfg(not(target_family = "wasm"))]
    {
        // Native Python環境では並列版を使用
        Ok(solve_transposed_par_core(target, max_coins, coins_slice))
    }
    #[cfg(target_family = "wasm")]
    {
        // Python on WASM (Pyodide等) の場合はシーケンシャル
        Ok(solve_transposed_seq_core(target, max_coins, coins_slice))
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn coin_problem_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}


// -----------------------------------------------------------------------------
// Module: WebAssembly Interface (wasm-bindgen)
// -----------------------------------------------------------------------------
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn solve_js(target: usize, max_coins: usize, coins: &[i64]) -> u64 {
    // ブラウザ環境では常にシーケンシャル版を使用
    solve_transposed_seq_core(target, max_coins, coins)
}